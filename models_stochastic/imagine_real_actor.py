import sys
import os
sys.path.append('../')
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from flow_pvcnn import RectifiedFlow as FlowPvcnn
from pvcnn import PVCNN
import numpy as np
import torch
import open3d as o3d
from utils import to_o3d_pcd,estimate_rigid_transform_open3d
from create_rlbench_pick import rand_sample
import clip_revised as clip_revised
from clip_revised.clip import build_model,tokenize
from transform import Transform, Rotation
import copy

def rand_sample(pts, number=1024):

    if len(pts)>=number:
        rand_index = torch.randperm(len(pts))[:number]
    else:
        rand_index = torch.randint(low=0,high=len(pts),size=(number-len(pts),))
        rand_index = torch.cat((torch.arange(0,len(pts)),rand_index), dim=0)
    
    assert len(rand_index)==number

    return rand_index

class ImagineActor():
    def __init__(self, pick_n=1e4, place_n = 1e4, device = 0, sample_ode_steps=1000, ds_size=0.003, rands=3096, gripper_pcd=None, use_color=False,):
        torch.cuda.set_device('cuda:{}'.format(device))
        #load picker
        self.picker = FlowPvcnn(model=PVCNN(use_color=use_color), device=device)
        pick_checkpoint = './checkpoints/real/pick/{}.pt'.format(pick_n)
        self.picker.load(pick_checkpoint)
        self.voxel_size = ds_size
        self.rands = rands
        self.sample_ode_steps = sample_ode_steps
        
        # load placer
        self.placer= FlowPvcnn(model=PVCNN(use_color=use_color), device=device)
        place_checkpoint = './checkpoints/real/place/{}.pt'.format(place_n)
        self.placer.load(place_checkpoint)
        ##
        # add this to dataset
        self.device = self.picker.device
        clip_model_name = "ViT-B/32"
        model, _ = clip_revised.load(clip_model_name, device=device)
        self.clip_vit32 = build_model(model.state_dict()).to(device)
        del model
        self.gripper_pcd = gripper_pcd
        self.use_color = use_color

    def preprocess_gripper(self,):
        # downsample
        gripper = copy.deepcopy(self.gripper_pcd)
        g_sample = gripper.voxel_down_sample(voxel_size=self.voxel_size)
        g_points = np.asarray(g_sample.points)
        g_colors = np.asarray(g_sample.colors)

        # to torch tensor
        g_points = torch.from_numpy(g_points).float()
        g_colors = torch.from_numpy(g_colors).float()
        
        # center
        g_mean = torch.mean(g_points,dim=0,keepdim=True)
        g_points = g_points-g_mean

        # randsample
        if self.rands>0:
            
            g_rand_index = rand_sample(g_points, number=self.rands)
            g_points = g_points[g_rand_index,:]
            g_colors = g_colors[g_rand_index,:]
        g_points = g_points.to(self.device)
        g_colors = g_colors.to(self.device)

        return g_points, g_colors


    def act(self, pa_pcd=None, pb_pcd=None, pick_lan=None, preplace_lan=None,place_lan=None, plot_action=False):
        
        pa_ds_points, pa_ds_colors, pb_ds_points, pb_ds_colors, pick_emd, preplace_emd, place_emd,pa_mean,pb_mean = self.preprocess_data( pa_pcd, pb_pcd, pick_lan, preplace_lan,place_lan)
        
        g_points, g_colors = self.preprocess_gripper()
        
        # print(g_points.shape,pa_ds_points.shape, pb_ds_points.shape)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        # print(pa_ds_points.shape, g_points.shape, pb_ds_points.shape)
        # o3d.visualization.draw_geometries([mesh_frame, to_o3d_pcd(g_points, g_colors)])
        # o3d.visualization.draw_geometries([mesh_frame,to_o3d_pcd(pa_ds_points,pa_ds_colors), to_o3d_pcd(pb_ds_points,pb_ds_colors)])
        # o3d.visualization.draw_geometries([pa_pcd, pb_pcd])
        
        ## calculate pick pose
        print(pick_lan)
        p0 = torch.cat([g_points, pb_ds_points], dim=0)
        p0_color = torch.cat([g_colors, pb_ds_colors], dim=0)
        mask = torch.cat((torch.zeros(len(g_points)),torch.ones(len(pb_ds_points))),dim=0).to(self.device)
        pick_traj,pick_pdf = self.picker.sample_ode(p0=p0, mask=mask,color=p0_color, sample_ode_steps=self.sample_ode_steps, id='pick', bi=False,lan_emd=pick_emd, scale=0.1)
        pick_pb_output_points = pick_traj[-1][mask.cpu().numpy()==1.]
        pick_pose = estimate_rigid_transform_open3d(pb_ds_points.cpu().numpy(), pick_pb_output_points)
        pick_pose = Transform.from_matrix(pick_pose)
        
        plot_action = True
        if plot_action:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([mesh_frame, to_o3d_pcd(torch.from_numpy(pick_traj[-1])[mask.cpu()==1], orange=True), to_o3d_pcd(torch.from_numpy(pick_traj[-1])[mask.cpu()==0], g_colors)])
            o3d.visualization.draw_geometries([mesh_frame, to_o3d_pcd(pb_ds_points,red=True).transform(pick_pose.as_matrix()),to_o3d_pcd(g_points,g_colors)],window_name="pick")
            
        # calculate the pick pose in world frame
        gripper_center_off = Transform(rotation=Rotation.identity(), translation=[0.000,0.,0.1])
        pb_center_off = Transform(rotation=Rotation.identity(), translation=pb_mean.cpu().numpy()[0])
        
        o3d.visualization.draw_geometries([pa_pcd, to_o3d_pcd(pb_ds_points,grey=True).transform(pb_center_off.as_matrix()),to_o3d_pcd(g_points,grey=True).transform((pb_center_off * (pick_pose.inverse())).as_matrix())],window_name="pick")
        pick_pose =pb_center_off*pick_pose.inverse()*gripper_center_off
        o3d.visualization.draw_geometries([mesh_frame.transform(pick_pose.as_matrix()), copy.deepcopy(pb_pcd),],window_name='pick')

        ##calculate the preplace
        print(preplace_lan)
        p0 = torch.cat([pa_ds_points, pb_ds_points], dim=0)
        p0_color = torch.cat([pa_ds_colors, pb_ds_colors], dim=0)
        mask = torch.cat((torch.zeros(len(pa_ds_points)),torch.ones(len(pb_ds_points))),dim=0).to(self.device)
        preplace_traj,preplace_pdf = self.placer.sample_ode(p0=p0, mask=mask, color=p0_color, sample_ode_steps=self.sample_ode_steps, id='preplace', bi=True, lan_emd=preplace_emd, scale=0.1)
        preplace_pa_output_points = preplace_traj[-1][mask.cpu().numpy()==0.]
        preplace_pb_output_points = preplace_traj[-1][mask.cpu().numpy()==1.]

        Ta = estimate_rigid_transform_open3d(pa_ds_points.cpu().numpy(), preplace_pa_output_points)
        Tb = estimate_rigid_transform_open3d(pb_ds_points.cpu().numpy(), preplace_pb_output_points)
        Tab_preplace = Ta.T @ Tb
        preplace_pose = Transform.from_matrix(Tab_preplace)
        ## start here
        if plot_action:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([mesh_frame, to_o3d_pcd(pb_ds_points,red=True).transform(preplace_pose.as_matrix()),to_o3d_pcd(pa_ds_points,red=True)],window_name="preplace")
            o3d.visualization.draw_geometries([to_o3d_pcd(preplace_traj[-1], orange=True)])
        # calculate the preplace pose in world frame
        pa_center_off = Transform(rotation=Rotation.identity(), translation=pa_mean.cpu().numpy()[0])
        pb_center_off = Transform(rotation=Rotation.identity(), translation=pb_mean.cpu().numpy()[0])
        preplace_pose_delta = pa_center_off*preplace_pose* (pb_center_off.inverse())

        o3d.visualization.draw_geometries([pa_pcd, to_o3d_pcd(pb_ds_points.cpu()+pb_mean.cpu(), grey=True).transform(preplace_pose_delta.as_matrix())],window_name='preplace')
        preplace_pose =  preplace_pose_delta * pick_pose
        o3d.visualization.draw_geometries([mesh_frame.transform(preplace_pose.as_matrix()), pa_pcd, pb_pcd],window_name='preplace')

        ##calculate the place
        print(place_lan)
        p0 = torch.cat([pa_ds_points, pb_ds_points], dim=0)
        p0_color = torch.cat([pa_ds_colors, pb_ds_colors], dim=0)
        mask = torch.cat((torch.zeros(len(pa_ds_points)),torch.ones(len(pb_ds_points))),dim=0).to(self.device)
        place_traj, place_pdf = self.placer.sample_ode(p0=p0, mask=mask, color=p0_color, sample_ode_steps=self.sample_ode_steps, id='place', bi=True, lan_emd=place_emd, scale=0.1)
        place_pa_output_points = place_traj[-1][mask.cpu().numpy()==0.]
        place_pb_output_points = place_traj[-1][mask.cpu().numpy()==1.]

        Ta = estimate_rigid_transform_open3d(pa_ds_points.cpu().numpy(), place_pa_output_points)
        Tb = estimate_rigid_transform_open3d(pb_ds_points.cpu().numpy(), place_pb_output_points)
        Tab_place = Ta.T @ Tb
        place_pose = Transform.from_matrix(Tab_place)
        ## start here
        if plot_action:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([mesh_frame, to_o3d_pcd(pb_ds_points,red=True).transform(place_pose.as_matrix()),to_o3d_pcd(pa_ds_points,red=True)],window_name="place")
            o3d.visualization.draw_geometries([to_o3d_pcd(place_traj[-1],orange=True)])
        # calculate the preplace pose in world frame
        pa_center_off = Transform(rotation=Rotation.identity(), translation=pa_mean.cpu().numpy()[0])
        pb_center_off = Transform(rotation=Rotation.identity(), translation=pb_mean.cpu().numpy()[0])
        place_pose_delta = pa_center_off*place_pose* (pb_center_off.inverse())
        
        o3d.visualization.draw_geometries([pa_pcd, to_o3d_pcd(pb_ds_points.cpu()+pb_mean.cpu(),grey=True).transform(place_pose_delta.as_matrix())],window_name='place')
        place_pose =  place_pose_delta * pick_pose
        o3d.visualization.draw_geometries([mesh_frame.transform(place_pose.as_matrix()),pa_pcd, pb_pcd],window_name='place')

        return pick_pose, preplace_pose, place_pose



    def preprocess_data(self, pa_pcd=None, pb_pcd=None, pick_lan=None, preplace_lan=None,place_lan=None,):
        # downsample
        pa_sample = pa_pcd.voxel_down_sample(voxel_size=self.voxel_size)
        pa_ds_points = np.asarray(pa_sample.points)
        pa_ds_colors = np.asarray(pa_sample.colors)

        pb_sample = pb_pcd.voxel_down_sample(voxel_size=self.voxel_size)
        pb_ds_points = np.asarray(pb_sample.points)
        pb_ds_colors = np.asarray(pb_sample.colors)

        # to torch tensor
        pa_ds_points = torch.from_numpy(pa_ds_points).float()
        pb_ds_points = torch.from_numpy(pb_ds_points).float()

        pa_ds_colors = torch.from_numpy(pa_ds_colors).float()
        pb_ds_colors = torch.from_numpy(pb_ds_colors).float()
        
        # center
        pa_ds_mean = torch.mean(pa_ds_points,dim=0,keepdim=True)
        pb_ds_mean = torch.mean(pb_ds_points,dim=0,keepdim=True)

        pa_ds_points = pa_ds_points-pa_ds_mean
        pb_ds_points = pb_ds_points-pb_ds_mean

        # randsample
        if self.rands>0:
            
            pa_rand_index = rand_sample(pa_ds_points, number=self.rands)
            pa_ds_points = pa_ds_points[pa_rand_index,:]
            pa_ds_colors = pa_ds_colors[pa_rand_index,:]

            pb_rand_index = rand_sample(pb_ds_points, number=self.rands)
            pb_ds_points = pb_ds_points[pb_rand_index,:]
            pb_ds_colors = pb_ds_colors[pb_rand_index,:]

        pa_ds_points = pa_ds_points.to(self.device)
        pa_ds_colors = pa_ds_colors.to(self.device)
        pb_ds_points = pb_ds_points.to(self.device)
        pb_ds_colors = pb_ds_colors.to(self.device)

        # encode lan
        pick_emd = self.encode_lan(pick_lan)
        preplace_emd = self.encode_lan(preplace_lan)
        place_emd = self.encode_lan(place_lan)

        return pa_ds_points, pa_ds_colors, pb_ds_points, pb_ds_colors, pick_emd, preplace_emd, place_emd, pa_ds_mean, pb_ds_mean


    
    def encode_lan(self,lan):
        with torch.no_grad():
            tokens = tokenize([lan]).to(self.device)
            text_feat, _ = self.clip_vit32.encode_text_with_embeddings(tokens)
        lan_emd = text_feat
        return lan_emd.float()