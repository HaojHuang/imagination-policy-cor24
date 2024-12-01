import argparse
from models_stochastic.flow_pvcnn import RectifiedFlow as FlowPvcnn
from models_stochastic.pvcnn import PVCNN
from create_dataset import PCD_Dataset, addLanEmd, VoxelizedPcd, augData
from torch_geometric.transforms import Compose

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import to_o3d_pcd, estimate_rigid_transform, estimate_rigid_transform_open3d, rotation_error
import open3d as o3d
from transform import Transform
from vis_high_d_feature import pcd_hd_vis

parser = argparse.ArgumentParser(description='train_flow')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--save_dir', type=str, default='./checkpoints/ycb')
parser.add_argument('--n', type=int, default=5000)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--add_noise', action='store_true', default=False)
parser.add_argument('--model', type=str, default='pn')
parser.add_argument('--aug1', action='store_true', default=False)
parser.add_argument('--aug2', action='store_true', default=False)
parser.add_argument('--bi', action='store_true', default=False)
parser.add_argument('--use_lan', action='store_true', default=False)

parser.add_argument('--vox', action='store_true', default=True)
parser.add_argument('--disp', action='store_true', default=False)
parser.add_argument('--plot', action='store_true', default=False)
parser.add_argument('--icp', action='store_true', default=False)
parser.add_argument('--randsample', action='store_true', default=False)
parser.add_argument('--use_color', action='store_true', default=False)

# parser.add_argument('--save_traj', action='store_true', default=False)
args = parser.parse_args()
import numpy as np


def test(args):
    if args.device >-1:
        device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device('cuda:{}'.format(args.device))
    else:
        device = torch.device('cpu')
    
    if args.model =='pvcnn':
        flower = FlowPvcnn(model=PVCNN(use_color=args.use_color), device=args.device, lr=2e-4)

    save_file = '{}.pt'.format(args.n)
    
    para_path = os.path.join(args.save_dir,save_file)
    

    if args.plot:
        save_curve = '{}.npy'.format(args.n)
        curve_path = os.path.join(args.save_dir, save_curve)
        curve = np.load(curve_path)
        plt.plot(np.linspace(0, len(curve)-1, len(curve)), curve)
        plt.title('Training Loss Curve')
        plt.show()
    
    
    flower.load(para_path)
    tr_dataset = PCD_Dataset(root='./data', pre_transform=Compose([addLanEmd(),VoxelizedPcd(voxel_size=0.004)]), transform=augData(aug1=args.aug1, aug2=args.aug2, randsample=args.randsample), train=True)
    # tr_dataset = tr_dataset[:4]
    rot_errors = []
    trans_errors = []
    # for i in range(len(tr_dataset)):
    for i in range(50):
        # torch.manual_seed(i+1)
        idx = i % len(tr_dataset)
        # idx = i % 1
        # idx=1
        # original, original_color, mask, sample, sample_color, sample_mask, aT, id, lan, lan_emd, src_n, tgt_n = train_set[idx]
        # if not args.use_lan:
        #     lan_emd = None
        # else:
        #     lan_emd = lan_emd.to(device)
        # src, tgt, raw = original
        # src, tgt, raw, mask = src.to(device), tgt.to(device), raw.to(device), mask.to(device)
        # print(src.shape, tgt.shape, raw.shape, mask.shape)

        data = tr_dataset[idx]
        print(idx, data['lan'])
        if not args.use_lan:
            lan_emd = None
        else:
            lan_emd = data['lan_emd'].to(device)
        
        pa, pb, pab, mask = data['pa_ds'].to(device), data['pb_ds'].to(device), data['pab_ds'].to(device), data['mask_ds'].to(device)        
        p0 = torch.cat([pa, pb], dim=0)
        mask_input = mask
        pa_color, pb_color = data['pa_ds_color'].to(device), data['pb_ds_color'].to(device)
        x0_color = torch.cat([pa_color, pb_color], dim=0)
        # print(mask)
        # print(len(pa),len(pb))
        # o3d.visualization.draw_geometries([to_o3d_pcd(pa)])
        # o3d.visualization.draw_geometries([to_o3d_pcd(pb)])
        # id = 1
        traj,pdf = flower.sample_ode(p0=p0, mask=mask_input,color=x0_color, sample_ode_steps=1000, id=data['id'].clone(), bi=args.bi,lan_emd=lan_emd)
        
        if args.disp:
            # print(pdf.shape, p0.shape)
            pdf = pdf.cpu().numpy()
            # print(pdf)
            # print(pdf.sum())
            pcd_hd_vis(points=p0[mask==1].cpu().numpy(),features=pdf[mask.cpu().numpy()==1],model='tsne')
            pcd_hd_vis(points=p0[mask==1].cpu().numpy(),features=pdf[mask.cpu().numpy()==1],model='pca')
        # print(traj[-1].shape)
        # print(delta.shape)
        pb_input_points = p0[mask_input.cpu().numpy()==1.].cpu().numpy()
        pb_output_points = traj[-1][mask_input.cpu().numpy()==1.]
        Tb = estimate_rigid_transform_open3d(pb_input_points, pb_output_points)
        
        if args.icp:
            reg_p2p = o3d.pipelines.registration.registration_icp(
            to_o3d_pcd(pb_input_points), to_o3d_pcd(pb_output_points), 0.01, Tb,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
            print(reg_p2p)
            Tb = reg_p2p.transformation
            # o3d.visualization.draw_geometries([to_o3d_pcd(pb_input_points,grey=True).transform(reg_p2p.transformation), to_o3d_pcd(pb_output_points)], window_name="ICP")
        pa_input_points = p0[mask_input.cpu().numpy()==0.].cpu().numpy()
        pa_output_points = traj[-1][mask_input.cpu().numpy()==0.]
        Ta = estimate_rigid_transform_open3d(pa_input_points,pa_output_points)
        Tab = Ta.T @ Tb

        if args.disp:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

            o3d.visualization.draw_geometries([to_o3d_pcd(pb,grey=True), to_o3d_pcd(pa,grey=True), mesh_frame], window_name="input: pa and pb")
            
            c = np.zeros_like(traj[-1])
            c[:,:] = np.asarray([255, 165, 0])/255
            if not args.bi:
                c[mask_input.cpu().numpy()==0.] = pa_color.cpu().numpy()
            o3d.visualization.draw_geometries([to_o3d_pcd(traj[-1],colors=c)], window_name="output: generated pts", width=800, height=800, )

            o3d.visualization.draw_geometries([to_o3d_pcd(traj[-1]), to_o3d_pcd(pab, grey=True), mesh_frame], window_name="generated pts vs pab", )

            o3d.visualization.draw_geometries([to_o3d_pcd(pb,red=True).transform(Tb),to_o3d_pcd(pa,red=True).transform(Ta)],window_name="calculate Ta Tb")
            
            o3d.visualization.draw_geometries([to_o3d_pcd(traj[-1]), to_o3d_pcd(pb,red=True).transform(Tab),to_o3d_pcd(pa,red=True), mesh_frame], window_name="measure correspdonse")
            
            o3d.visualization.draw_geometries([to_o3d_pcd(pb,red=True).transform(Tab),to_o3d_pcd(pa,red=True),
                                            to_o3d_pcd(pb,grey=True).transform(data['Tab']),], window_name="measure Ta Tb")
        
        Rab = torch.from_numpy(Tab[:3,:3]).float()
        rot_error = rotation_error(Rab.unsqueeze(dim=0), data['Tab'][:3,:3].unsqueeze(dim=0))
        trans_error = np.sqrt(((Tab[:3,-1] - data['Tab'][:3,-1].numpy())**2).sum())
        rot_error = rot_error.item()*180/np.pi
        trans_error = trans_error*100
        rot_errors.append(rot_error)
        trans_errors.append(trans_error)
        traj.insert(0, p0.cpu().numpy())
        print('{} trans error {} (cm) rot error {} (degree)'.format(data['lan'],trans_error,rot_error))
        
        # save the generation trajectory
        traj_fold = './traj'
        if not os.path.exists(traj_fold):
            os.mkdir(traj_fold)
        np.save(os.path.join(traj_fold,'traj.npy'),np.asarray(traj))
    
    trans_errors_avg = np.asarray(trans_errors).mean()
    rot_errors_avg = np.asarray(rot_errors).mean()
    trans_errors_min = np.asarray(trans_errors).min()
    rot_errors_min = np.asarray(rot_errors).min()
    trans_errors_max = np.asarray(trans_errors).max()
    rot_errors_max = np.asarray(rot_errors).max()
    print(trans_errors, rot_errors)
    print('{}: min trans_error {} (cm), min rot_error {} (degree)'.format(len(trans_errors), trans_errors_min, rot_errors_min))
    print('{}: mean trans_error {} (cm), mean rot_error {} (degree)'.format(len(trans_errors), trans_errors_avg, rot_errors_avg))
    print('{}: max trans_error {} (cm), max rot_error {} (degree)'.format(len(trans_errors), trans_errors_max, rot_errors_max))

    


if __name__ == '__main__':
    test(args)