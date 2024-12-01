import torch_geometric
from torch_geometric.data import Data, Batch, DataLoader, InMemoryDataset
from typing import Optional, Callable, List
import os
import torch
import pickle
import models_stochastic.clip_revised as clip_revised
from models_stochastic.clip_revised.clip import build_model,tokenize
import open3d as o3d
import numpy as np
from transform import Transform, Rotation
from torch_geometric.transforms import Compose

def to_o3d_pcd(xyz, colors=None):
    """
    Convert array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def rand_sample(pts, number=1024):

    if len(pts)>=number:
        rand_index = torch.randperm(len(pts))[:number]
    else:
        rand_index = torch.randint(low=0,high=len(pts),size=(number-len(pts),))
        rand_index = torch.cat((torch.arange(0,len(pts)),rand_index), dim=0)
    
    assert len(rand_index)==number

    return rand_index


class RLBbench_Pick_Dataset(InMemoryDataset):
    def __init__(self, root: str, train: bool = True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.pcd_path = root
        self.task_list = ['phone_on_base', 'stack_wine', 'plug_charger_in_power_supply', 'put_knife_in_knife_block', 
                          'put_plate_in_colored_dish_rack', 'put_toilet_roll_on_stand' ]
        # self.task_list = ['phone_on_base']
        self.voxel_size = 0.004
        self.demos = 10
        super(RLBbench_Pick_Dataset,self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self) -> str:
        return 'data'

    @property
    def processed_file_names(self) -> List[str]:
        return ['training.pt', 'test.pt']

    def download(self):
        return
    
    def process(self,):
        
        data_list = []
        gripper = pickle.load(open(os.path.join('./data', 'gripper.pkl'), 'rb'))
        
        for task_name in self.task_list:
            num_demo = 0
            for fname in sorted(os.listdir(os.path.join('./rlbench_data_processed',task_name,'dict'))):
                
                # collect only 10 demos
                if num_demo == self.demos:
                    break
                num_demo +=1
                datapoint = pickle.load(open(os.path.join('./rlbench_data_processed',task_name ,'dict',fname), 'rb'))
                # gripper
                pa = torch.from_numpy(gripper['points']).to(torch.float32)
                pa_color = torch.from_numpy(gripper['colors']).to(torch.float32)
                
                # downsample pa
                pa_pcd = to_o3d_pcd(pa, pa_color)
                pa_sample = pa_pcd.voxel_down_sample(voxel_size=self.voxel_size)
                pa = torch.from_numpy(np.asarray(pa_sample.points)).float()
                pa_color = torch.from_numpy(np.asarray(pa_sample.colors)).float()

                pa_mean = torch.mean(pa,dim=0,keepdims=True)
                pa  = pa - pa_mean
                Ta = np.eye(4)
                Ta =  torch.from_numpy(Ta).to(torch.float32)
                # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                # o3d.visualization.draw_geometries([mesh_frame,to_o3d_pcd(pa)])
                
                # object B 
                gripper_center_off = Transform(rotation=Rotation.identity(), translation=[0.000,0.,0.1])
                pb = datapoint['obj_points'].to(torch.float32)
                pb_color = datapoint['obj_colors'].to(torch.float32)
                pick_pose = Transform.from_matrix(datapoint['pick_pose'])
                # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                # pick_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(pick_pose.as_matrix())
                # o3d.visualization.draw_geometries([pick_frame,to_o3d_pcd(pb),mesh_frame])
                
                # downsample_pb
                pb_pcd = to_o3d_pcd(pb, pb_color)
                pb_sample = pb_pcd.voxel_down_sample(voxel_size=self.voxel_size)
                pb = torch.from_numpy(np.asarray(pb_sample.points)).float()
                pb_color = torch.from_numpy(np.asarray(pb_sample.colors)).float()


                pb_mean = torch.mean(pb,dim=0,keepdim=True)
                pb = pb - pb_mean
                pb_center_off = Transform(rotation=Rotation.identity(), translation=pb_mean[0].numpy())
                Tb = gripper_center_off * pick_pose.inverse() * pb_center_off
                
                mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                # o3d.visualization.draw_geometries([mesh_frame, to_o3d_pcd(pa, pa_color), to_o3d_pcd(pb,pb_color).transform(Tb.as_matrix()), to_o3d_pcd(pb)])
                
                # print(pa.shape, pb.shape)

                pb_transformed = to_o3d_pcd(pb,pb_color).transform(Tb.as_matrix())
                pab = torch.cat((pa, torch.from_numpy(np.asarray(pb_transformed.points)).to(torch.float)),dim=0)
                pab_color = torch.cat((pa_color,torch.from_numpy(np.asarray(pb_transformed.colors)).to(torch.float)),dim=0)
                
                # o3d.visualization.draw_geometries([mesh_frame, to_o3d_pcd(pab, pab_color)])
                # action description
                lan = datapoint['lan_pick']
                # print(lan)
                Tb = torch.from_numpy(Tb.as_matrix()).float()
                
                data = Data(pa=pa, pa_color=pa_color, pa_ds=pa, pa_ds_color =pa_color, Ta=Ta, 
                            pb=pb, pb_color=pb_color, pb_ds=pb, pb_ds_color = pb_color, Tb=Tb, 
                            pab=pab, pab_color=pab_color, pab_ds = pab, pab_ds_color = pab_color,
                            lan=lan, id='pick')
                data_list.append(data)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        total_num = len(data_list)
        torch.save(self.collate(data_list[:int(total_num)]), self.processed_paths[0])
        torch.save(self.collate(data_list[:int(total_num)]), self.processed_paths[1])



class addLanEmd:
    def __init__(self) -> None:
        pass

    def __call__(self, data):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # add this to dataset
        clip_model_name = "ViT-B/32"
        model, _ = clip_revised.load(clip_model_name, device=device)
        clip_vit32 = build_model(model.state_dict()).to(device)
        del model
        lan = data['lan']
        with torch.no_grad():
            tokens = tokenize([lan]).to(device)
            text_feat, _ = clip_vit32.encode_text_with_embeddings(tokens)
        data['lan_emd'] = text_feat.cpu().float()
        return data



# class VoxelizedPcd:
#     def __init__(self, voxel_size=0.003):
#         self.voxel_size = voxel_size
    
#     def __call__(self, data):
#         # process pa
#         pa_pcd = to_o3d_pcd(data['pa'], data['pa_color'])
#         pa_sample = pa_pcd.voxel_down_sample(voxel_size=self.voxel_size)
#         pa_ds_points = np.asarray(pa_sample.points)
#         pa_ds_colors = np.asarray(pa_sample.colors)
                
#         # process pb
#         pb_pcd = to_o3d_pcd(data['pb'], data['pb_color'])
#         pb_sample = pb_pcd.voxel_down_sample(voxel_size=self.voxel_size)
#         pb_ds_points = np.asarray(pb_sample.points)
#         pb_ds_colors = np.asarray(pb_sample.colors)

#         # process pab
#         pa_transform = Transform.from_matrix(data['Ta'].numpy())
#         pb_transform = Transform.from_matrix(data['Tb'].numpy())
        
#         pa_ds_points_t = pa_transform.transform_point(pa_ds_points)
#         pb_ds_points_t = pb_transform.transform_point(pb_ds_points)

#         pab_ds_points = np.concatenate((pa_ds_points_t, pb_ds_points_t),axis=0)
#         pab_ds_colors = np.concatenate((pa_ds_colors, pb_ds_colors),axis=0)
#         # print(pab_ds_points.shape)

#         data['pa_ds'] = torch.from_numpy(pa_ds_points).to(torch.float32)
#         data['pa_ds_color'] = torch.from_numpy(pa_ds_colors).to(torch.float32)
#         data['pb_ds'] = torch.from_numpy(pb_ds_points).to(torch.float32)
#         data['pb_ds_color'] = torch.from_numpy(pb_ds_colors).to(torch.float32)
#         data['pab_ds'] = torch.from_numpy(pab_ds_points).to(torch.float32)
#         data['pab_ds_color'] = torch.from_numpy(pab_ds_colors).to(torch.float32)

#         return data



class augData:
    
    def __init__(self, aug1=False, aug2=False, add_noise=False, randsample=False):
        # self.aug0 = aug0
        self.aug1 = aug1
        self.aug2 = aug2
        self.add_noise = add_noise
        self.randsample = randsample
        # self.vox = vox
        # self.center_pab = center_pab
    
    def __call__(self, data):
        if self.aug1:
            aT = Transform(rotation=Rotation.random(), translation=np.zeros(3)).as_matrix()
        else:
            aT = np.eye(4)
        aT = torch.from_numpy(aT).to(torch.float32)
        aR = aT[:3, :3]

        if self.aug2:
            bT = Transform(rotation=Rotation.random(), translation=np.zeros(3)).as_matrix()
        else:
            bT = np.eye(4)

        bT = torch.from_numpy(bT).to(torch.float32)
        bR = bT[:3, :3]

        pa = data['pa']
        pa_ds = data['pa_ds']
        pa_ds_color = data['pa_ds_color']

        pb = data['pb']
        pb_ds = data['pb_ds']
        pb_ds_color = data['pb_ds_color']

        pab_ds = data['pab_ds']
        pab = data['pab']
        
        if self.randsample:
            # rand sample 1024 points for each object
            mask_ds = torch.cat((torch.zeros(len(pa_ds)), torch.ones(len(pb_ds))), dim=0)
            pa_rand_index = rand_sample(pa_ds, number=2048)
            pa_ds = pa_ds[pa_rand_index,:]
            pa_ds_color = pa_ds_color[pa_rand_index,:]


            pb_rand_index = rand_sample(pb_ds, number=2048)
            pb_ds = pb_ds[pb_rand_index,:]
            pb_ds_color = pb_ds_color[pb_rand_index,:]

            pab_ds_a = pab_ds[mask_ds==0,:][pa_rand_index,:].clone()
            pab_ds_b = pab_ds[mask_ds==1,:][pb_rand_index,:].clone()
            pab_ds = torch.cat((pab_ds_a,pab_ds_b),dim=0)
            pab_ds_color = torch.cat((pa_ds_color,pb_ds_color),dim=0)


        # scale = True
        # if scale:
        #     print('scale')
        #     pb_ds_mean = torch.mean(pb_ds,axis=0,keepdims=True)
        #     pb_ds = pb_ds - pb_ds_mean
        #     pb_ds = pb_ds * 0.1
        #     pb_ds = pb_ds + pb_ds_mean

        # center
        pa_mean = torch.mean(pa,dim=0,keepdim=True)
        pb_mean = torch.mean(pb,dim=0,keepdim=True)
        pa_ds_mean = torch.mean(pa_ds,dim=0,keepdim=True)
        pb_ds_mean = torch.mean(pb_ds,dim=0,keepdim=True)
        # print(pa_mean,pb_mean)


        
        # center
        # pa = pa -pa_mean
        # pb = pb - pb_mean
        # pab = data['pab'] - pa_mean # assume Pa is contained in Pab

        # pa_ds = pa_ds -pa_ds_mean
        # pb_ds = pb_ds - pb_ds_mean # to get a fixed mean across different object should be good
        # pab_ds = pab_ds - pa_ds_mean # assume Pa is contained in Pab
        


        # so3 aug
        pa = pa @ aR.T # (R @ (3,1)).T == (1, 3) @ R.T 
        pa_ds = pa_ds @ aR.T

        pb = pb @ bR.T
        pb_ds = pb_ds @ bR.T

        # # go back
        # pa = pa + pa_mean
        # pb = pb + pb_mean
        # pab = pab + pa_mean

        # pa_ds = pa_ds + pa_ds_mean
        # pb_ds = pb_ds + pb_ds_mean 
        # pab_ds = pab_ds + pa_ds_mean

        # pab_ds_mean = torch.mean(pab_ds,dim=0, keepdim=True)
        # pab_ds = pab_ds - pab_ds_mean
        # pab_ds = pab_ds @ aR.T
        # pab_ds = pab_ds + pab_ds_mean

        # print(data['Tb'])
        # update Ta Tb
        Tab = (aT@data['Ta'].T) @ (data['Tb']@ bT.T) #how to arrange the object to the placement, please note Ta is Identity
        
        pb_target = pb.clone()
        pb_target = pb_target @ (Tab[:3,:3].T) + Tab[:3, -1]
        data['pb_target'] = pb_target

        pb_ds_target = pb_ds.clone()
        pb_ds_target = pb_ds_target @ (Tab[:3,:3].T) + Tab[:3, -1]
        data['pb_ds_target'] = pb_ds_target

        data['Tab'] = Tab
        data['pa'] = pa
        data['pb'] = pb
        data['pab'] = pab
        data['pa_ds'] = pa_ds
        data['pb_ds'] = pb_ds
        data['pab_ds'] = pab_ds
        data['pa_ds_color'] = pa_ds_color
        data['pb_ds_color'] = pb_ds_color
        if self.randsample:
            data['pab_ds_color'] = pab_ds_color

        data['mask'] = torch.cat((torch.zeros(len(pa)), torch.ones(len(pb))), dim=0)
        data['mask_ds'] = torch.cat((torch.zeros(len(pa_ds)), torch.ones(len(pb_ds))), dim=0)
        
        return data
    

# tr_dataset = RLBbench_Pick_Dataset(root='./rlbench_data_processed/pick', pre_transform=Compose([addLanEmd(),]), transform=augData(aug1=True, aug2=True, randsample=True), train=True)

# for data in tr_dataset:
#     print(data)

# for i in range(7):
#     data = tr_dataset[i]
#     pa = to_o3d_pcd(data['pa'],data['pa_color'])
#     pb = to_o3d_pcd(data['pb'],data['pb_color'])
#     pab = to_o3d_pcd(data['pab'],data['pab_color'])
#     Tab = data['Tab'].numpy()
#     mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

#     o3d.visualization.draw_geometries([to_o3d_pcd(data['pa_ds'],data['pa_ds_color']), to_o3d_pcd(data['pb_ds_target'],data['pb_ds_color']), mesh_frame], width=1000, height=800)
    
#     o3d.visualization.draw_geometries([pa, pb, pab, mesh_frame], width=1000, height=800)
#     o3d.visualization.draw_geometries([pa, pb.transform(Tab), mesh_frame], width=1000, height=800)

#     pa_ds = to_o3d_pcd(data['pa_ds'],data['pa_ds_color'])
#     pb_ds = to_o3d_pcd(data['pb_ds'],data['pb_ds_color'])
#     pab_ds = to_o3d_pcd(data['pab_ds'],data['pab_ds_color'])
#     o3d.visualization.draw_geometries([pa_ds, pb_ds, pab_ds, mesh_frame], width=1000, height=800)
#     o3d.visualization.draw_geometries([pa_ds, pb_ds.transform(Tab), mesh_frame], width=1000, height=800)