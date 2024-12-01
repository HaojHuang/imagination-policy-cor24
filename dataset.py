import numpy as np
import open3d as o3d
import torch
from torch.utils import data
from utils import square_distance, to_o3d_pcd, generate_transform, FarthestSamplerTorch
from transform import Transform, Rotation
import os
import pickle


def sample_and_transform(src, tgt, raw, src_color, tgt_color, raw_color, aug1=False, aug2=False, add_noise = False, sample_n=128*4, center_raw=False):
    
    if sample_n >0:
        fps_torch = FarthestSamplerTorch()

    if aug1:
        aT = generate_transform()
    else:
        aT = np.eye(4)[:3,:] # 3 times 4 rotation matrix
    
    # R, t = aT[:, :3].T, aT[:, :3].T @ -aT[:, 3:]
    R = aT[:, :3]

    if aug2:
        aT2 = generate_transform()
    else:
        aT2 = np.eye(4)[:3,:] # 3 times 4 rotation matrix
    
    R2 = aT2[:, :3]

    R = torch.from_numpy(R).float()
    R2 = torch.from_numpy(R2).float()
    src = torch.from_numpy(src).float()
    tgt = torch.from_numpy(tgt).float()
    raw = torch.from_numpy(raw).float()

    src_color = torch.from_numpy(src_color).float()
    tgt_color = torch.from_numpy(tgt_color).float()
    raw_color = torch.from_numpy(raw_color).float()


    src_sample_centered, tgt_sample_centered,raw_sample_centered = None, None, None
    src_sample_color, tgt_sample_color, raw_sample_color = None, None, None
    
    mask_sample = None
    if sample_n>0:
        # use fps to sample the src
        if src.shape[0]>sample_n:
            src_sample, src_sample_idx = fps_torch(src,sample_n)
            src_sample_color = src_color[src_sample_idx,:]
        else:
            src_sample, src_sample_idx = None, None
            src_sample_color = None
        
        if tgt.shape[0]>sample_n:
            tgt_sample, tgt_sample_idx = fps_torch(tgt,sample_n)
            tgt_sample_color = tgt_color[tgt_sample_idx,:]
        else:
            tgt_sample, tgt_sample_idx = None, None
            tgt_sample_color = None
        
        
        raw_tgt_sample = raw[:len(tgt),:][tgt_sample_idx,:].clone()
        raw_src_sample = raw[len(tgt):,:][src_sample_idx,:].clone()
        raw_sample = torch.cat((raw_tgt_sample,raw_src_sample),dim=0)
        raw_sample_color = torch.cat((tgt_sample_color,src_sample_color),dim=0)
        mask_sample = torch.cat((torch.zeros(len(tgt_sample)), torch.ones(len(src_sample))), dim=0)
        
        # transform
        # src_sample = (R @ src_sample.T ).T
        # tgt_sample = (R @ tgt_sample.T ).T

        src_sample = Rotation.from_matrix(R).apply(src_sample)
        tgt_sample = Rotation.from_matrix(R2).apply(tgt_sample)
        # center to zero
        src_sample_mean = torch.mean(src_sample, dim=0, keepdim=True)
        tgt_sample_mean = torch.mean(tgt_sample, dim=0, keepdim=True)
        
        src_sample_centered = src_sample - src_sample_mean.repeat(len(src_sample),1)
        tgt_sample_centered = tgt_sample - tgt_sample_mean.repeat(len(tgt_sample),1)
        
        if aug2 or center_raw:
            raw_sample_centered = raw_sample - torch.mean(raw_sample, dim=0, keepdim=True).repeat(len(raw_sample),1)
        else:
            raw_sample_centered = raw_sample - tgt_sample_mean.repeat(len(raw_sample),1)

    # src = (R_to_src @ src.T + t_to_src).T
    # src = (R @ src.T ).T # the picked object
    # tgt = (R2 @ tgt.T ).T # the gripper or the placement
    src = Rotation.from_matrix(R).apply(src.numpy())
    tgt = Rotation.from_matrix(R2).apply(tgt.numpy())
    src = torch.from_numpy(src).float()
    tgt = torch.from_numpy(tgt).float()
    mask = torch.cat((torch.zeros(len(tgt)), torch.ones(len(src))), dim=0)
    assert len(mask)==len(raw)

    if add_noise:
        # print('add_noise')
        # todo add crop
        src_noise = np.random.normal(loc=0.0,scale=0.0008, size=(len(src),3))
        src_noise[np.random.permutation(src.shape[0]//2),:] =0.
        src  = src + torch.from_numpy(src_noise).float()
        
        tgt_noise = np.random.normal(loc=0.0,scale=0.0008, size=(len(tgt),3))
        tgt_noise[np.random.permutation(tgt.shape[0]//2),:] =0.
        tgt  = tgt + torch.from_numpy(tgt_noise).float()
        # raw  = src + np.random.normal(loc=0.0,scale=0.0005, size=(len(src),3))
        
    # center to the origin
    src_mean = torch.mean(src, dim=0, keepdim=True)
    tgt_mean = torch.mean(tgt, dim=0, keepdim=True)
    
    src_centered = src - src_mean.repeat(len(src),1)
    tgt_centered = tgt - tgt_mean.repeat(len(tgt),1)
    
    if aug2 or center_raw:
        print('center raw')
        raw_centered = raw - torch.mean(raw, dim=0, keepdim=True).repeat(len(raw),1)
    else:
        raw_centered = raw - tgt_mean.repeat(len(raw),1)

    return (src_centered, tgt_centered,raw_centered), (src_color, tgt_color, raw_color), mask, (src_sample_centered, tgt_sample_centered,raw_sample_centered), (src_sample_color, tgt_sample_color, raw_sample_color), mask_sample, (aT,aT2)
    # return src, tgt, T,raw


class TrainDataset(data.Dataset):
    def __init__(self, aug1=False, aug2=False, add_noise=False, sample_n = -1, data_dir='./data', center_raw=False, diverse=False, vox=False):
        self.data = []
        self.datacolor =[]
        self.raw = []
        self.raw_color = []
        self.transformation = []
        self.id = []
        self.lan = []
        self.lan_emd = []
        self.normal = []
        self.aug1 = aug1
        self.aug2 = aug2
        self.add_noise = add_noise
        self.sample_n = sample_n
        self.center_raw = center_raw
        
        
        gripper = pickle.load(open(os.path.join(data_dir, 'gripper.pkl'), 'rb'))
        for fname in sorted(os.listdir(os.path.join(data_dir,'pick'))):

            datapoint = pickle.load(open(os.path.join(data_dir,'pick' ,fname), 'rb'))
            
            if (not diverse) and (datapoint['id']>1):
                # print(datapoint['id'])
                continue

            if vox:
                self.data.append([
                datapoint['points_vox'],
                gripper['points_vox']
                ])
                self.normal.append([datapoint['normals_vox'], gripper['normals_vox']])
                self.datacolor.append([
                    datapoint['colors_vox'],
                    gripper['colors_vox']
                ])
                self.raw.append(datapoint['combined_points_vox'])
                self.raw_color.append(datapoint['combined_colors_vox'])


            else:
                self.data.append([
                    datapoint['points'],
                    gripper['points']
                ])
                self.datacolor.append([
                    datapoint['colors'],
                    gripper['colors']
                ])
                self.raw.append(datapoint['combined_points'])
                self.raw_color.append(datapoint['combined_colors'])



            self.transformation.append([datapoint['T'], gripper['T']])
            self.id.append(datapoint['id'])
            self.lan.append(datapoint['lan'])
            self.lan_emd.append(datapoint['clip_lan_emd'])

    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, index):
        # there could be an extensive data augmentation for each datapoint
        src, tgt, raw = self.data[index][0], self.data[index][1], self.raw[index]
        src_normal, tgt_normal = self.normal[index][0], self.normal[index][1]
        src_color, tgt_color =self.datacolor[index][0], self.datacolor[index][1],
        raw_color = self.raw_color[index]  # Get raw color
        id = self.id[index]
        lan = self.lan[index]
        lan_emd = torch.from_numpy(self.lan_emd[index]).float()        
        original, original_color, mask, sample, sample_color, sample_mask, aT  = sample_and_transform(src, tgt, raw, src_color, tgt_color, raw_color, aug1=self.aug1, add_noise=self.add_noise, aug2=self.aug2, sample_n=self.sample_n, center_raw=self.center_raw)
        # transform the normal
        at_src, at_tgt = aT[0], aT[1]
        src_normal = Transform.from_matrix(at_src).transform_vector(src_normal)
        tgt_normal = Transform.from_matrix(at_tgt).transform_vector(tgt_normal)
        src_normal = torch.from_numpy(src_normal).float()
        tgt_normal = torch.from_numpy(tgt_normal).float()
        return original, original_color, mask, sample, sample_color, sample_mask, aT, id, lan, lan_emd, src_normal, tgt_normal


# if __name__ == '__main__':
#     device = torch.device("cuda:0")
#     # pokemon_train = PokemonTrain(overlap_rate=0.9)
#     train_dataset = TrainDataset(add_noise=False,aug1=True,aug2=False,vox=True)
#     for i in range(0, len(train_dataset)):
#         print(i)
#         original, original_color, mask, sample, sample_color, sample_mask, aT, id,lan,lan_emd, src_n, tgt_n =  train_dataset[i]
#         src, tgt, raw = original
#         src_color, tgt_color, raw_color = original_color
#         src_sample, tgt_sample, raw_sample = sample
#         src_sample_color, tgt_sample_color, raw_sample_color = sample_color


#         src, tgt, raw = src.to(device), tgt.to(device), raw.to(device)
#         # src_sample, tgt_sample, raw_sample = src_sample.to(device), tgt_sample.to(device), raw_sample.to(device)
#         src_color, tgt_color, raw_color = src_color.to(device), tgt_color.to(device), raw_color.to(device)
#         # src_sample_color, tgt_sample_color, raw_sample_color = src_sample_color.to(device), tgt_sample_color.to(device), raw_sample_color.to(device)
#         mask = mask.to(device)
#         # mask_sample = sample_mask.to(device)

#         src_pcd, tgt_pcd = to_o3d_pcd(src,src_color,normals=src_n), to_o3d_pcd(tgt,tgt_color,normals=tgt_n)
#         # src_sample_pcd, tgt_sample_pcd = to_o3d_pcd(src_sample,src_sample_color), to_o3d_pcd(tgt_sample,tgt_sample_color)
#         print(raw.shape, tgt.shape[0]+src.shape[0])
#         raw_pcd = to_o3d_pcd(raw,raw_color)
#         o3d.visualization.draw_geometries([src_pcd], width=1000, height=800)
#         o3d.visualization.draw_geometries([tgt_pcd], width=1000, height=800)
#         # raw_sample_pcd = to_o3d_pcd(raw_sample, raw_sample_color)
#         # o3d.visualization.draw_geometries([to_o3d_pcd(raw[mask==1],raw_color[mask==1])], width=1000, height=800)
#         # o3d.visualization.draw_geometries([to_o3d_pcd(raw[mask==0],raw_color[mask==0])], width=1000, height=800)
#         # o3d.visualization.draw_geometries([raw_pcd, src_pcd, tgt_pcd], width=1000, height=800)
#         # # o3d.visualization.draw_geometries([raw_pcd], width=1000, height=800, window_name="raw")
#         # # o3d.visualization.draw_geometries([src_pcd, tgt_pcd], width=1000, height=800, window_name="raw")
#         # draw samples
#         # o3d.visualization.draw_geometries([to_o3d_pcd(raw_sample[mask_sample==1],raw_sample_color[mask_sample==1])], width=1000, height=800)
#         # o3d.visualization.draw_geometries([to_o3d_pcd(raw_sample[mask_sample==0],raw_sample_color[mask_sample==0])], width=1000, height=800)
#         # o3d.visualization.draw_geometries([raw_sample_pcd, src_sample_pcd, tgt_sample_pcd], width=1000, height=800)
#         # o3d.visualization.draw_geometries([raw_sample_pcd], width=1000, height=800, window_name="raw")
#         # o3d.visualization.draw_geometries([src_sample_pcd, tgt_sample_pcd], width=1000, height=800, window_name="raw")