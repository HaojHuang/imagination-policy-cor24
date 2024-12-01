import models_stochastic.clip_revised as clip_revised
from models_stochastic.clip_revised.clip import build_model,tokenize
import torch
import os
import pickle
import os
import pickle
import open3d as o3d
import numpy as np
from transform import Transform, Rotation

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


def add_lan_emd():
    # add clip_lan_emb:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # add this to dataset
    clip_model_name = "ViT-B/32"
    model, _ = clip_revised.load(clip_model_name, device=device)
    clip_vit32 = build_model(model.state_dict()).to(device)
    del model

    data_dir = './data'
    sub_dir = './pick'
    for fname in sorted(os.listdir(os.path.join(data_dir,sub_dir))):
        datapoint = pickle.load(open(os.path.join(data_dir,sub_dir ,fname), 'rb'))
        lan = datapoint['lan']
        if 'clip_lan_emd' in datapoint:
            pass
        else:
            with torch.no_grad():
                tokens = tokenize([lan]).to(device)
                text_feat, _ = clip_vit32.encode_text_with_embeddings(tokens)
            print(lan, text_feat.shape)
            datapoint['clip_lan_emd'] = text_feat.cpu().numpy()
            with open(os.path.join(data_dir, sub_dir ,fname), 'wb') as f:
                pickle.dump(datapoint, f)

    print('finish inserting clip lan_embedding')

def add_vox(voxel_size = 0.004):
    data_dir = './data'
    sub_dir = './pick'

    gripper = pickle.load(open(os.path.join(data_dir,'gripper.pkl'), 'rb'))
    gripper_pcd = to_o3d_pcd(gripper['points'], gripper['colors'])
    gripper_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    gripper_pcd_down_sample = gripper_pcd.voxel_down_sample(voxel_size=voxel_size)
    gripper_pcd_ds_normals = np.asarray(gripper_pcd_down_sample.normals)
    gripper_pcd_ds_points = np.asarray(gripper_pcd_down_sample.points)
    gripper_pcd_ds_colors = np.asarray(gripper_pcd_down_sample.colors)
    if 'points_vox' in gripper:
        pass
    else:
        gripper['points_vox'] = gripper_pcd_ds_points
        gripper['colors_vox'] = gripper_pcd_ds_colors
        with open(os.path.join(data_dir,'gripper.pkl'), 'wb') as f:
            pickle.dump(gripper, f)
    if 'normals_vox' in gripper:
        pass
    else:
        gripper['normals_vox'] = gripper_pcd_ds_normals
        with open(os.path.join(data_dir,'gripper.pkl'), 'wb') as f:
            pickle.dump(gripper, f)

    for fname in sorted(os.listdir(os.path.join(data_dir,sub_dir))):
        datapoint = pickle.load(open(os.path.join(data_dir,sub_dir ,fname), 'rb'))
        
        print('meta: ', datapoint['meta']) #indicate it's pick or place
        print('lan: ', datapoint['lan']) # pick language instruction: object part, e.g., mug hanlde
        print('id: ', datapoint['id']) # number from 1 to infinity: indicate the grasp style, e.g., mug handle id 1, mug body id 2 
        print('points: ', datapoint['points'].shape) # numpy array: the nx3 matrix for the point cloud of the picked object
        print('colors: ', datapoint['colors'].shape) # numpy array: the nx3 matrix for the color of point cloud of the picked object
        print('gripper points: ', gripper['points'].shape)
        print('gripper colors: ', gripper['colors'].shape)
        print('combined_points: ', datapoint['combined_points'].shape) # the nx3 matrix for the point cloud of the picked object and the gripper
        print('combined_colors: ', datapoint['combined_colors'].shape)
        print('T: ', datapoint['T'].shape) # how to transform the object to be picked to the combined_points
        print('gripper T: ', gripper['T'].shape) # how to transform the gripper to be picked to the combined_points
        
        
        picked_object_pcd = to_o3d_pcd(datapoint['points'], datapoint['colors'])
        combined_pcd = to_o3d_pcd(datapoint['combined_points'], datapoint['combined_colors'])
        # estimate normal
        picked_object_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        picked_object_pcd.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # downsample object
        picked_object_pcd_down_sample = picked_object_pcd.voxel_down_sample(voxel_size=voxel_size)
        
        picked_object_pcd_ds_covariances = np.asarray(picked_object_pcd_down_sample.covariances)
        picked_object_pcd_ds_normals = np.asarray(picked_object_pcd_down_sample.normals)
        picked_object_pcd_ds_points = np.asarray(picked_object_pcd_down_sample.points)
        picked_object_pcd_ds_colors = np.asarray(picked_object_pcd_down_sample.colors)
        print(picked_object_pcd_ds_points.shape)
        print(picked_object_pcd_ds_normals.shape, 'normal++++++++++++')
        print(picked_object_pcd_ds_covariances.shape, 'covariance++++++++++++')
        # downsample gripper
        print(gripper_pcd_ds_points.shape)
        # get the combined sampled points
        object_transform = Transform.from_matrix(datapoint['T'])
        gripper_transform = Transform.from_matrix(gripper['T'])
        gripper_pcd_ds_points_t = gripper_transform.transform_point(gripper_pcd_ds_points)
        picked_object_pcd_ds_points_t = object_transform.transform_point(picked_object_pcd_ds_points)
        raw_ds_points = np.concatenate((gripper_pcd_ds_points_t,picked_object_pcd_ds_points_t),axis=0)
        raw_ds_colors = np.concatenate((gripper_pcd_ds_colors,picked_object_pcd_ds_colors),axis=0)
        print(raw_ds_points.shape)
        raw_ds_pcd = to_o3d_pcd(raw_ds_points,colors=raw_ds_colors)
        
        if 'points_vox' in datapoint:
            pass
        else:
            datapoint['points_vox'] = picked_object_pcd_ds_points
            datapoint['colors_vox'] = picked_object_pcd_ds_colors
            datapoint['combined_points_vox'] = raw_ds_points
            datapoint['combined_colors_vox'] = raw_ds_colors
            with open(os.path.join(data_dir,sub_dir ,fname), 'wb') as f:
                pickle.dump(datapoint, f)

        if 'normals_vox' in datapoint:
            pass
        else:
            datapoint['normals_vox'] = picked_object_pcd_ds_normals
            with open(os.path.join(data_dir,sub_dir ,fname), 'wb') as f:
                pickle.dump(datapoint, f)
        
        # o3d.visualization.draw_geometries([picked_object_pcd_down_sample.transform(datapoint['T']), gripper_pcd_down_sample.transform(gripper['T'])], 
        #                                   window_name="transform object and gripper to match the combined points")
        # o3d.visualization.draw_geometries([raw_ds_pcd], 
        #                                   window_name="sampled combined points")
    print('finish voxelization')

# add_lan_emd()
add_vox(voxel_size = 0.004)