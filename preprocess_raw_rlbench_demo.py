import numpy as np
import os
from rlbench_utils.arm_utils_new import _keypoint_discovery
from rlbench_utils.arm_utils_new import  flat_pcd_image
from rlbench_utils.arm_utils_new import plot_keypoints
from rlbench.utils import get_stored_demo
from rlbench.backend.utils import extract_obs
import torch
from transform import Rotation,Transform
from utils import to_o3d_pcd
import open3d as o3d
import pickle
from rlbench_utils.dataset_from_demos import shape_index, get_papb_mask, get_frame_pattern, correct_demo


'''
AN easy to read DATASET FOR RLBENCH
'''


def save_data_from_demo(root_dir='./RLBench/rlbench_data', task_name='open_drawer',
                   var_num=0, episode_idx=0,N=180, disp=False):
    device = torch.device('cpu')
    TASK = task_name
    VAR_NUM =var_num
    ROOT_DIR = root_dir
    EPISODES_FOLDER = '{}/variation{}/episodes'.format(TASK,VAR_NUM)
    CAMERAS = ['front', 'left_shoulder', 'right_shoulder', 'wrist']
    # IMAGE_SIZE =  256  #if you want to use higher voxel resolutions like 200^3, you might want to regenerate the dataset with larger images
    data_path = os.path.join(ROOT_DIR, EPISODES_FOLDER)
    #sample a demo based on episode_idx
    #get a single demo
    demo, base_init_matrix, obj_init_matrix = get_stored_demo(data_path=data_path,index=episode_idx,init_matrix=True)
    
    episode_keypoints = _keypoint_discovery(demo)
    #####
    # plot_keypoints(demo, episode_keypoints, CAMERAS=CAMERAS, camera_name='front')
    quadruples = []
    idx_pattern, number_steps = get_frame_pattern(task_name, episode_keypoints)
    
    # handle wrong keyframe of put_roll
    if idx_pattern == None:
        return

    for i in range(len(idx_pattern)):
        if i==0:
            obs_time = idx_pattern[i][0]
            obs_frame = demo[obs_time]
        else:
            obs_time = episode_keypoints[idx_pattern[i][0]]
            obs_frame = demo[obs_time]

        pick_frame = demo[episode_keypoints[idx_pattern[i][1]]]
        preplace_frame = demo[episode_keypoints[idx_pattern[i][2]]]
        place_frame = demo[episode_keypoints[idx_pattern[i][3]]]
        quadruple = [i,number_steps,obs_time,obs_frame,pick_frame,preplace_frame, place_frame]
        quadruples.append(quadruple)

    print('TASK: {}, VAR: {}, EPISODE ID: {}, NUM_STEP {}'.format(task_name, var_num, episode_idx, number_steps))
    
    for quadruple in quadruples:

        obs_time = quadruple[-5]
        obs_frame = quadruple[-4]
        pick_frame = quadruple[-3]
        preplace_frame = quadruple[-2]
        place_frame = quadruple[-1]

        def get_action(frame):
            translation = frame.gripper_pose[:3]
            quat = frame.gripper_pose[3:]
            quat = np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)
            # print(quat)
            # print(translation)
            trans = Transform(rotation=Rotation.from_quat(quat),translation=translation)

            return trans

        pick_action = get_action(pick_frame)
        preplace_action = get_action(preplace_frame)
        place_action = get_action(place_frame)
        
        obs_dict = extract_obs(obs_frame, CAMERAS, t=obs_time, prev_action=None)
        pcd_flat, flat_imag_features, flat_mask, pcds = flat_pcd_image(obs_dict, CAMERAS)
        
        # get mask
        pa_binary_mask, pb_binary_mask = get_papb_mask(task_name,flat_mask)

        # get points and color
        pa_points = pcd_flat[0][pa_binary_mask]
        pa_colors = flat_imag_features[0][pa_binary_mask]
        pb_points = pcd_flat[0][pb_binary_mask]
        pb_colors = flat_imag_features[0][pb_binary_mask]

        pa_pcd = to_o3d_pcd(pa_points,pa_colors)
        pb_pcd = to_o3d_pcd(pb_points,pb_colors)
        pcd = to_o3d_pcd(pcd_flat[0], colors=flat_imag_features[0])

        base_init_pose = Transform.from_matrix(base_init_matrix)
        object_init_pose = Transform.from_matrix(obj_init_matrix)
        # print(pick_action)
        
        Tab_preplace = preplace_action * pick_action.inverse()
        Tab_place = place_action * pick_action.inverse()
        # print(pick_action)
        if disp:
            o3d.visualization.draw_geometries([pa_pcd,pb_pcd],window_name='pa + pb')
            o3d.visualization.draw_geometries([pa_pcd,pb_pcd.transform(Tab_place.as_matrix())], window_name='p_ab')

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        pick_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        preplace_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        place_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        base_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        object_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([pcd, base_draw.transform(base_init_pose.as_matrix()), object_draw.transform(object_init_pose.as_matrix())])
        if disp:
            o3d.visualization.draw_geometries([pcd, mesh_frame, pick_draw.transform(pick_action.as_matrix()),
                                            preplace_draw.transform(preplace_action.as_matrix()),place_draw.transform(place_action.as_matrix())], window_name='pick preplace place action')
            
        data_dic = {'task_name': task_name, 'total_steps': number_steps, 
                    'base_points': pa_points, 'base_colors': pa_colors,
                    'obj_points': pb_points, 'obj_colors': pb_colors,
                    'base_init_matrix': base_init_matrix, 'obj_init_matrix': obj_init_matrix,
                    'pick_pose':pick_action.as_matrix(), 'preplace_pose': preplace_action.as_matrix(),
                    'place_pose': place_action.as_matrix(), 'lan_pick': shape_index[task_name]['lan_pick'],
                    'lan_preplace': shape_index[task_name]['lan_preplace'],'lan_place': shape_index[task_name]['lan_place'] }
        
        if not os.path.exists(os.path.join('./rlbench_data_processed',task_name, 'dict')):
            os.makedirs(os.path.join('./rlbench_data_processed',task_name, 'dict'))
        save_folder = os.path.join('./rlbench_data_processed',task_name, 'dict')
        num_pkl_files = len([f for f in os.listdir(save_folder) if os.path.isfile(os.path.join(save_folder, f)) and f.endswith('.pkl')])
        # assert num_pkl_files <= episode_idx # delete the fold and preprocess again
        fname = '{}.pkl'.format(num_pkl_files)
        with open(os.path.join(save_folder ,fname), 'wb') as f:
            pickle.dump(data_dic, f)

def build_dataset(num_demos=10,root_dir='./RLBench/rlbench_data', task_name='open_drawer',var_num=0,disp=False):
    # correct_indices = correct_demo[task_name]
    for i in range(num_demos):
        save_data_from_demo(root_dir=root_dir, task_name=task_name,var_num=var_num,episode_idx=i,disp=disp)
        

import argparse

parser = argparse.ArgumentParser(description='preprocess_raw_rlbench_demo')
parser.add_argument('--task_name', type=str, default='all')
parser.add_argument('--num_demos', type=int, default=1)
parser.add_argument('--disp', action='store_true', default=False)


args = parser.parse_args()
if args.task_name == 'all':
    for task in ['phone_on_base', 'stack_wine', 'plug_charger_in_power_supply', 'put_knife_in_knife_block', 'put_plate_in_colored_dish_rack', 'put_toilet_roll_on_stand' ]:
        build_dataset(num_demos=args.num_demos,task_name=task,disp=args.disp)
else:
    build_dataset(num_demos=args.num_demos,task_name=args.task_name,disp=args.disp)