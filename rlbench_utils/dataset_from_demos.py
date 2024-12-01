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


shape_index = {'phone_on_base':{'pa_mask_index': 93, 'pb_mask_index':95, 'lan_pick':'pick up the phone', 'lan_preplace': 'preplace the phone above the base', 'lan_place': 'place the phone on the base'},
               'put_knife_in_knife_block':{'pa_mask_index': 104, 'pb_mask_index':108, 'lan_pick':'pick up the knife', 'lan_preplace': 'preplace the knife above the knife block', 'lan_place': 'place the knife inside the knife block'},
               'stack_wine':{'pa_mask_index': 104, 'pb_mask_index':96, 'lan_pick':'pick up the wine by the neck', 'lan_preplace': 'preplace the wine above the rack', 'lan_place': 'place the wine on the rack'},
               'put_toilet_roll_on_stand':{'pa_mask_index': 104, 'pb_mask_index':98, 'lan_pick':'pick up the toilet roll', 'lan_preplace': 'preplace the toilet roll near the stand', 'lan_place': 'place the toilet roll on the stand'},
               'plug_charger_in_power_supply':{'pa_mask_index': 101, 'pb_mask_index':116, 'lan_pick':'pick up the power charger', 'lan_preplace': 'preplace the charger near the power supply', 'lan_place': 'plug charger in the power supply'},
               'put_plate_in_colored_dish_rack':{'pa_mask_index': [114,115,116,117,118,119,108], 'pb_mask_index':104, 'lan_pick':'pick up the plate', 'lan_preplace': 'preplace the plate above the read spoke', 'lan_place': 'place the plate between red spokes'},
               }

correct_demo = {'phone_on_base':[0,1,2,3,4,5,6,7,8,9], 
               'put_knife_in_knife_block': [0,1,2,3,4,5,6,7,8,9], 
               'stack_wine':[0,1,2,3,4,5,6,7,8,9], 
               'put_toilet_roll_on_stand':[0,1,2,3,4,5,6,7,8,9], 
               'plug_charger_in_power_supply': [0,1,2,3,4,5,6,7,8,9], 
               'put_plate_in_colored_dish_rack': [0,1,2,3,4,5,6,7,8,9], 
               }


def get_action(frame):
        translation = frame.gripper_pose[:3]
        quat = frame.gripper_pose[3:]
        quat = np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)
        # print(quat)
        # print(translation)
        trans = Transform(rotation=Rotation.from_quat(quat),translation=translation)

        return trans

def get_frame_pattern(task_name, episode_keypoints):
    number_steps = None
    if task_name=='stack_wine' or task_name=='insert_onto_square_peg' or task_name=='put_plate_in_colored_dish_rack' or task_name=='phone_on_base' or task_name=='put_knife_in_knife_block':
        number_steps = 1
        idx_pattern = [[0, 1, 3, 4, 0, 2]]
    
    elif task_name =='put_toilet_roll_on_stand':
        number_steps = 1
        idx_pattern = [[0, 1, 3, 4, 0, 2]]
        if len(episode_keypoints) <6:
            print('wrong demo skip this data')
            return None, None

    elif task_name=='plug_charger_in_power_supply':
        number_steps = 1
        idx_pattern = [[0, 0, 2, 3]]
    return idx_pattern, number_steps

def get_papb_mask(task_name, flat_mask):
    flat_mask = flat_mask[0,:,0] # vector
    pa_mask_index = shape_index[task_name]['pa_mask_index']
    pb_mask_index = shape_index[task_name]['pb_mask_index']
    
    if task_name == 'insert_onto_square_peg':
        pa_binary_mask = (flat_mask == pa_mask_index[0]) | (flat_mask==pa_mask_index[1]) | (flat_mask==pa_mask_index[2]) | (flat_mask==pa_mask_index[3])
        pb_binary_mask = flat_mask==pb_mask_index
    elif task_name == 'put_plate_in_colored_dish_rack':
        pa_binary_mask = (flat_mask == pa_mask_index[0]) | (flat_mask==pa_mask_index[1]) | (flat_mask==pa_mask_index[2]) | (flat_mask == pa_mask_index[3]) | (flat_mask==pa_mask_index[4]) | (flat_mask==pa_mask_index[5]) | (flat_mask==pa_mask_index[6])
        pb_binary_mask = flat_mask==pb_mask_index
    else:
        pa_binary_mask = flat_mask==pa_mask_index
        pb_binary_mask = flat_mask==pb_mask_index
    return pa_binary_mask, pb_binary_mask


def get_data(demo, task_name, CAMERAS=['front', 'left_shoulder', 'right_shoulder', 'wrist']):
    
    # plot_keypoints(demo, episode_keypoints, CAMERAS=CAMERAS, camera_name='front')
    quadruples = []
    episode_keypoints = _keypoint_discovery(demo)
    idx_pattern, number_steps = get_frame_pattern(task_name, episode_keypoints)
    if idx_pattern == None:
        return idx_pattern, number_steps
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

        prepick_frame = demo[episode_keypoints[idx_pattern[i][-2]]]
        postpick_frame = demo[episode_keypoints[idx_pattern[i][-1]]]
        prepick_action = get_action(prepick_frame)
        # print('prepick', get_action(prepick_frame).translation)
        # print('pick', get_action(pick_frame).translation)
        # print('postpick', get_action(postpick_frame).translation)

    data_list = []
    for quadruple in quadruples:

        obs_time = quadruple[-5]
        obs_frame = quadruple[-4]
        pick_frame = quadruple[-3]
        preplace_frame = quadruple[-2]
        place_frame = quadruple[-1]

        pick_action = get_action(pick_frame)
        preplace_action = get_action(preplace_frame)
        place_action = get_action(place_frame)
        
        obs_dict = extract_obs(obs_frame, CAMERAS, t=obs_time, prev_action=None)
        pcd_flat, flat_imag_features, flat_mask, pcds = flat_pcd_image(obs_dict, CAMERAS)
        pa_binary_mask, pb_binary_mask = get_papb_mask(task_name, flat_mask)

        pa_points = pcd_flat[0][pa_binary_mask]
        pa_colors = flat_imag_features[0][pa_binary_mask]
        pb_points = pcd_flat[0][pb_binary_mask]
        pb_colors = flat_imag_features[0][pb_binary_mask]

        pa_pcd = to_o3d_pcd(pa_points,pa_colors)
        pb_pcd = to_o3d_pcd(pb_points,pb_colors)
        pcd = to_o3d_pcd(pcd_flat[0], colors=flat_imag_features[0])

        # print(pick_action)
        Tab_preplace = preplace_action * pick_action.inverse()
        Tab_place = place_action * pick_action.inverse()
        # print(pick_action)

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        pick_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(pick_action.as_matrix())
        preplace_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(preplace_action.as_matrix())
        place_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(place_action.as_matrix())

        base_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        object_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        
        # import copy
        # o3d.visualization.draw_geometries([pa_pcd,pb_pcd, pick_draw], window_name = 'obs')
        # o3d.visualization.draw_geometries([pa_pcd,copy.deepcopy(pb_pcd).transform(Tab_preplace.as_matrix()), preplace_draw],  window_name = 'preplace')
        # o3d.visualization.draw_geometries([pa_pcd,copy.deepcopy(pb_pcd).transform(Tab_place.as_matrix()), place_draw],  window_name = 'place')

        data = {'task_name': task_name, 'total_steps': number_steps, 
                'base_points': pa_points, 'base_colors': pa_colors,
                'obj_points': pb_points, 'obj_colors': pb_colors,
                'pick_pose':pick_action, 'preplace_pose': preplace_action,
                'place_pose': place_action, 'lan_pick': shape_index[task_name]['lan_pick'],
                'Tab_preplace': Tab_preplace, 'Tab_place': Tab_place,
                'lan_preplace': shape_index[task_name]['lan_preplace'],'lan_place': shape_index[task_name]['lan_place'] }
        data_list.append(data)
    
    return data_list, prepick_action
    