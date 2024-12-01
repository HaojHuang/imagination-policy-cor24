import numpy as np
import os
from rlbench_utils.arm_utils_new import VoxelGrid,_keypoint_discovery,_get_action
from rlbench_utils.arm_utils_new import reindex_label_for_pytorch, flat_pcd_image
from rlbench_utils.arm_utils_new import reindex_input_for_pytorch, delta_to_discrete_idx_faster
from rlbench.utils import get_stored_demo
from rlbench_utils.arm_utils_new import extract_obs,set_task_voxel_info,visualise_samples
import torch
import pickle
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append('..')

'''
AN easy to read DATASET FOR RLBENCH
'''

def data_from_demo(sv, root_dir='./RLBench/rlbench_data', task_name_list=['stack_blocks'],
                   var_num=0, episode_idx=0,N=180,fineN=6000,so3_points=None,img_size=160):


    data_dic = {'task_name_list': task_name_list}
    data_for_pyg = []

    for i,task_name in enumerate(task_name_list):
        device = torch.device('cpu')
        TASK = task_name
        VAR_NUM =var_num
        ROOT_DIR = root_dir
        EPISODES_FOLDER = '{}/variation{}/episodes'.format(TASK,VAR_NUM)
        CAMERAS = ['front', 'left_shoulder', 'right_shoulder', 'wrist']
        IMAGE_SIZE =  img_size  # 128x128 - if you want to use higher voxel resolutions like 200^3, you might want to regenerate the dataset with larger images
        SCENE_BOUNDS, VOXEL_SIZES, res = set_task_voxel_info(task_name,var_num,sv=sv) # [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
        data_path = os.path.join(ROOT_DIR, EPISODES_FOLDER)
        description_path = os.path.join(ROOT_DIR,'{}/variation{}/variation_descriptions.pkl'.format(TASK,VAR_NUM))
        with open(description_path, 'rb') as f:
            des = pickle.load(f)

        BATCH_SIZE=1
        #sample a demo based on episode_idx
        #get a single demo
        demo = get_stored_demo(data_path=data_path,index=episode_idx)
        episode_keypoints = _keypoint_discovery(demo)
        #plot_keypoints(demo, episode_keypoints, CAMERAS=CAMERAS, camera_name='front')
        bounds = torch.tensor(SCENE_BOUNDS, device=device).unsqueeze(0)

        triples = []
        #number_steps = None
        if task_name=='open_drawer':
            number_steps = 1
            idx_pattern = [[0,1,2]]
            #todo task specific settings to get (ot,a_pick,a_place)
            for i in range(len(idx_pattern)):
                obs_time = idx_pattern[i][0]
                obs_frame = demo[obs_time]
                pick_frame = demo[episode_keypoints[idx_pattern[i][1]]]
                place_frame = demo[episode_keypoints[idx_pattern[i][2]]]
                triple = [i,number_steps,obs_time,obs_frame,pick_frame,place_frame]
                triples.append(triple)
        if task_name=='stack_blocks':
            number_steps = 2
            idx_pattern = [[0,1,4],[6,8,11]]
            #todo task specific settings to get (ot,a_pick,a_place)
            for i in range(len(idx_pattern)):
                if i==0:
                    obs_time = idx_pattern[i][0]
                    obs_frame = demo[obs_time]
                else:
                    obs_time = episode_keypoints[idx_pattern[i][0]]
                    obs_frame = demo[obs_time]

                pick_frame = demo[episode_keypoints[idx_pattern[i][1]]]
                place_frame = demo[episode_keypoints[idx_pattern[i][2]]]
                triple = [i,number_steps,obs_time,obs_frame,pick_frame,place_frame]
                triples.append(triple)

        if task_name=='stack_wine' or task_name=="place_cups" or task_name=='put_plate_in_colored_dish_rack' or task_name=='phone_on_base':
            number_steps = 1
            idx_pattern = [[0, 1,4]]
            for i in range(len(idx_pattern)):
                if i==0:
                    obs_time = idx_pattern[i][0]
                    obs_frame = demo[obs_time]
                else:
                    obs_time = episode_keypoints[idx_pattern[i][0]]
                    obs_frame = demo[obs_time]

                pick_frame = demo[episode_keypoints[idx_pattern[i][1]]]
                place_frame = demo[episode_keypoints[idx_pattern[i][2]]]
                triple = [i,number_steps,obs_time,obs_frame,pick_frame,place_frame]
                triples.append(triple)

        if task_name =='stack_cups':
            number_steps = 2
            idx_pattern = [[0, 1, 4], [5, 7, 10]]
            # todo task specific settings to get (ot,a_pick,a_place)
            for i in range(len(idx_pattern)):
                if i == 0:
                    obs_time = idx_pattern[i][0]
                    obs_frame = demo[obs_time]
                else:
                    obs_time = episode_keypoints[idx_pattern[i][0]]
                    obs_frame = demo[obs_time]

                pick_frame = demo[episode_keypoints[idx_pattern[i][1]]]
                place_frame = demo[episode_keypoints[idx_pattern[i][2]]]
                triple = [i, number_steps, obs_time, obs_frame, pick_frame, place_frame]
                triples.append(triple)

        print('TASK: {}, VAR: {}, EPISODE ID: {}, NUM_STEP {}'.format(task_name, var_num, episode_idx, number_steps))


        for triple in triples:
            obs_time = triple[-4]
            obs_frame = triple[-3]
            pick_frame = triple[-2]
            place_frame = triple[-1]
            #trans_idx, hepix idx, quat,raw_action =(trans, unnormalized quat, gripper_open)
            trans_indicies0, idx0, quat0, action0, _ = _get_action(pick_frame,rlbench_scene_bounds=SCENE_BOUNDS,voxel_sizes=VOXEL_SIZES,N=N,so3_points=so3_points)
            trans_indicies1, _, quat1, action1, _ = _get_action(place_frame,rlbench_scene_bounds=SCENE_BOUNDS,voxel_sizes=VOXEL_SIZES,N=N,so3_points=so3_points)
            delta_idx = delta_to_discrete_idx_faster(quat0,quat1,N=N,so3_points=so3_points)
            #print('delta_idx',delta_idx)
            trans_indicies0 = np.asarray(trans_indicies0)
            p0 = reindex_label_for_pytorch(trans_idx=trans_indicies0,voxel_size=VOXEL_SIZES)
            trans_indicies1 = np.asarray(trans_indicies1)
            p1 = reindex_label_for_pytorch(trans_idx=trans_indicies1,voxel_size=VOXEL_SIZES)
            a0 = (p0,idx0)
            a1 = (p1,delta_idx)
            vox_grid = VoxelGrid(coord_bounds=SCENE_BOUNDS,voxel_size=VOXEL_SIZES,device=device,batch_size=BATCH_SIZE,feature_size=3,max_num_coords=np.prod([IMAGE_SIZE,IMAGE_SIZE])*len(CAMERAS))
            # get the visualization
            obs_dict = extract_obs(obs_frame,CAMERAS,t=obs_time,prev_action=None)
            pcd_flat,flat_imag_features,pcds = flat_pcd_image(obs_dict,CAMERAS)
            # voxel_grid = vox_grid.coords_to_bounding_voxel_grid(pcd_flat,
            #                                                     coord_features=flat_imag_features,
            #                                                     coord_bounds=bounds)
            #
            # vis_voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach().cpu().numpy()
            # voxel_pyt = reindex_input_for_pytorch(vis_voxel_grid)
            # samples_pyt = (voxel_pyt,a0,a1,action0,action1,pcds,flat_imag_features,CAMERAS,IMAGE_SIZE)
            samples_pyt = (a0, a1, action0, action1, pcds, flat_imag_features, CAMERAS, IMAGE_SIZE, task_name, des)
            data_for_pyg.append(samples_pyt)
            #visualise_samples((samples_pyt[0], samples_pyt[1], samples_pyt[2]), SCENE_BOUNDS, VOXEL_SIZES, delta=True)
            # print(samples_pyt)
            #print(samples_pyt)

    data_dic['data_for_pyg'] = data_for_pyg
    # data_dic['action_pick'] = action0
    # data_dic['action_place'] = action1
    return data_dic

from collections import deque
import random

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)
    def push(self, data):
        """Save a triple"""
        self.memory.append(data)
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)
    def save(self,file_name,):
        file = open(file_name,'wb')
        pickle.dump(self.memory,file)
        print('dataset is saved')

def build_dataset(num_demos=1,root_dir='../RLBench/rlbench_data', save_path='../saved_data', task_name_list=['open_drawer'],var_lists=[[0,1,2]], sv=True,img_size=160):

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    var_num_total = sum([len(varlist) for varlist in var_lists])
    print(var_num_total)
    file_name = os.path.join(save_path, '{}_{}_{}.obj'.format(len(task_name_list), num_demos, var_num_total))
    if os.path.exists(file_name):
        print('data was saved')
        return None
    else:
        dataset = ReplayMemory(capacity=10000)
        for i in range(num_demos):
            # each task has a var_list
            for id, task in enumerate(task_name_list):
                for var_num in var_lists[id]:
                    data_dic = data_from_demo(root_dir=root_dir, task_name_list=[task],var_num=var_num,episode_idx=i,sv=sv,img_size=img_size)
                    for data in data_dic['data_for_pyg']:
                        print(data[-1])
                        dataset.push(data)
        print(task_name_list)
        print('number of data point:', len(dataset))
        dataset.save(file_name)
        return None

def voxel_from_demo(sv,demo,task_name,var_num,obs_time=0,img_size=160):
    IMAGE_SIZE = img_size
    device = torch.device('cpu')
    SCENE_BOUNDS,VOXEL_SIZES,_ = set_task_voxel_info(task_name,var_num,sv=sv)
    bounds = torch.tensor(SCENE_BOUNDS, device=device).unsqueeze(0)
    BATCH_SIZE = 1
    CAMERAS = ['front', 'left_shoulder', 'right_shoulder', 'wrist']

    obs_frame = demo[obs_time]
    vox_grid = VoxelGrid(coord_bounds=SCENE_BOUNDS, voxel_size=VOXEL_SIZES, device=device, batch_size=BATCH_SIZE,
                         feature_size=3, max_num_coords=np.prod([IMAGE_SIZE, IMAGE_SIZE]) * len(CAMERAS))
    # get the visualization
    obs_dict = extract_obs(obs_frame, CAMERAS, t=obs_time, prev_action=None)
    pcd_flat, flat_imag_features = flat_pcd_image(obs_dict, CAMERAS)
    voxel_grid = vox_grid.coords_to_bounding_voxel_grid(pcd_flat,
                                                        coord_features=flat_imag_features,
                                                        coord_bounds=bounds)
    vis_voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach().cpu().numpy()
    voxel_pyt = reindex_input_for_pytorch(vis_voxel_grid)
    return voxel_pyt


from rlbench.backend.observation import Observation

REMOVE_KEYS = ['joint_velocities', 'joint_positions', 'joint_forces',
               'gripper_open', 'gripper_pose',
               'gripper_joint_positions', 'gripper_touch_forces',
               'task_low_dim_state', 'misc']

# for i in range(10):
#     data_dic = data_from_demo(episode_idx=i)

def extract_obs_for_acting(obs: Observation,
                cameras,
                prev_action=None,
                channels_last: bool = False):
    obs.joint_velocities = None
    grip_mat = obs.gripper_matrix
    grip_pose = obs.gripper_pose
    joint_pos = obs.joint_positions
    obs.gripper_pose = None
    obs.gripper_matrix = None
    obs.wrist_camera_matrix = None
    obs.joint_positions = None
    if obs.gripper_joint_positions is not None:
      obs.gripper_joint_positions = np.clip(
        obs.gripper_joint_positions, 0., 0.04)

    obs_dict = vars(obs)
    obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
    # for k, v in obs_dict.items():
    #   print('raw data', k)
    robot_state = np.array([
      obs.gripper_open,
      *obs.gripper_joint_positions])
    # remove low-level proprioception variables that are not needed
    obs_dict = {k: v for k, v in obs_dict.items()
                if k not in REMOVE_KEYS}

    if not channels_last:
      # swap channels from last dim to 1st dim
      obs_dict = {k: np.transpose(
        v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
                  for k, v in obs_dict.items()}
    else:
      # add extra dim to depth data
      obs_dict = {k: v if v.ndim == 3 else np.expand_dims(v, -1)
                  for k, v in obs_dict.items()}
    obs_dict['low_dim_state'] = np.array(robot_state, dtype=np.float32)

    # binary variable indicating if collisions are allowed or not while planning paths to reach poses
    # obs_dict['ignore_collisions'] = np.array([obs.ignore_collisions], dtype=np.float32)
    for (k, v) in [(k, v) for k, v in obs_dict.items() if 'point_cloud' in k]:
      obs_dict[k] = v.astype(np.float32)

    for camera_name in cameras:
      obs_dict['%s_camera_extrinsics' % camera_name] = obs.misc['%s_camera_extrinsics' % camera_name]
      obs_dict['%s_camera_intrinsics' % camera_name] = obs.misc['%s_camera_intrinsics' % camera_name]

    # # add timestep to low_dim_state
    # episode_length = 10  # TODO fix this
    # time = (1. - (t / float(episode_length - 1))) * 2. - 1.
    # obs_dict['low_dim_state'] = np.concatenate(
    #   [obs_dict['low_dim_state'], [time]]).astype(np.float32)

    obs.gripper_matrix = grip_mat
    obs.joint_positions = joint_pos
    obs.gripper_pose = grip_pose

    return obs_dict


def get_actions_from_demo(sv,demo, task_name='open_drawer',var_num=0,N=180, so3_points=None,img_size=160):
    device = torch.device('cpu')
    TASK = task_name
    CAMERAS = ['front', 'left_shoulder', 'right_shoulder', 'wrist']
    IMAGE_SIZE =  img_size  # 128x128 - if you want to use higher voxel resolutions like 200^3, you might want to regenerate the dataset with larger images
    SCENE_BOUNDS,VOXEL_SIZES,_ = set_task_voxel_info(task_name,var_num,sv=sv) # [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
    BATCH_SIZE=1

    episode_keypoints = _keypoint_discovery(demo)
    #plot_keypoints(demo, episode_keypoints, CAMERAS=CAMERAS, camera_name='front')
    bounds = torch.tensor(SCENE_BOUNDS, device=device).unsqueeze(0)
    triples = []
    if task_name=='open_drawer':
        number_steps = 1
        idx_pattern = [[0,1,2]]
        #todo task specific settings to get (ot,a_pick,a_place)
        for i in range(len(idx_pattern)):
            obs_time = idx_pattern[i][0]
            obs_frame = demo[obs_time]
            pick_frame = demo[episode_keypoints[idx_pattern[i][1]]]
            place_frame = demo[episode_keypoints[idx_pattern[i][2]]]
            triple = [i,number_steps,obs_time,obs_frame,pick_frame,place_frame]
            triples.append(triple)

    if task_name=='stack_blocks':
        number_steps = 2
        idx_pattern = [[0,1,4],[6,8,11]]
        #todo task specific settings to get (ot,a_pick,a_place)
        for i in range(len(idx_pattern)):
            if i==0:
                obs_time = idx_pattern[i][0]
                obs_frame = demo[obs_time]
            else:
                obs_time = episode_keypoints[idx_pattern[i][0]]
                obs_frame = demo[obs_time]

            pick_frame = demo[episode_keypoints[idx_pattern[i][1]]]
            place_frame = demo[episode_keypoints[idx_pattern[i][2]]]
            triple = [i,number_steps,obs_time,obs_frame,pick_frame,place_frame]
            triples.append(triple)

    if task_name=='stack_wine' or task_name=='place_cups' or task_name=='put_plate_in_colored_dish_rack' or task_name=='phone_on_base':
        number_steps = 1
        idx_pattern = [[0,1,4]]
        for i in range(len(idx_pattern)):
            if i==0:
                obs_time = idx_pattern[i][0]
                obs_frame = demo[obs_time]
            else:
                obs_time = episode_keypoints[idx_pattern[i][0]]
                obs_frame = demo[obs_time]

            pick_frame = demo[episode_keypoints[idx_pattern[i][1]]]
            place_frame = demo[episode_keypoints[idx_pattern[i][2]]]
            triple = [i,number_steps,obs_time,obs_frame,pick_frame,place_frame]
            triples.append(triple)

    if task_name =='stack_cups':
        number_steps = 2
        idx_pattern = [[0, 1, 4], [5, 7, 10]]
        # todo task specific settings to get (ot,a_pick,a_place)
        for i in range(len(idx_pattern)):
            if i == 0:
                obs_time = idx_pattern[i][0]
                obs_frame = demo[obs_time]
            else:
                obs_time = episode_keypoints[idx_pattern[i][0]]
                obs_frame = demo[obs_time]

            pick_frame = demo[episode_keypoints[idx_pattern[i][1]]]
            place_frame = demo[episode_keypoints[idx_pattern[i][2]]]
            triple = [i, number_steps, obs_time, obs_frame, pick_frame, place_frame]
            triples.append(triple)


    data_dic = {'task_name':task_name,'total_steps':number_steps,}
    data_for_pyg = []
    for triple in triples:
        obs_time = triple[-4]
        obs_frame = triple[-3]
        pick_frame = triple[-2]
        place_frame = triple[-1]
        #trans_idx, hepix idx, quat,raw_action =(trans, unnormalized quat, gripper_open)
        trans_indicies0, idx0, quat0, action0, _ = _get_action(pick_frame,rlbench_scene_bounds=SCENE_BOUNDS,voxel_sizes=VOXEL_SIZES,N=N,so3_points=so3_points)
        trans_indicies1, _, quat1, action1, _ = _get_action(place_frame,rlbench_scene_bounds=SCENE_BOUNDS,voxel_sizes=VOXEL_SIZES,N=N,so3_points=so3_points)
        idx1 = delta_to_discrete_idx_faster(quat0,quat1,N=N,so3_points=so3_points)
        trans_indicies0 = np.asarray(trans_indicies0)
        p0 = reindex_label_for_pytorch(trans_idx=trans_indicies0,voxel_size=VOXEL_SIZES)
        trans_indicies1 = np.asarray(trans_indicies1)
        p1 = reindex_label_for_pytorch(trans_idx=trans_indicies1,voxel_size=VOXEL_SIZES)
        a0 = (p0,idx0)
        a1 = (p1,idx1)
        vox_grid = VoxelGrid(coord_bounds=SCENE_BOUNDS,voxel_size=VOXEL_SIZES,device=device,batch_size=BATCH_SIZE,feature_size=3,max_num_coords=np.prod([IMAGE_SIZE,IMAGE_SIZE])*len(CAMERAS))
        # get the visualization
        obs_dict = extract_obs(obs_frame,CAMERAS,t=obs_time,prev_action=None)
        pcd_flat,flat_imag_features,_ = flat_pcd_image(obs_dict,CAMERAS)
        # voxel_grid = vox_grid.coords_to_bounding_voxel_grid(pcd_flat,
        #                                                     coord_features=flat_imag_features,
        #                                                     coord_bounds=bounds)
        #
        # vis_voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach().cpu().numpy()
        # voxel_pyt = reindex_input_for_pytorch(vis_voxel_grid)
        # samples_pyt = (voxel_pyt,a0,a1, action0[:3],quat0, action1[:3],quat1)
        samples_pyt = (a0, a1, action0[:3], quat0, action1[:3], quat1)
        data_for_pyg.append(samples_pyt)
        #visualise_samples(samples_pyt,SCENE_BOUNDS,VOXEL_SIZES[0],)
        #print(samples_pyt)
    data_dic['data_for_pyg'] = data_for_pyg
    return data_dic

def voxel_from_obs(sv,obs,task_name,var_num,img_size=160,normalized=False):
    IMAGE_SIZE = img_size
    device = torch.device('cpu')
    SCENE_BOUNDS,VOXEL_SIZES,_ = set_task_voxel_info(task_name,var_num,sv=sv)
    bounds = torch.tensor(SCENE_BOUNDS, device=device).unsqueeze(0)
    BATCH_SIZE = 1
    CAMERAS = ['front', 'left_shoulder', 'right_shoulder', 'wrist']
    vox_grid = VoxelGrid(coord_bounds=SCENE_BOUNDS, voxel_size=VOXEL_SIZES, device=device, batch_size=BATCH_SIZE,
                         feature_size=3, max_num_coords=np.prod([IMAGE_SIZE, IMAGE_SIZE]) * len(CAMERAS))
    # get the visualization
    obs_dict = extract_obs_for_acting(obs, CAMERAS, prev_action=None)
    pcd_flat, flat_imag_features,_ = flat_pcd_image(obs_dict, CAMERAS)
    voxel_grid = vox_grid.coords_to_bounding_voxel_grid(pcd_flat,
                                                        coord_features=flat_imag_features,
                                                        coord_bounds=bounds)
    vis_voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach().cpu().numpy()
    voxel_pyt = reindex_input_for_pytorch(vis_voxel_grid,normalized=normalized)
    return voxel_pyt
