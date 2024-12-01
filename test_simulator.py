import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append('..')

import argparse
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
#from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import PutPlateInColoredDishRack,PhoneOnBase,PutToiletRollOnStand,StackWine,PutKnifeInKnifeBlock,PlugChargerInPowerSupply
from rlbench_utils.dataset_from_demos import get_data
import open3d as o3d
from utils import to_o3d_pcd
import torch
from scipy.spatial.transform import Rotation
from termcolor import colored
import time
from torch.backends import cudnn
import warnings
from transform import Transform,Rotation
from rlbench.utils import get_stored_demo
warnings.filterwarnings("ignore")
import pickle
from models_stochastic.imagine_actor import ImagineActor


parser = argparse.ArgumentParser(description='RL-Bench-Test')
parser.add_argument('--train_dir', type=str, default='.')
parser.add_argument('--data_dir', type=str, default='./RLbench/rlbench_data')
parser.add_argument('--task', type=str, default='phone_on_base')
#parser.add_argument('--task', type=str, default='put_plate_in_colored_dish_rack')
parser.add_argument('--var', type=int, default=0)
parser.add_argument('--n_demos', type=int,default=10)
parser.add_argument('--n', type=int,default=200008)
parser.add_argument('--n_tests', type=int,default=25)
parser.add_argument('--rands', type=int,default=2048)
parser.add_argument('--start_test', type=int, default=10)
parser.add_argument('--disp', action='store_true', default=False)
parser.add_argument('--plot_action', action='store_true', default=False)
parser.add_argument('--expert', action='store_true', default=False)
parser.add_argument('--use_color', action='store_true', default=False)
parser.add_argument('--model_name', type=str, choices=['imagine'], default='imagine')
parser.add_argument('--device', type=int,default=0)

args = parser.parse_args()

def delta_along_axis(trans,ori,axis='z',delta=-0.05):
    if axis=='x':
        column_num = 0
    elif axis=='y':
        column_num = 1
    elif axis=='z':
        column_num = 2
    else:
        raise  ValueError('axis is wrong')
    axis = ori.as_matrix()[:, column_num]
    delta = delta * axis
    trans = trans + delta
    return trans, ori

def trans_along_axis(trans, ori, axis='z', delta=-0.05):
    if axis == 'x':
        column_num = 0
    elif axis == 'y':
        column_num = 1
    elif axis == 'z':
        column_num = 2
    else:
        raise ValueError('axis is wrong')
    axis = np.eye(3)[:, column_num]
    delta = delta * axis
    trans = trans + delta
    # ori = ori
    return trans, ori

def rot_along_axix(ori,axis='z',delta=45):
    rot = Rotation.from_euler(axis,delta,degrees=True)
    ori = rot*ori
    return ori

def main(args):
    # np.random.seed(1)
    # torch.set_num_threads(1)
    # torch.manual_seed(1)
    gripper = pickle.load(open(os.path.join('./data', 'gripper.pkl'), 'rb'))
    gripper = to_o3d_pcd(gripper['points'],gripper['colors'])
    # o3d.visualization.draw_geometries([gripper])
    
       
    if not args.expert:
        torch.cuda.set_device('cuda:{}'.format(args.device))
        name_pick = ''
        print('loading the agent ....')
        n = args.n
        actor = ImagineActor(pick_n=n,place_n=n,gripper_pcd=gripper,ds_size=0.004,rands=2048, use_color=args.use_color, device=args.device)#rands -1 means using all the points
        #print(torch.cuda.memory_summary())
        print('agent set done')
        torch.cuda.empty_cache()
    else:
        print('use the expert action, but it is not 100% success rate due to the motion planning')

    # setting up the dataset
    # To use 'saved' demos, set the path below, and set live_demos=False
    live_demos = False
    DATA_FOLDER = './RLBench/rlbench_data'
    # EPISODES_FOLDER = 'open_drawer/variation0/episodes'
    # DATA_FOLDER = os.path.join(DATA_FOLDER, EPISODES_FOLDER)
    DATASET = '' if live_demos else DATA_FOLDER
    headless = not args.disp
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    #collision
    action_mode = MoveArmThenGripper(arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True,collision_checking=False), gripper_action_mode=Discrete())
    env = Environment(action_mode, DATASET, obs_config, headless=headless)
    env.launch()
    print(args.task)
    if args.task == 'phone_on_base':
        task = env.get_task(PhoneOnBase)
        # print(task._scene._workspace_minx)
        # print(task._scene._workspace_minz)
        # print(task._scene._workspace_miny)
        max_trial = 1

    elif args.task == 'stack_wine':
        task = env.get_task(StackWine)
        max_trial = 1

    elif args.task == 'put_roll':
        task = env.get_task(PutToiletRollOnStand)
        # print(task._scene._workspace_minx)
        # print(task._scene._workspace_minz)
        # print(task._scene._workspace_miny)
        max_trial = 1

    elif args.task == "put_plate":
        task = env.get_task(PutPlateInColoredDishRack)

        max_trial = 1

    elif args.task == "insert_knife":
        task = env.get_task(PutKnifeInKnifeBlock)
        
        max_trial = 1

    elif args.task == "plug_charger":
        task = env.get_task(PlugChargerInPowerSupply)
        
        max_trial = 1

    else:
        raise ValueError('task name wrong')

    # set the variation
    task.set_variation(args.var)
    # to get a certain number of demos in order
    print('Evaluate the agent on', task.get_name(), ', for {} tests'.format(args.n_tests), ', start from {} episode'.format(args.start_test))
    results = []

    for j in range(args.n_tests):
        # demo = task.get_demos(from_episode_number=0+j, amount=1, image_paths=False, live_demos=False, random_selection=False)
        # demo = demo[0]
        EPISODES_FOLDER = '{}/variation{}/episodes'.format(task.get_name(),args.var)
        data_path = data_path = os.path.join('./RLBench/rlbench_data', EPISODES_FOLDER)
        demo, base_init_matrix, obj_init_matrix = get_stored_demo(data_path=data_path,index=args.start_test+j, init_matrix=True)
        print('task: {} test: {}'.format(args.task,j+1))
        # reset th env
        
        descriptions, obs = task.reset_to_demo(demo)
        
        def get_action_from_obs(obs):
            #print(obs.gripper_pose)
            trans = obs.gripper_pose[:3]
            quat = obs.gripper_pose[3:]
            quat = np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)
            if quat[-1] < 0:
                quat = -quat
            #gripper_open = obs.gripper_pose[-1]
            ori = Rotation.from_quat(np.asarray([quat[0],quat[1],quat[2],quat[3]]))
            return trans, ori
        
        initial_pose = get_action_from_obs(obs)
        initial_trans, initial_ori = initial_pose
        initial_action = np.concatenate([initial_trans, initial_ori.as_quat(), [1.0]], axis=-1)

        test_action_space = False
        if test_action_space:
            print('========================')
            initial_trans[-1] = initial_trans[-1] - 0.5
            initial_action =  np.concatenate([initial_trans,initial_ori.as_quat(),[1.0]],axis=-1)
            print(initial_trans)
            _ = task.step(initial_action)
            print('trans - along - z')
            time.sleep(5)
            initial_ori = rot_along_axix(initial_ori,'z', 45)
            action = np.concatenate([initial_trans,initial_ori.as_quat(),[1.0]],axis=-1)
            _ = task.step(action)
            time.sleep(5)

            move_x_trans = initial_trans.copy()
            move_x_trans[0] -= 0.15
            print(move_x_trans)

            move_x = np.concatenate([move_x_trans,initial_ori.as_quat(),[1.0]],axis=-1)
            print('move along x')
            _ = task.step(move_x)
            time.sleep(5)

            initial_ori = rot_along_axix(initial_ori, 'x', 45)
            move_x = np.concatenate([move_x_trans, initial_ori.as_quat(), [1.0]], axis=-1)
            _ = task.step(move_x)
            time.sleep(5)

            move_y_trans = move_x_trans.copy()
            move_y_trans[1] += 0.1
            print(move_x_trans)
            move_x = np.concatenate([move_y_trans, initial_ori.as_quat(), [1.0]], axis=-1)
            print('move along y')
            _ = task.step(move_x)
            time.sleep(5)

            initial_ori = rot_along_axix(initial_ori, 'y', 45)
            move_y = np.concatenate([move_y_trans, initial_ori.as_quat(), [1.0]], axis=-1)
            _ = task.step(move_y)
            time.sleep(5)

        # try at most max_trial steps
        data, demo_prepick = get_data(demo,task_name=task.get_name())
        if data is None:
            # skip this trial since the key episodes is wrong
            continue
        data = data[0]
        pa_points = data['base_points']
        pa_colors = data['base_colors']
        pb_points = data['obj_points']
        pb_colors = data['obj_colors']

        expert_pick = data['pick_pose']
        expert_preplace = data['preplace_pose']
        expert_place = data['place_pose']

        pa_pcd = to_o3d_pcd(pa_points,pa_colors)
        pb_pcd = to_o3d_pcd(pb_points,pb_colors)
        
        #print(expert_action)
        #print('expert',expert_action[1:])
        reward = 0
        for i in range(max_trial):
            ####
            if not args.expert:
                time00 = time.time()
                pick_pose, preplace_pose, place_pose = actor.act(pa_pcd,pb_pcd,data['lan_pick'],data['lan_preplace'],data['lan_place'],plot_action=False)
                infer_time = time.time() - time00
                torch.cuda.empty_cache()
                print('inference time: {}'.format(infer_time))

            if args.expert:
                pick_pose = expert_pick
                preplace_pose = expert_preplace
                place_pose = expert_place
            
            if args.plot_action:
                pick_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(pick_pose.as_matrix())
                preplace_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(preplace_pose.as_matrix())
                place_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(place_pose.as_matrix())
                pa = to_o3d_pcd(pa_points, pa_colors)
                pb = to_o3d_pcd(pb_points, pb_colors)
                o3d.visualization.draw_geometries([pa,pb,pick_draw,preplace_draw, place_draw])                 
                pass
            

            pick_trans, pick_ori = pick_pose.translation, pick_pose.rotation
            pre_place_trans, pre_place_ori = preplace_pose.translation, preplace_pose.rotation
            place_trans, place_ori = place_pose.translation, place_pose.rotation
            # print(place_trans)
            pre_transfer = False
            
            if args.task == 'phone_on_base':
                pre_pick_trans, pre_pick_ori = delta_along_axis(pick_trans, pick_ori, axis='z', delta=-0.05)
                post_pick_trans, post_pick_ori = trans_along_axis(pick_trans, pick_ori, axis='z', delta=0.1) 
                # pre_place_trans, pre_place_ori = delta_along_axis(place_trans, place_ori, axis='z', delta=-0.03)
                post_place_trans, post_place_ori = delta_along_axis(place_trans, place_ori, axis='z', delta=-0.05)
                pre_pick_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(Transform(pre_pick_ori,pre_pick_trans).as_matrix())
                post_pick_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(Transform(post_pick_ori,post_pick_trans).as_matrix())
                post_place_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(Transform(post_place_ori,post_place_trans).as_matrix())
                # o3d.visualization.draw_geometries([pa, pb, pre_pick_draw, pick_draw,post_pick_draw,preplace_draw, place_draw])

            if args.task == 'stack_wine':
                pre_pick_trans, pre_pick_ori = delta_along_axis(pick_trans, pick_ori, axis='z', delta=-0.08)
                post_pick_trans, post_pick_ori = trans_along_axis(pick_trans, pick_ori, axis='z', delta=0.1) 
                pre_place_trans, pre_place_ori = delta_along_axis(place_trans, place_ori, axis='z', delta=-0.1)
                post_place_trans, post_place_ori = delta_along_axis(place_trans, place_ori, axis='z', delta=-0.05)
                # pre_pick_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(Transform(pre_pick_ori,pre_pick_trans).as_matrix())
                # post_pick_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(Transform(post_pick_ori,post_pick_trans).as_matrix())
                # post_place_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(Transform(post_place_ori,post_place_trans).as_matrix())
                # o3d.visualization.draw_geometries([pa, pb,place_draw])
            
            if args.task == 'put_plate':
                pre_pick_trans, pre_pick_ori = delta_along_axis(pick_trans, pick_ori, axis='z', delta=-0.06)
                post_pick_trans, post_pick_ori = trans_along_axis(pick_trans, pick_ori, axis='z', delta=0.09) 
                # pre_place_trans, pre_place_ori = delta_along_axis(place_trans, place_ori, axis='z', delta=-0.03)
                post_place_trans, post_place_ori = delta_along_axis(place_trans, place_ori, axis='z', delta=-0.03)
                # post_pick_trans, post_pick_ori = delta_along_axis(pick_trans, pick_ori, axis='z', delta=-0.06)
                # post_pick_trans, post_pick_ori = trans_along_axis(post_pick_trans, post_pick_ori, axis='z', delta= 0.11)#0.15
                
                pre_pick_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(Transform(pre_pick_ori,pre_pick_trans).as_matrix())
                post_pick_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(Transform(post_pick_ori,post_pick_trans).as_matrix())
                post_place_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(Transform(post_place_ori,post_place_trans).as_matrix())
                # o3d.visualization.draw_geometries([pa, pb, pre_pick_draw, pick_draw,post_pick_draw,preplace_draw, place_draw])

            if args.task == 'insert_knife':
                pre_pick_trans, pre_pick_ori = delta_along_axis(pick_trans, pick_ori, axis='z', delta=-0.04)
                post_pick_trans, post_pick_ori = trans_along_axis(pick_trans, pick_ori, axis='z', delta=0.13) 
                # pre_place_trans, pre_place_ori = delta_along_axis(pre_place_trans, pre_place_ori, axis='z', delta=-0.03)
                post_place_trans, post_place_ori = delta_along_axis(place_trans, place_ori, axis='z', delta=-0.05)
                # post_pick_trans, post_pick_ori = delta_along_axis(pick_trans, pick_ori, axis='z', delta=-0.06)
                # post_pick_trans, post_pick_ori = trans_along_axis(post_pick_trans, post_pick_ori, axis='z', delta= 0.11)#0.15
                
                pre_pick_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(Transform(pre_pick_ori,pre_pick_trans).as_matrix())
                post_pick_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(Transform(post_pick_ori,post_pick_trans).as_matrix())
                post_place_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(Transform(post_place_ori,post_place_trans).as_matrix())
                # o3d.visualization.draw_geometries([pa, pb, pre_pick_draw, pick_draw,post_pick_draw,preplace_draw, place_draw])

            if args.task == 'plug_charger':
                pre_pick_trans, pre_pick_ori = delta_along_axis(pick_trans, pick_ori, axis='z', delta=-0.05)
                post_pick_trans, post_pick_ori = trans_along_axis(pick_trans, pick_ori, axis='z', delta=0.08) 
                # pre_place_trans, pre_place_ori = delta_along_axis(place_trans, place_ori, axis='z', delta=-0.03)
                post_place_trans, post_place_ori = delta_along_axis(place_trans, place_ori, axis='z', delta=-0.05)
                # post_pick_trans, post_pick_ori = delta_along_axis(pick_trans, pick_ori, axis='z', delta=-0.06)
                # post_pick_trans, post_pick_ori = trans_along_axis(post_pick_trans, post_pick_ori, axis='z', delta= 0.11)#0.15
                
                pre_pick_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(Transform(pre_pick_ori,pre_pick_trans).as_matrix())
                post_pick_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(Transform(post_pick_ori,post_pick_trans).as_matrix())
                post_place_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(Transform(post_place_ori,post_place_trans).as_matrix())
                # o3d.visualization.draw_geometries([pa, pb, pre_pick_draw, pick_draw,post_pick_draw,preplace_draw, place_draw])

            if args.task == 'put_roll':
               
                #
                # pre_pick_trans, pre_pick_ori = demo_prepick.translation, demo_prepick.rotation
                
                pre_pick_trans, pre_pick_ori = delta_along_axis(pick_trans, pick_ori, axis='z', delta= -0.14)
                pre_pick_trans, pre_pick_ori = trans_along_axis(pre_pick_trans, pre_pick_ori, axis='z', delta= 0.02)
                             
                pre_pick_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(Transform(pre_pick_ori,pre_pick_trans).as_matrix())
                
                #
                # pre_trans_trans, pre_trans_ori = delta_along_axis(pre_pick_trans, pre_pick_ori, axis='z', delta= -0.075)
                # pre_trans_trans, pre_trans_ori = trans_along_axis(pre_trans_trans, pre_trans_ori, axis='z', delta= 0.18)
                pre_trans_trans, pre_trans_ori = trans_along_axis(pick_trans, initial_ori, axis='z', delta= 0.28)
                pre_transfer = True
                pre_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(Transform(pre_trans_ori,pre_trans_trans).as_matrix())
                
                post_pick_trans, post_pick_ori = trans_along_axis(pick_trans, pick_ori, axis='z', delta=0.25)
                post_pick_trans, post_pick_ori = delta_along_axis(post_pick_trans, post_pick_ori, axis='z', delta=-0.05) 
                # pre_place_trans, pre_place_ori = delta_along_axis(place_trans, place_ori, axis='z', delta=-0.03)
                post_pick_draw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(Transform(post_pick_ori,post_pick_trans).as_matrix())
                
                # o3d.visualization.draw_geometries([pa,pb, pre_draw, pre_pick_draw, pick_draw,post_pick_draw,preplace_draw, place_draw])

                post_place_trans, post_place_ori = delta_along_axis(pre_place_trans, place_ori, axis='z', delta=-0.00)

            # if args.task == 'put_plate_in_colored_dish_rack':
            #     pre_pick_trans, pre_pick_ori = delta_along_axis(pick_trans, pick_ori, axis='z', delta=-0.06)
            #     post_pick_trans, post_pick_ori = delta_along_axis(pick_trans, pick_ori, axis='z', delta=-0.06)
            #     post_pick_trans, post_pick_ori = trans_along_axis(post_pick_trans, post_pick_ori, axis='z', delta= 0.11)#0.15

            #     transfer = True
            #     transfer_trans, transfer_ori = trans_along_axis(place_trans, place_ori, axis='z', delta=0.14) #0.16
            #     pre_place_trans, pre_place_ori = trans_along_axis(place_trans, place_ori, axis='z', delta=0.09)
            #     post_place_trans, post_place_ori = trans_along_axis(place_trans, place_ori, axis='z', delta=0.09)
            #     #post_place_trans, post_place_ori = delta_along_axis(place_trans, place_ori, axis='z', delta=-0.05)

            # prepare the pick actions
            if pre_transfer == True:
                pre_trans = np.concatenate([pre_trans_trans, pre_trans_ori.as_quat(),[1.0]],axis=-1)
            pre_pick = np.concatenate([pre_pick_trans,pre_pick_ori.as_quat(),[1.0]],axis=-1)
            pick_open_gripper = np.concatenate([pick_trans, pick_ori.as_quat(), [1.0]], axis=-1)
            pick_close_gripper = np.concatenate([pick_trans, pick_ori.as_quat(), [0.0]], axis=-1)
            post_pick = np.concatenate([post_pick_trans, post_pick_ori.as_quat(), [0.0]], axis=-1)
            
           
            #prepare the place actions
            pre_place = np.concatenate([pre_place_trans, pre_place_ori.as_quat(), [0.0]], axis=-1)
            place_close_gripper = np.concatenate([place_trans, place_ori.as_quat(), [0.0]], axis=-1)
            place_open_gripper = np.concatenate([place_trans, place_ori.as_quat(), [1.0]], axis=-1)
            post_place = np.concatenate([post_place_trans, post_place_ori.as_quat(), [1.0]], axis=-1)

            ## ACTION!!!
            # the test_mode will make this trial as a failure if there is a path planning error
            if pre_transfer == True:
                res = task.step(pre_trans,test_mode=True)
                if res is None:
                    print('the pre_transfer wrong, skip this trial')
                    break
                time.sleep(0.2)

            res = task.step(pre_pick,test_mode=True)
            if res is None:
                print('the pre_pick wrong, skip this trial')
                break
            time.sleep(0.2)
            res = task.step(pick_open_gripper, test_mode=True)
            if res is None:
                print('pick path planning wrong')
                break
            time.sleep(0.2)
            _ = task.step(pick_close_gripper)
            time.sleep(0.2)
            res = task.step(post_pick, test_mode=True)
            if res is None:
                print('post pick path planning wrong')
                break
            time.sleep(0.2)
            grasped_object_list = task._robot.gripper.get_grasped_objects()
            if len(grasped_object_list)>0:
                print(colored('pick an object', 'green'))
                # 1 means open, whilst 0 means closed
                print('gripper open amount: ', task._robot.gripper.get_open_amount())
            else:
                print(colored('pick failed', 'red'))
                obs, _, _ = task.step(initial_action)
                continue
            time.sleep(0.2)
            res = task.step(pre_place,test_mode=True)
            if res is None:
                print('path planning wrong, preplace')
                break
            time.sleep(0.2)
            res = task.step(place_close_gripper,test_mode=True)
            if res is None:
                print('path_planning wrong, place')
                break
            time.sleep(0.2)
            # print(place_close_gripper)
            res = task.step(place_open_gripper) # open gripper should be ok
            
            time.sleep(0.2)
            obs, reward, terminate = task.step(place_open_gripper) # check here to prevent postplace fail
            time.sleep(0.1)
            res = task.step(post_place, test_mode=True)
            if res is None:
                print('path_planning wrong, post place')
                break
            time.sleep(0.2)
            obs, reward, terminate = task.step(initial_action)
            print('Task: {}, Test: {}, Step: {}, Reward {}, Terminate {} ======='.format(args.task, j+1, i, reward, terminate))
            print('')
            print('')
            if terminate:
                break
        results.append(reward)

    num_scenes = len(results)
    success_rate = np.asarray(results).mean()
    print('')
    print('test on {} {} scenes with {} %'.format(num_scenes,args.task,success_rate*100))

if __name__=="__main__":
    main(args)
