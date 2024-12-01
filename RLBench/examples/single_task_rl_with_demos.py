import numpy as np
import os
from rlbench.action_modes.action_mode import MoveArmThenGripper
#from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import OpenDrawer
import time
def normalize_quaternion(quat):
    return np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)

class Agent(object):

    def __init__(self, action_shape):
        self.action_shape = action_shape

    def ingest(self, demos):
        pass

    def act(self, obs):
        #print(obs.gripper_pose)
        arm = np.random.normal(0.0, 0.1, size=(self.action_shape[0] - 1,))
        arm[:3] = obs.gripper_pose[:3]
        arm = obs.gripper_pose
        if arm[-1]<0:
            arm[3:] = -arm[3:]
        arm[3:] = normalize_quaternion(arm[3:])

        # arm[0] = 0.5
        # arm[1] = 0.6
        # arm[2] = 1.5
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)


# To use 'saved' demos, set the path below, and set live_demos=False
live_demos = False

DATA_FOLDER ='../rlbench_data'
#EPISODES_FOLDER = 'open_drawer/variation0/episodes'
#DATA_FOLDER = os.path.join(DATA_FOLDER, EPISODES_FOLDER)

DATASET = '' if live_demos else DATA_FOLDER

obs_config = ObservationConfig()
obs_config.set_all(True)

action_mode = MoveArmThenGripper(
    arm_action_mode=EndEffectorPoseViaPlanning(), gripper_action_mode=Discrete())

env = Environment(
    action_mode, DATASET, obs_config, headless=False)
env.launch()

task = env.get_task(OpenDrawer)
# to set the variation
task.set_variation(0)
# to get a certain number of demos in order
print(task.get_name())

demos = task.get_demos(2, live_demos=False,random_selection=False)
print(demos)


agent = Agent(env.action_shape)
#agent.ingest(demos)

training_steps = 40
episode_length = 40
obs = None

quat0 = np.asarray([0.70715151, 0.01436131, 0.01554552, 0.70674524])
trans0 = np.asarray([0.23462337,  0.06333873 , 0.91470951])

quat = np.array([0.70682761, 0.01550241, 0.01566284, 0.70704249])
trans = np.asarray([0.23851131, -0.02362473,  0.915609])

quat_b = quat
trans_b = np.asarray([ 0.2284274,   0.21238697,  0.91540492])

action0 = np.concatenate((trans0,quat0,np.array([1.])),axis=0)
action1 = np.concatenate((trans,quat,np.array([0.])),axis=0)
action2 = np.concatenate((trans_b,quat_b,np.array([0.])),axis=0)


for i in range(training_steps):
    print(i)
    if i % episode_length == 0:
        print('Reset Episode')
        #descriptions, obs = task.reset()
        descriptions,obs = task.reset_to_demo(demos[0])
        #print(descriptions)
    if i==0:
        action = action0
        print(action0)
    elif i ==1:
        action = action1
    elif i==2:
        action = action2
    else:
        action = agent.act(obs)
        #print(action)
    obs, reward, terminate = task.step(action)
    time.sleep(2)
    print(reward,terminate)

print('Done')
env.shutdown()
