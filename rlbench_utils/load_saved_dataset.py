from collections import deque
import random
import pickle
import os


# def scene_bound(original=True):
#     if original:
#         return [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6] # [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
#     else:
#         #return [-0.2, -0.45, 0.6, -0.2+0.9, -0.45+0.9, 0.6+0.9]
#         return [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]


class ReplayMemory(object):
    def __init__(self, capacity=100,task_name=None,numbers=None, var=False, var_num_t=None):
        #self.memory = deque([],maxlen=capacity)


        self.var = var
        self.var_num_t = var_num_t
        self.memory = self.load(task_name=task_name, numbers=numbers)
        print(type(self.memory))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def load(self,path='../saved_data',task_name=None,numbers=None):
        if self.var:
            file_name = os.path.join(path, '{}_{}_{}.obj'.format(task_name, numbers, self.var_num_t))
        else:
            file_name = os.path.join(path,'{}_{}.obj'.format(task_name,numbers))

        print(file_name)
        file = open(file_name,'rb')
        #self.memory = pickle.load(file)
        print('dataset is loaded')
        return pickle.load(file)
