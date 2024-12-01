import torch
from arm_utils_new import vis_voxel,_so3_hopf_uniform_grid
# given the pick (x,y,z) and orientation, generate the gripper approach voxels
import numpy as np


def vis_virtual(virtual):
    color = torch.zeros_like(virtual)
    color[virtual>0]=1.
    color = torch.cat((color, torch.zeros_like(color), torch.zeros_like(color)), dim=1)
    signal = torch.cat((color,virtual),dim=1)
    vis_voxel(signal.cpu().numpy(),reindex=False)




def pick_2_virtual(half_size,so_points,n):
    center = half_size

    virtual = torch.zeros(1, 1, 2 * half_size, 2 * half_size, 2 * half_size)
    rot = so_points[n].as_matrix()
    rot = torch.from_numpy(rot).to(torch.float)
    z = -rot[:,-1]
    x = rot[:, 1]

    max_length = half_size//2
    sample_number = max_length * 3
    interval = max_length / sample_number
    deltas = torch.arange(0, sample_number) * interval

    #x_shift_2 = x.reshape(1, 3).repeat(sample_number, 1) * 2
    x_shift_5 = x.reshape(1, 3).repeat(len(deltas),1) * 5
    # print(deltas)
    # print(deltas.reshape(-1, 1).repeat(1, 3))
    # print(z.reshape(1, 3).repeat(sample_number, 1))
    line = z.reshape(1, 3).repeat(len(deltas), 1) * (deltas.reshape(-1, 1).repeat(1, 3))
    #line_int = torch.cat((line+center, line + center + x_shift_2, line + center - x_shift_2,line + center + x_shift_5, line + center - x_shift_5),dim=0)
    line_int = torch.cat((line + center + x_shift_5, line + center - x_shift_5), dim=0)
    line_int = torch.round(line_int).to(torch.long)
    mask = torch.logical_or(line_int > (2 * half_size - 1), line_int <=0)
    mask = torch.any(mask, dim=-1)
    line_int = line_int[~mask]
    #print(line_int)
    index_x, index_y, index_z = line_int[:, 0], line_int[:, 1], line_int[:, 2]
    virtual[0, 0, index_x, index_y, index_z] = 1.

    return virtual

# rots = _so3_hopf_uniform_grid(N=200)
# virtual = pick_2_virtual(24,rots,20)
# print(virtual.shape,'=====')
# vis_virtual(virtual)


# half_size = 24
# center = half_size-1
# virtual = torch.zeros(1,1,2*half_size,2*half_size,2*half_size)
# # index_x = [1,3,4]
# # index_y = [1,3,4]
# # index_z = [1,3,4]
# # virtual[0,0,index_x,index_y,index_z]=1.
# print(torch.sum(virtual))
# print(virtual.shape)
# #vis_virtual(virtual)
#
#
# rots = torch.from_numpy(_so3_hopf_uniform_grid(N=200).as_matrix()).to(torch.float)
# idx = 100
# rot = rots[100]
# z = rot[:,-1]
# print(z)
# max_length = int(np.sqrt(half_size**2 + half_size**2 + half_size**2))
# print(max_length)
# sample_number  = max_length*2
# interval = max_length/sample_number
#
# deltas = torch.arange(0,sample_number) * interval
# #origin_line = torch.as_tensor([center]).reshape(1,1).repeat(3,sample_number)
# #print(origin_line.shape)
# #print(deltas)
# line = z.reshape(1,3).repeat(sample_number,1) + deltas.reshape(-1,1).repeat(1,3)
# print(line.size())
# line_int = line.to(torch.long) + center
# mask = line_int > (2*half_size-1)
# print(mask.shape)
# mask = torch.any(mask,dim=-1)
# print(mask.shape)
# line_int = line_int[~mask]
# print(line_int)
# #print(line_int)
# index_x,index_y,index_z = line_int[:,0],line_int[:,1],line_int[:,2]
# virtual[0,0,index_x,index_y,index_z]=1.
# vis_virtual(virtual)
#


