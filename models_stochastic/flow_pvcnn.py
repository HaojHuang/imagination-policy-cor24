import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import open3d as o3d
from utils import to_o3d_pcd
import sys
import os
sys.path.append('../')
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)



class RectifiedFlow():
    def __init__(self, model, device = 1, lr=5e-3):
        
        if device >-1:
            self.device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device('cpu')

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-6)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.998)


    @torch.no_grad()
    def get_train_tuple(self, pairs, mask, color, batch_size =-1, step=0, bi=False, scale=0.1):
        
        
        indices = torch.randperm(len(pairs)).to(self.device)
        
        batch = pairs[indices]
        mask_batch = mask[indices]
        color_batch = color[indices]

        p0 = batch[:, 0].clone()
        z0 = torch.randn(*p0.shape,device=self.device) * scale

        z1 = batch[:, 1].clone()
        
        if not bi:
            print('use pa')
            z0[mask_batch==0] = z1[mask_batch==0].clone()
        t = torch.rand(1).to(self.device)
        z_t =  t * z1 + (1.-t) * z0
        target = z1 - z0
        t = (t*999).reshape(-1)
        return z_t, t, p0, target, mask_batch, color_batch

    def train_one_step(self, pairs, mask, color, batch_size=-1, step=None, id=None, bi=False, lan_emd=None, scale=0.1):
        # print(bi)
        self.model.train()
        z_t, t, p0, target, mask_batch, color_batch = self.get_train_tuple(pairs, mask, color, batch_size=batch_size, step=step, bi=bi, scale=scale)
        if mask_batch.sum() ==0 or mask_batch.sum()==len(mask_batch):
            return 0.001
        
        else:
            self.optimizer.zero_grad()
            # self.model.train()
            if lan_emd is not None:
                pred,_ = self.model(z_t, p0, t, mask_batch, id=id, lan_emd=lan_emd, p0_color=color_batch)
            else:
                pred,_ = self.model(z_t, p0, t, mask_batch, id=id, p0_color=color_batch)
            if bi:
                loss = (target - pred).view(len(pred), -1).abs().pow(2).sum(dim=1)
            else:
                loss = (target[mask_batch==1.] - pred[mask_batch==1.]).view(len(pred[mask_batch==1]), -1).abs().pow(2).sum(dim=1)
            # print(loss.shape)
            # print(np.log((target[mask_batch==1.] - pred[mask_batch==1.]).pow(2).mean().item()))
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()

            return np.log(loss.item())
    
    def save(self,path):
        self.model.eval()
        torch.save(self.model.state_dict(), path)
        print('save {}'.format(path))
    
    def load(self, path):
        self.model.eval()
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print('load {}'.format(path))

    def get_betas(self, schedule_type, b_start, b_end, time_num):
        if schedule_type == 'linear':
            betas = np.linspace(b_start, b_end, time_num)
        elif schedule_type == 'warm0.1':

            betas = b_end * np.ones(time_num, dtype=np.float64)
            warmup_time = int(time_num * 0.1)
            betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
        elif schedule_type == 'warm0.2':

            betas = b_end * np.ones(time_num, dtype=np.float64)
            warmup_time = int(time_num * 0.2)
            betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
        elif schedule_type == 'warm0.5':

            betas = b_end * np.ones(time_num, dtype=np.float64)
            warmup_time = int(time_num * 0.5)
            betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
        else:
            raise NotImplementedError(schedule_type)
        return betas

    @torch.no_grad()
    def sample_ode(self, p0, mask, color=None, noise =None, sample_ode_steps=1000, id=None, bi=False, lan_emd=None, scale=0.1):
        self.model.eval()
        ### NOTE: Use Euler method to sample from the learned flow
        dt = 1./sample_ode_steps
        traj = [] # to store the trajectory
        # betas = self.get_betas(schedule_type='warm0.2', b_start=1e-4, b_end=1.1/sample_ode_steps, time_num=sample_ode_steps) # integral == 1
        # betas = self.get_betas(schedule_type='linear', b_start=0, b_end=1.0/sample_ode_steps, time_num=sample_ode_steps) # integral == 1
        # print(len(betas))
        # print(betas)
        if noise is not None:
            z = noise.to(self.device)
        else:
            z = torch.randn(*p0.shape, device=self.device) * scale
            # z = torch.rand(*p0.shape,device=self.device)
            # z = p0.clone()
            # z = torch.zeros_like(p0)
            # z = p0.clone() + torch.rand(*p0.shape, device=self.device)
        
        batchsize = z.shape[0]
        traj.append(z.clone().cpu().numpy())

        if not bi:
            z[mask==0] = p0[mask==0].clone()
        unchange_part = p0[mask==0]
        for i in range(sample_ode_steps):
            # dt = betas[i]
            t = torch.empty(1, dtype=torch.int64, device=self.device).fill_(i)
            if lan_emd is None:
                
                pred, pdf = self.model(z, p0, t, mask, id, p0_color=color)
                # o3d.visualization.draw_geometries([to_o3d_pcd(p0)])
                # o3d.visualization.draw_geometries([to_o3d_pcd(z)])
            else:
                if i ==0:
                    pred, pdf = self.model(z, p0, t, mask, id, lan_emd, p0_color=color)
                else:
                    pred, _ = self.model(z, p0, t, mask, id, lan_emd, p0_color=color, pdf=pdf)
                    # print(i)
                
            if i % 1000 ==0:
                print(t.item(), dt)
            if not bi:
                z[mask==1.] = z.clone()[mask==1.] + pred[mask==1] * dt
                # z = z.clone() + pred * dt
                
                change_part = z[mask==1]
                # print(change_part.shape, unchange_part.shape)         
                traj.append(torch.cat((unchange_part.cpu().clone(), change_part.cpu().clone()),dim=0).numpy())
            else:
                z = z.clone() + pred * dt
                
                traj.append(z.clone().cpu().numpy())
            if i==0:
                pdf0=pdf.clone()
            # traj.append(z.cpu().clone().numpy())        
        #to do save the point cloud flow to video
        return traj, pdf