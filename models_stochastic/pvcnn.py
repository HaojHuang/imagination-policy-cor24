
import sys
import os
sys.path.append('../')
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import torch
import numpy as np
import torch.nn as nn

from models_stochastic.pvcnn_generation_modify import PVCNN2Base
from models_stochastic.pvcnn_encoder import PVCNN2Base as PVCNNEncoder


class MLPG(nn.Module):
    def __init__(self, in_dim=3, hidden_num=256, out_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc4 = nn.Linear(hidden_num*2, hidden_num, bias=True)
        self.fc5 = nn.Linear(hidden_num, out_dim, bias=True)
        
        self.relu = torch.nn.ReLU()
        

    def forward(self, x):
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        g = torch.max(x,dim=0,keepdim=True)[0]
        g = g.repeat(len(x),1)
        x = torch.cat((x,g),dim=-1)
        x = self.relu(self.fc4(x))
        x = self.fc5(x)

        return x
    

class PVCNN2(PVCNN2Base):
    sa_blocks = [
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]
    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, num_classes, embed_dim, use_att,dropout, extra_feature_channels=3, width_multiplier=1,
                 voxel_resolution_multiplier=1):
        super().__init__(
            num_classes=num_classes, embed_dim=embed_dim, use_att=use_att,
            dropout=dropout, extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )

class PVCNNEn(PVCNNEncoder):
    sa_blocks = [
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]
    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, num_classes, embed_dim, use_att,dropout, extra_feature_channels=3, width_multiplier=1,
                 voxel_resolution_multiplier=1):
        super().__init__(
            num_classes=num_classes, embed_dim=embed_dim, use_att=use_att,
            dropout=dropout, extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )


class MLPPVCNN(nn.Module):
    def __init__(self, input_dim=3+3+1, hidden_num=256, use_color=False):
        super().__init__()
        

        if use_color:
            init_dim = 6
        else:
            init_dim = 3

        self.pa_encoder = MLPG(in_dim = init_dim, out_dim=64)
        self.pb_encoder = MLPG(in_dim = init_dim, out_dim=64)

        self.pvcnn = PVCNN2(num_classes=3, embed_dim=64, use_att=True, dropout=0.1, extra_feature_channels=96)
        self.lan_proj = nn.Linear(512,32,bias=True)
        self.use_color = use_color

    def forward(self, x_input, p0, t, mask=None, id=None, lan_emd=None, p0_encoder = None, p0_color=None):
        
        pa = p0[mask==0]
        pb = p0[mask==1]

        pa_color = p0_color[mask==0]
        pb_color = p0_color[mask==1]

        if self.use_color:
            pa = torch.cat((pa,pa_color),dim=-1)
            pb = torch.cat((pb,pb_color),dim=-1)
        pdf_a = self.pa_encoder(pa)
        pdf_b = self.pb_encoder(pb)

        pdf = torch.zeros(len(mask),pdf_a.shape[-1]).to(p0.device)
        pdf[mask==0,:] = pdf_a
        pdf[mask==1,:] = pdf_b
        
        lan = self.lan_proj(lan_emd)
        lan = lan.repeat(len(pdf),1)
        pdf = torch.cat((pdf,lan),dim=-1)
        # print(pdf.shape)
        x = x_input.unsqueeze(dim=0).permute(0,2,1)        
        out, _ = self.pvcnn(x, p0, t, mask, lan_emd, pdf)
        out = out.permute(0,2,1).squeeze(dim=0)
        return out, pdf
    


class PVCNNPVN(nn.Module):
    def __init__(self, input_dim=3+3+1, hidden_num=256, use_color=False):
        super().__init__()
        # self.inv_pa_encoder = SO3MLP(out_dim=3)
        # self.inv_pb_encoder = SO3MLP(out_dim=3)
        self.pvcnn = PVCNN2(num_classes=3, embed_dim=64, use_att=True, dropout=0.1, extra_feature_channels=96)
        
        self.lan_proj = nn.Linear(512,32,bias=True)
        self.pdf_proj = nn.Linear(2049, 64, bias=True)

    def forward(self, x_input, p0, t, mask=None, id=None, lan_emd=None, p0_encoder = None, p0_color=None):
        
        with torch.no_grad():
            pts_a ={}
            pts_a['point_cloud'] = p0[mask==0].unsqueeze(dim=0)
            pts_a['coords'] = p0[mask==0].unsqueeze(dim=0)
            out_dict_a = p0_encoder(pts_a)
            pdf_a = out_dict_a['features']
            pdf_a = pdf_a.squeeze(dim=0)
            ##
            pts_b ={}
            pts_b['point_cloud'] = p0[mask==1].unsqueeze(dim=0)
            pts_b['coords'] = p0[mask==1].unsqueeze(dim=0)
            out_dict_b = p0_encoder(pts_b)
            pdf_b = out_dict_b['features']
            pdf_b = pdf_b.squeeze(dim=0)
            ##
            # import pickle
            # flower = pickle.load(open(os.path.join('./intro/data.pkl'), 'rb'))
            # pts_f = {}
            # points = torch.from_numpy(flower['points']).to(torch.float).to(p0.device)
            # pts_f['point_cloud'] = points.unsqueeze(dim=0)
            # pts_f['coords'] = points.unsqueeze(dim=0)
            # out_dict_f = p0_encoder(pts_f)
            # pdf_f = out_dict_f['features']
            # pdf_f = pdf_f.squeeze(dim=0)
            # from vis_high_d_feature import pcd_hd_vis
            # pcd_hd_vis(points.cpu().numpy(),features=pdf_f.cpu().numpy())
            # pcd_hd_vis(points=p0[mask==0].cpu().numpy(),features=pdf_a.cpu().numpy())
            # pcd_hd_vis(points=p0[mask==1].cpu().numpy(),features=pdf_b.cpu().numpy())




        

        from vis_high_d_feature import pcd_hd_vis
        

        pdf = torch.zeros(len(mask),pdf_a.shape[-1]).to(p0.device)
        pdf[mask==0,:] = pdf_a
        pdf[mask==1,:] = pdf_b

        pdf = self.pdf_proj(pdf)
        lan = self.lan_proj(lan_emd)
        lan = lan.repeat(len(pdf),1)
        pdf = torch.cat((pdf,lan),dim=-1)
        # print(pdf.shape)
        
        x = x_input.unsqueeze(dim=0).permute(0,2,1)
        # print(x.shape)
        # print(x.shape, t)
        out, _ = self.pvcnn(x, p0, t, mask, lan_emd, pdf)
        # print(out.shape)
        out = out.permute(0,2,1).squeeze(dim=0)
        # print(out.shape)
        return out, pdf


class PVCNN(nn.Module):
    def __init__(self, input_dim=3+3+1, hidden_num=256, use_color=False):
        super().__init__()
        
        # self.inv_pa_encoder = SO3MLP(out_dim=3)
        # self.inv_pb_encoder = SO3MLP(out_dim=3)
        self.pvcnn = PVCNN2(num_classes=3, embed_dim=64, use_att=True, dropout=0.1, extra_feature_channels=64+32)
        self.use_color = use_color
        if use_color:
            init_dim = 6
        else:
            init_dim = 3
        self.pa_encoder = PVCNNEn(num_classes=64, embed_dim=1, use_att=True, dropout=0.1, extra_feature_channels=init_dim)
        self.pb_encoder = PVCNNEn(num_classes=64, embed_dim=1, use_att=True, dropout=0.1, extra_feature_channels=init_dim)
        
        self.lan_proj = nn.Linear(512,32,bias=True)
        
    def forward(self, x_input, p0, t, mask=None, id=None, lan_emd=None, p0_encoder = None, p0_color=None, pdf=None):
        
        lan = self.lan_proj(lan_emd).unsqueeze(dim=-1)
        lan = lan.repeat(1,1,len(mask))
        
        if pdf is None:
            pa = p0[mask==0].unsqueeze(dim=0).permute(0,2,1)
            pb = p0[mask==1].unsqueeze(dim=0).permute(0,2,1)
            
            if self.use_color:
                # import open3d as o3d
                # from utils import to_o3d_pcd
                # o3d.visualization.draw_geometries([to_o3d_pcd(p0,p0_color)], width=1000, height=800)

                pa_color = p0_color[mask==0].unsqueeze(dim=0).permute(0,2,1)
                pb_color = p0_color[mask==1].unsqueeze(dim=0).permute(0,2,1)
                pa_emd,_ = self.pa_encoder(pa, pa_color)
                pb_emd,_ = self.pb_encoder(pb, pb_color)
                
            else:
                pa_emd,_ = self.pa_encoder(pa)
                pb_emd,_ = self.pb_encoder(pb)
            p_emd = torch.zeros(pa_emd.shape[0], pa_emd.shape[1],len(mask)).to(p0.device)
            p_emd[:,:,mask==0] =pa_emd
            p_emd[:,:,mask==1] =pb_emd
            
            pdf = torch.cat((p_emd,lan),dim=1)
            pdf = pdf.squeeze(dim=0).permute(1,0)
        
        # print(pdf.shape)
        x = x_input.unsqueeze(dim=0).permute(0,2,1)
        # print(x.shape)
        # print(x.shape, t)
        out, _ = self.pvcnn(x, p0, t, mask, lan_emd, pdf)
        # print(out.shape)
        out = out.permute(0,2,1).squeeze(dim=0)
        # print(out.shape)
        return out, pdf