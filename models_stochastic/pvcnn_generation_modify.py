import functools
import torch.nn as nn
import torch
import numpy as np
from modules import SharedMLP, PVConv, PointNetSAModule, PointNetAModule, PointNetFPModule, Attention, Swish
from so3_mlp import SO3MLP
import torch.nn.functional as F
from utils import pcd_hd_vis, to_o3d_pcd
import open3d as o3d



def _linear_gn_relu(in_channels, out_channels):
    return nn.Sequential(nn.Linear(in_channels, out_channels), nn.GroupNorm(8,out_channels), Swish())


def create_mlp_components(in_channels, out_channels, classifier=False, dim=2, width_multiplier=1):
    r = width_multiplier

    if dim == 1:
        block = _linear_gn_relu
    else:
        block = SharedMLP
    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or (len(out_channels) == 1 and out_channels[0] is None):
        return nn.Sequential(), in_channels, in_channels

    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(r * oc)
            layers.append(block(in_channels, oc))
            in_channels = oc
    if dim == 1:
        if classifier:
            layers.append(nn.Linear(in_channels, out_channels[-1]))
        else:
            layers.append(_linear_gn_relu(in_channels, int(r * out_channels[-1])))
    else:
        if classifier:
            layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
        else:
            layers.append(SharedMLP(in_channels, int(r * out_channels[-1])))
    return layers, out_channels[-1] if classifier else int(r * out_channels[-1])


def create_pointnet_components(blocks, in_channels, embed_dim, with_se=False, normalize=True, eps=0,
                               width_multiplier=1, voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier

    layers, concat_channels = [], 0
    c = 0
    for k, (out_channels, num_blocks, voxel_resolution) in enumerate(blocks):
        out_channels = int(r * out_channels)
        for p in range(num_blocks):
            attention = k % 2 == 0 and k > 0 and p == 0
            if voxel_resolution is None:
                block = SharedMLP
            else:
                block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution), attention=attention,
                                          with_se=with_se, normalize=normalize, eps=eps)

            if c == 0:
                layers.append(block(in_channels, out_channels))
            else:
                layers.append(block(in_channels+embed_dim, out_channels))
            in_channels = out_channels
            concat_channels += out_channels
            c += 1
    return layers, in_channels, concat_channels


def create_pointnet2_sa_components(sa_blocks, extra_feature_channels, embed_dim=64, use_att=False,
                                   dropout=0.1, with_se=False, normalize=True, eps=0,
                                   width_multiplier=1, voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier
    in_channels = extra_feature_channels
    
    sa_layers, sa_in_channels = [], []
    c = 0
    for conv_configs, sa_configs in sa_blocks:
        # print('conv configs',conv_configs)
        k = 0
        sa_in_channels.append(in_channels)
        sa_blocks = []

        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            for p in range(num_blocks):
                attention = (c+1) % 2 == 0 and use_att and p == 0
                # print(attention,'****')#???
                if voxel_resolution is None:
                    block = SharedMLP
                else:
                    block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution), attention=attention,
                                              dropout=dropout,
                                              with_se=with_se, with_se_relu=True,
                                              normalize=normalize, eps=eps)

                if c == 0:
                    sa_blocks.append(block(in_channels, out_channels))
                elif k ==0:
                    sa_blocks.append(block(in_channels+embed_dim, out_channels))
                in_channels = out_channels
                k += 1
            extra_feature_channels = in_channels
        # print(len(sa_blocks),'after conv fig +++++++++')
        num_centers, radius, num_neighbors, out_channels = sa_configs
        _out_channels = []
        for oc in out_channels:
            if isinstance(oc, (list, tuple)):
                _out_channels.append([int(r * _oc) for _oc in oc])
            else:
                _out_channels.append(int(r * oc))
        out_channels = _out_channels
        if num_centers is None:
            block = PointNetAModule
        else:
            block = functools.partial(PointNetSAModule, num_centers=num_centers, radius=radius,
                                      num_neighbors=num_neighbors)
        sa_blocks.append(block(in_channels=extra_feature_channels+(embed_dim if k==0 else 0 ), out_channels=out_channels,
                               include_coordinates=True))
        # print(len(sa_blocks),'++++after sa_configs')
        c += 1
        in_channels = extra_feature_channels = sa_blocks[-1].out_channels
        if len(sa_blocks) == 1:
            sa_layers.append(sa_blocks[0])
        else:
            sa_layers.append(nn.Sequential(*sa_blocks))

    return sa_layers, sa_in_channels, in_channels, 1 if num_centers is None else num_centers


def create_pointnet2_fp_modules(fp_blocks, in_channels, sa_in_channels, embed_dim=64, use_att=False,
                                dropout=0.1,
                                with_se=False, normalize=True, eps=0,
                                width_multiplier=1, voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier

    fp_layers = []
    c = 0
    for fp_idx, (fp_configs, conv_configs) in enumerate(fp_blocks):
        fp_blocks = []
        out_channels = tuple(int(r * oc) for oc in fp_configs)
        fp_blocks.append(
            PointNetFPModule(in_channels=in_channels + sa_in_channels[-1 - fp_idx] + embed_dim, out_channels=out_channels)
        )
        in_channels = out_channels[-1]

        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            for p in range(num_blocks):
                attention = (c+1) % 2 == 0 and c < len(fp_blocks) - 1 and use_att and p == 0
                if voxel_resolution is None:
                    block = SharedMLP
                else:
                    block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution), attention=attention,
                                              dropout=dropout,
                                              with_se=with_se, with_se_relu=True,
                                              normalize=normalize, eps=eps)

                fp_blocks.append(block(in_channels, out_channels))
                in_channels = out_channels
        if len(fp_blocks) == 1:
            fp_layers.append(fp_blocks[0])
        else:
            fp_layers.append(nn.Sequential(*fp_blocks))

        c += 1

    return fp_layers, in_channels



class PVCNN2Base(nn.Module):

    def __init__(self, num_classes, embed_dim, use_att, dropout=0.1,
                 extra_feature_channels=3, width_multiplier=1, voxel_resolution_multiplier=1, ):
        super().__init__()
        assert extra_feature_channels >= 0
        self.embed_dim = embed_dim
        self.in_channels = extra_feature_channels
        # print(self.sa_blocks)
        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks, extra_feature_channels=extra_feature_channels, with_se=True, embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        self.global_att = None if not use_att else Attention(channels_sa_features, 8, D=1)

        # only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels, with_se=True, embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.fp_layers = nn.ModuleList(fp_layers)


        layers, _ = create_mlp_components(in_channels=channels_fp_features, out_channels=[128, dropout, num_classes], # was 0.5
                                          classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

        self.embedf = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim//2),
        )
        
        self.embedm = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim//2),
        )

        self.inv_pa_encoder = SO3MLP(out_dim=64)
        self.inv_pb_encoder = SO3MLP(out_dim=64)
        self.lan_proj = nn.Linear(512,32,bias=True)
        self.inv_g_proj = nn.Linear(128,32,bias=True)
        self.inv_dpf_proj = nn.Linear(64,32,bias=True)

    def get_timestep_embedding(self, timesteps, device):
        assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32

        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
        # emb = tf.range(num_embeddings, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:  # zero pad
            # emb = tf.concat([emb, tf.zeros([num_embeddings, 1])], axis=1)
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        assert emb.shape == torch.Size([timesteps.shape[0], self.embed_dim]) # bs x 64
        return emb
    
    def get_mask_embedding(self, mask, device):
        assert len(mask.shape) == 1  # and timesteps.dtype == tf.int32

        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
        # emb = tf.range(num_embeddings, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
        emb = mask[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:  # zero pad
            # emb = tf.concat([emb, tf.zeros([num_embeddings, 1])], axis=1)
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        assert emb.shape == torch.Size([mask.shape[0], self.embed_dim]) # bs x 64
        return emb

    def forward(self, inputs, p0=None, t=None, mask=None, lan_emd=None, pdf=None):
        
        if pdf is None:
            # print(p0.shape)
            # o3d.visualization.draw_geometries([to_o3d_pcd(inputs.squeeze(dim=0).permute(1,0))])
            # o3d.visualization.draw_geometries([to_o3d_pcd(p0)])
            pa_input = self.inv_pa_encoder.in_type(p0[mask==0])
            pb_input = self.inv_pb_encoder.in_type(p0[mask==1])
            pa_inv_dpf = self.inv_pa_encoder(pa_input)
            pb_inv_dpf = self.inv_pb_encoder(pb_input)
            # global feature and fuse together
            pa_g_inv = torch.max(pa_inv_dpf,dim=0,keepdim=True)[0]
            pb_g_inv = torch.max(pb_inv_dpf,dim=0,keepdim=True)[0]
            pab_g_inv = torch.cat((pa_g_inv,pb_g_inv),dim=-1)
            pab_g_inv = F.relu(self.inv_g_proj(pab_g_inv))
            lan_emd = self.lan_proj(lan_emd)
            pab_g = torch.cat((pab_g_inv,lan_emd),dim=-1)
            # dense point feature inv
            pab_inv_dpf = torch.zeros(len(mask), pa_inv_dpf.shape[-1],device=inputs.device)
            pab_inv_dpf[mask==0] = pa_inv_dpf
            pab_inv_dpf[mask==1] = pb_inv_dpf        
            # print(pab_inv_dpf)
            pab_inv_dpf = F.relu(self.inv_dpf_proj(pab_inv_dpf))
            pab_inv_dpf_g = torch.cat((pab_inv_dpf,pab_g.expand(len(mask),-1)),dim=-1)
            pdf = pab_inv_dpf_g.unsqueeze(dim=0).permute(0,2,1)
        else:
            pdf = pdf.unsqueeze(dim=0).permute(0,2,1)



        temb =  self.embedf(self.get_timestep_embedding(t, inputs.device))[:,:,None]
        # print(temb.shape,'temd')
        temb = temb.expand(-1,-1,inputs.shape[-1]) # bs X emd x npts 
        # print(temb.shape, 't expand')
        # inputs : [B, in_channels + S, N]
        memb =  self.embedm(self.get_mask_embedding(mask, inputs.device))
        memb = memb.unsqueeze(dim=0).permute(0,2,1)
        temb = torch.cat((temb,memb),dim=1)


        coords = inputs[:, :3, :].contiguous()
        features = pdf # 1 x 96 x N
        features_0 = pdf.clone()
        # print(features_0)
        # pcd_hd_vis(p0[mask==1].cpu().numpy(), features_0.squeeze(dim=0).permute(1,0)[mask==1].detach().cpu().numpy(),model='tsne')
        # pcd_hd_vis(p0[mask==1].cpu().numpy(), features_0.squeeze(dim=0).permute(1,0)[mask==1].detach().cpu().numpy(), model='pca')
        # print(features.shape, )
        # features = inputs
        # print(features.shape)
        
        # print('input coords',coords.shape)
        # print('input feature',features.shape)
        
        coords_list, in_features_list = [], []
        for i, sa_blocks  in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)
            if i == 0:
                features, coords, temb = sa_blocks ((features, coords, temb))
            else:
                features, coords, temb = sa_blocks ((torch.cat([features,temb],dim=1), coords, temb))
            # print('sa blocks', i, features.shape)
        
        # in_features_list[0] = inputs[:, 3:, :].contiguous()
        in_features_list[0] = features_0
        
        # print('in feature list [0]', in_features_list[0])
        if self.global_att is not None:
            features = self.global_att(features)
            # print('after attention',features.shape)
        for fp_idx, fp_blocks  in enumerate(self.fp_layers):
            features, coords, temb = fp_blocks((coords_list[-1-fp_idx], coords, torch.cat([features,temb],dim=1), in_features_list[-1-fp_idx], temb))
            # print('fp idx', fp_idx,features.shape)

        # print('before final mlp',features.shape)
        out = self.classifier(features)
        # print('final output',out.shape)
        return out, features_0[0].permute(1,0)

