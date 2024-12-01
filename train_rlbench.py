import argparse
from models_stochastic.flow_pvcnn import RectifiedFlow as FlowPvcnn
from models_stochastic.pvcnn import PVCNN

# from dataset import TrainDataset
# from torch.utils import data
from create_rlbench_pick import RLBbench_Pick_Dataset, addLanEmd, augData
from create_rlbench_place import RLBbench_Place_Dataset
from torch_geometric.transforms import Compose
from torch_geometric.data import DataLoader
import os
import torch
import numpy as np
import open3d as o3d
from utils import to_o3d_pcd


parser = argparse.ArgumentParser(description='train_flow')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--model', type=str, default='pvcnn')
parser.add_argument('--save_dir', type=str, default='./checkpoints')
parser.add_argument('--n', type=int, default=3000)
parser.add_argument('--save_steps', type=int, default=0)
parser.add_argument('--load', type=int, default=0)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--point_size', type=int, default=-1)
parser.add_argument('--aug1', action='store_true', default=False)
parser.add_argument('--aug2', action='store_true', default=False)
parser.add_argument('--bi', action='store_true', default=False)
parser.add_argument('--use_lan', action='store_true', default=False)
parser.add_argument('--vox', action='store_true', default=True)
parser.add_argument('--tab', action='store_true', default=False) # generate Tab * Pb
parser.add_argument('--randsample', action='store_true', default=False)
parser.add_argument('--gen', type=str, default='pick')
parser.add_argument('--use_color', action='store_true', default=False)
# parser.add_argument('--use_size', action='store_true', default=False)

args = parser.parse_args()


def train(args):
    
    if args.device >-1:
        device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device('cuda:{}'.format(args.device))
    else:
        device = torch.device('cpu')

    if args.model =='pvcnn':
        flower = FlowPvcnn(model=PVCNN(use_color=args.use_color), device=args.device, lr=5e-5)
    
    args.save_dir = os.path.join(args.save_dir,args.gen)
    if args.load>0:
         flower.load(path=os.path.join(args.save_dir,'{}.pt'.format(args.load)))
    if args.gen == 'pick':
        tr_dataset = RLBbench_Pick_Dataset(root='./rlbench_data_processed/pick', pre_transform=Compose([addLanEmd(),]), transform=augData(aug1=args.aug1, aug2=args.aug2, randsample=args.randsample), train=True)
    elif args.gen == 'place':
        tr_dataset = RLBbench_Place_Dataset(root='./rlbench_data_processed/place', pre_transform=Compose([addLanEmd(),]), transform=augData(aug1=args.aug1, aug2=args.aug2, randsample=args.randsample), train=True)
    
    # tr_loader = DataLoader(tr_dataset[:len(tr_dataset)], batch_size=4, shuffle=True)
    # train_set = TrainDataset(aug1=args.aug1, aug2=args.aug2, sample_n=args.sample_n, center_raw=args.center_raw, diverse=args.diverse, vox=args.vox)
    # train_loader = data.DataLoader(train_set, batch_size=1, num_workers=1, shuffle=False)
    # tr_dataset = tr_dataset[:4]
    # for batch in tr_loader:
    #     print(batch)

    loss_curve = []
    for i in range(args.load, args.n+args.load):
        # idx = i % len(tr_dataset)
        idx = np.random.randint(0,len(tr_dataset))
        # idx = i % 1    
        data = tr_dataset[idx]
        print(idx, data['lan'], data['pab_ds'].shape)
        if not args.use_lan:
            lan_emd = None
        else:
            
            lan_emd = data['lan_emd'].to(device)
        
        pa, pb, pab, mask = data['pa_ds'].to(device), data['pb_ds'].to(device), data['pab_ds'].to(device), data['mask_ds'].to(device)

        # print(scale)
        x_0 = torch.cat([pa, pb], dim=0)
        pa_color, pb_color = data['pa_ds_color'].to(device), data['pb_ds_color'].to(device)
        x0_color = torch.cat([pa_color, pb_color], dim=0)

        if args.tab:
            assert args.aug1 == True
            assert args.bi == False
            pb_target = data['pb_ds_target'].to(device)
            pab = torch.cat((pa, pb_target),dim=0)
            # o3d.visualization.draw_geometries([to_o3d_pcd(pab),], width=1000, height=800)
            pairs = torch.stack([x_0, pab], dim=1)
        else:
            # a different way to realize pick
            # assert args.aug2 == True
            # assert args.aug1 == False
            # pb_target = data['pb_ds_target'].to(device)
            # pab = torch.cat((pa, pb_target),dim=0)
            pairs = torch.stack([x_0, pab], dim=1)
            # o3d.visualization.draw_geometries([to_o3d_pcd(pab),to_o3d_pcd(x_0)], width=1000, height=800)

        if (i+1)% 100 ==0:
            flower.lr_scheduler.step()
        loss = flower.train_one_step(pairs=pairs,mask=mask,color=x0_color,batch_size=args.point_size, id=data['id'], bi=args.bi,lan_emd=lan_emd, scale=0.1)
        
        print(i, loss)
        loss_curve.append(loss)
        # print(args.save_steps)
        if args.save_steps > 0:
            if (i+1) % args.save_steps==0:
                save_file = '{}.pt'.format(args.load+i+1)
                # save_curve = '{}.npy'.format(i+1)
                if not os.path.exists(args.save_dir):
                        os.makedirs(args.save_dir)
                flower.save(path=os.path.join(args.save_dir,save_file))
                # np.save(os.path.join(args.save_dir, save_curve), np.asarray(loss_curve))
    
    save_file = '{}.pt'.format(args.n+args.load)
    save_curve = '{}.npy'.format(args.n+args.load)
    if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
    flower.save(path=os.path.join(args.save_dir,save_file))
    np.save(os.path.join(args.save_dir, save_curve), np.asarray(loss_curve))

if __name__ == '__main__':
    train(args)