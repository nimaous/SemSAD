# coding=utf-8
# Copyright 2021 The SemSAD Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3"
import torch
from torch import nn, optim
from tqdm import tqdm
import random
import argparse
import numpy as np
vis = True
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    vis = False
from torchvision.utils import make_grid
from torchvision import models
from torchvision.models.resnet import resnet34, resnet18

from utils import  count_parameters
from dataset_wrapper import DataSetWrapper
from change_resnet import modify_resnet_model
from torch.optim.lr_scheduler import LambdaLR

"""  if you get tensorflow related error 
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
"""

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



norm2 = torch.nn.PairwiseDistance(p=2)
log_sigmoid = nn.LogSigmoid()



wt = 0.001
def avg_loss(args, a_pos, a_neg, GN):                   
    loss = torch.mean(-(log_sigmoid(a_pos) + GN*log_sigmoid(-a_neg)), dim=0) # average over samples
    min_loss, min_dim = loss.topk(1, largest=False) # select smallest dim  
    loss_avg = loss.mean() #average over dimensions
    return loss_avg, min_dim

def train(epoch, args, loader1, loader2, s_net,  s_net_opt):
    h_net.eval()
    s_net.train()     
    data_itr = iter(loader1)    
    (img_r_neg , img_r_s , _, _), _ = next(data_itr)
    img_r_neg, img_r_s = img_r_neg.to(device), img_r_s.to(device)    
    loader2 = tqdm(loader2)
    for itr, ((img_neg, img_t_s, img_t1, img_t2), _) in enumerate(loader2):  
        if args.gamma_neg == -1 :
            GN = np.random.randint(args.min_gamma_neg, args.max_gamma_neg)
        else:
            GN = args.gamma_neg 
        s_net_opt.zero_grad()               
        img_neg, img_t_s  = img_neg.to(device), img_t_s.to(device)
        img_t1, img_t2   = img_t1.to(device), img_t2.to(device)
        with torch.no_grad():            
            h1 = h_net(img_t_s[0:args.bs//args.div])
            h2 = h_net(img_r_s)
            h1_norm = h1/norm2(h1, torch.zeros_like(h1)).view(-1,1)
            h2_norm = h2/norm2(h2, torch.zeros_like(h2)).view(-1,1)
            res_t = torch.matmul(h1_norm, h2_norm.transpose(1,0)) #[bs//div, comp_bs]
            _, k_best_idx = res_t.topk(args.n_neighbor, dim=1) # select top k best 
            rnd_neighbor_idx = torch.randint(0, args.n_neighbor, [args.bs//args.div,1]).to(device)#rnd idx for k best neighbors
            selected_idx = k_best_idx.gather(1, rnd_neighbor_idx) #select a neighbor based on random index
            similar_img = img_r_s[selected_idx].squeeze() #[bs//div]            
            rnd_idx1 = list(range(args.bs))
            random.shuffle(rnd_idx1)            
            ref_img = torch.cat((img_t_s[0:args.bs//args.div], img_t1[args.bs//args.div:]), dim=0)[rnd_idx1]
            pos_img = torch.cat((similar_img, img_t2[args.bs//args.div:]), dim=0)[rnd_idx1]             
        z_pos = s_net(torch.cat((ref_img, pos_img), dim=1))
        z_neg = s_net(torch.cat((ref_img, img_r_neg[0:args.bs]), dim=1)) 
        a_pos = z_pos - np.log(GN)
        a_neg = z_neg - np.log(GN)         
        loss, best_dim = avg_loss(args, a_pos, a_neg, GN)
        loss.backward()
#         if itr == (50000//args.bs)-1:
#             for tag, parm in s_net.named_parameters():
#                  writer.add_histogram(tag, parm.grad.data.cpu().numpy(), epoch)        
        s_net_opt.step()
        if vis == True:
            writer.add_scalar('LR',  s_net_opt.state_dict()['param_groups'][0]['lr'], global_step=epoch, walltime=wt)        
        rnd_idx2 = list(range(img_t1.size(0)))        
        random.shuffle(rnd_idx2)  
        img_r_neg = img_neg[rnd_idx2].clone()  
        random.shuffle(rnd_idx2)
        img_r_s = img_t_s[rnd_idx2].clone()  
                
        loader2.set_description(
            (
                f' Epoch: {epoch + 1};  Loss: {loss.item()}' 
            )
        ) 
    if vis == True:
        with torch.no_grad():  
            if epoch % 100 == 1:
                img_cat = torch.cat((ref_img[0:32],                               
                                     pos_img[0:32],
                                     img_r_neg[0:32] ),
                                       dim=0)
                writer.add_image("Img1Img2ImgR_32",make_grid(img_cat).detach().cpu(), global_step=epoch, walltime=wt)         
            writer.add_histogram('Positive Pairs', a_pos.cpu().numpy(), global_step=epoch, walltime=wt)
            writer.add_histogram('Negative Pairs', a_neg.cpu().numpy(), global_step=epoch, walltime=wt)           
            writer.add_scalar("Loss", loss.item(), global_step=epoch, walltime=wt)        
            writer.add_scalar("Min Dim", best_dim.item(), global_step=epoch, walltime=wt)  
    return best_dim.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--comp_bs', type=int, default=10000)
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--ds_dir', type=str, default='.')
    parser.add_argument('--h_net_path', type=str, default='h_net_cifar100.pt' )
    parser.add_argument('--epoch', type=int, default=700)
    parser.add_argument('--div', type=int, default=32)    
    parser.add_argument('--s_lr', type=float, default=5e-5)   
    parser.add_argument('--weight_decay', type=float, default=1e-6)   
    parser.add_argument('--s_dim', type=int, default=64)
    parser.add_argument('--n_neighbor', type=int, default=4)
    parser.add_argument('--gamma_neg', type=int, default=-1)                 
    parser.add_argument('--min_gamma_neg', type=int, default=1)                 
    parser.add_argument('--max_gamma_neg', type=int, default=10)                                            
    parser.add_argument('--nw', type=int, default=6)         
    args = parser.parse_args()    

    assert args.h_net_path is not None
    h_net_ckpt = torch.load(args.h_net_path)
    h_args = h_net_ckpt['args']
    h_net = resnet18(pretrained=False, progress=False, num_classes=h_args.h_dim)
    h_net = modify_resnet_model(h_net, args, mode='encoder')
    print("loading h net state dictionary")
    h_net.load_state_dict(h_net_ckpt['model'])         
    print("Number of h net Paramters: ", count_parameters(h_net))
    h_net = nn.DataParallel(h_net)    
    h_net = h_net.to(device)
    
                           

      
    s_net = resnet34(pretrained=False, progress=True, num_classes=args.s_dim)    
    s_net = modify_resnet_model(s_net, args, mode='discriminator')         
    print("Number of s net Paramters: ", count_parameters(s_net))
    s_net_opt = optim.Adam(s_net.parameters(), lr=args.s_lr, weight_decay=args.weight_decay,  amsgrad=True)
    s_net = nn.DataParallel(s_net)    
    s_net = s_net.to(device)     
       
   
    lambda1 = lambda epoch: 1 if epoch <200 else (0.2 if epoch <500 else 0.01 )
    scheduler = LambdaLR(s_net_opt, lr_lambda= lambda1)    
    
    ds_warped = DataSetWrapper(args.dataset, args.ds_dir, args.comp_bs, None, args.nw, mode='discriminator')
    loader1 , loader2 = ds_warped.get_loaders()
        

    gamma_s = f'gammaNeg{args.gamma_neg}' if args.gamma_neg != -1 else f'gammaNeg{args.min_gamma_neg}-{args.max_gamma_neg}'
    exp_num = f'{args.dataset}_zDim{args.s_dim}_bs{args.bs}_zlr{args.s_lr}_{gamma_s}_SemPotion{100.0/args.div}%'
    
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir('runs'):
        os.mkdir('runs')    
        
    if vis == True:
        writer = SummaryWriter(f'runs/cifar_{exp_num}')     

    for epoch in range(args.epoch+1):
        best_dim = train(epoch, args, loader1, loader2, s_net, s_net_opt)   
        scheduler.step()
        if epoch % 50 == 0:
            torch.save(
                 {'model': s_net.module.state_dict(), 'args': args , 'best_dim': best_dim},
                f'checkpoint/{exp_num}_s_net_{epoch}.pt',
            )
    
    if vis == True:
        writer.close()    

 
    

    