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
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import random
from tqdm import tqdm
import numpy as np
from data_reader import RawDataset
from utils import count_parameters
from encoder import Encoder
from Transformer import CNN

vis = False
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    vis = False

wt = 0.001
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def avg_loss(args, a_pos, a_neg, GN):
    log_sigmoid = nn.LogSigmoid()
    loss = torch.mean(-(log_sigmoid(a_pos) + GN*log_sigmoid(-a_neg)), dim=0) # average over samples
    min_loss, min_dim = loss.topk(1, largest=False) # select smallest dim  
    loss_avg = loss.mean() #average over dimensions
    return loss_avg, min_dim

def get_features(args, h_net, x):
    with torch.no_grad():
        batch = x.size()[0]
        t_1 = torch.randint(int(128-args.tau), (1,)).long() 
        t_2 = t_1 + args.tau
        z = h_net.module.encoder(x)
        z = z.transpose(1,2)       
        ref = z[:,t_1,:].view(batch,args.hdim)
        pos = z[:,t_2,:].view(batch,args.hdim)
    return ref.unsqueeze(1), pos.unsqueeze(1)

def train(epoch, args, train_loader1, train_loader2, h_net, s_net, s_net_opt):
    
    s_net.train()    
    x_neg = next(iter(train_loader2))
    x_neg = x_neg.float().unsqueeze(1).to(device) # add channel dimension
    _, z_neg = get_features(args, h_net, x_neg)
    loader = tqdm(train_loader1)
    for batch_idx, x in enumerate(loader):
        GN = np.random.randint(args.min_gamma_neg, args.max_gamma_neg)
        s_net_opt.zero_grad()
        x = x.float().unsqueeze(1).to(device) # add channel dimension
        bs = x.size(0)
        
        z_ref, z_pos = get_features(args, h_net, x)               
        s_pos = s_net(torch.cat((z_ref, z_pos), dim=1))
        s_neg = s_net(torch.cat((z_ref, z_neg[0:bs]), dim=1)) 
        a_pos = s_pos - np.log(GN)
        a_neg = s_neg - np.log(GN)         
        loss, best_dim = avg_loss(args, a_pos, a_neg, GN)
        loss.backward()       
        s_net_opt.step() 
            
        rnd_idx = list(range(x.size(0)))        
        random.shuffle(rnd_idx)  
        z_neg = z_pos[rnd_idx].clone() 
        
        loader.set_description(f'Train epoch: {epoch + 1}; loss: {loss.item():.5f}')
   
    if vis == True:
        with torch.no_grad():  
            writer.add_scalar("Loss", loss.item(), global_step=epoch, walltime=wt)   
            writer.add_histogram('Positive Pairs', a_pos.cpu().numpy(), global_step=epoch, walltime=wt)
            writer.add_histogram('Negative Pairs', a_neg.cpu().numpy(), global_step=epoch, walltime=wt)           
                       
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_raw', default='dataset/train-Librispeech.h5')
    parser.add_argument('--train_list', default='list/male_train_split.txt')
    parser.add_argument('--h_net_path', type=str, default='path to the trained h')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gamma_neg', type=int, default=-1) 
    parser.add_argument('--min_gamma_neg', type=int, default=1)                 
    parser.add_argument('--max_gamma_neg', type=int, default=10) 
    parser.add_argument('--audio_window', type=int, default=20480)
    parser.add_argument('--s_lr', type=float, default=5e-5)   
    parser.add_argument('--weight_decay', type=float, default=1e-6) 
    parser.add_argument('--sdim', type=int, default=64)
    parser.add_argument('--hdim', type=int, default=256)
    parser.add_argument('--len_seg', type=int, default=16)
    parser.add_argument('--input_channel', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--tau', type=int, default=5)
    args = parser.parse_args()
   
    ### get the dataloaders ###
    training_set = RawDataset(args.train_raw, args.train_list, args.audio_window)
    train_loader1 = data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True) 
    train_loader2 = data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    
    ### load the encoder checkpoint ###
    assert args.h_net_path is not None
    h_net_ckpt = torch.load(args.h_net_path)
    h_net = Encoder(args.audio_window, args.hdim) 
    h_net.load_state_dict(h_net_ckpt['model'])
    h_net = nn.DataParallel(h_net)
    h_net = h_net.to(device)
    h_net.eval()     
  
    ### initialise the discriminator ### 
    s_net = CNN(args.input_channel, 256, args.len_seg, args.sdim, args.num_layers)
    s_net_opt = optim.Adam(s_net.parameters(), lr=args.s_lr, weight_decay=args.weight_decay, amsgrad=True)
    s_net = nn.DataParallel(s_net)    
    s_net = s_net.to(device)  
    
    ### manage directory to save the checkpoint ###
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint') 
    if not os.path.isdir('runs'):
        os.mkdir('runs') 
    gamma = f'gammaNeg{args.gamma_neg}' if args.gamma_neg != -1 else f'gammaNeg{args.min_gamma_neg}-{args.max_gamma_neg}'    
    exp_num = f's_tau_{args.tau}_hdim{args.hdim}_sdim{args.sdim}_{gamma}' 
        
    if vis == True:
        writer = SummaryWriter(f'runs/{exp_num}')
        
    for epoch in range(args.epochs):
        train(epoch, args, train_loader1, train_loader2, h_net, s_net, s_net_opt)
        torch.save(
            {'model': s_net.module.state_dict(), 'args': args},
             f'checkpoint/{exp_num}_2.pt')  
    
    if vis == True:
        writer.close()






