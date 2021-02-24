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
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import argparse
import random
import numpy as np
from tqdm import tqdm
from utils import count_parameters, ScheduledOptim
from data_reader import RawDataset
from encoder import Encoder

vis = False
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    vis = False

wt = 0.001
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   
def train(epoch, args, train_loader, h_net, h_net_opt):
    
    h_net.train()
    loader = tqdm(train_loader)
    for i, data in enumerate(loader):
        data = data.float().unsqueeze(1).to(device) # add channel dimension
        h_net_opt.zero_grad()
        loss = h_net(data, args.tau)        
        loss.backward()
        h_net_opt.step()
        lr = h_net_opt.update_learning_rate()        
        loader.set_description(f'Train epoch: {epoch}; loss: {loss.item():.5f}') 
        
    if vis == True:
        with torch.no_grad():  
            writer.add_scalar("NCE Loss", loss.item(), global_step=epoch, walltime=wt) 
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_raw', default='dataset/train-Librispeech.h5')
    parser.add_argument('--train_list', default='list/male_train_split.txt')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--n_warmup_steps', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--audio_window', type=int, default=20480)
    parser.add_argument('--tau', type=int, default=5) 
    parser.add_argument('--hdim', type=int, default=256)
    parser.add_argument('--h_lr', type=int, default=1e-3)
    parser.add_argument('--weight_decay', type=int, default=1e-4)
    args = parser.parse_args()
    
    ### initialise the encoder ### 
    h_net = Encoder(args.audio_window, args.hdim)
    h_net = h_net.to(device)
    
    ### get the dataloader ###
    training_set   = RawDataset(args.train_raw, args.train_list, args.audio_window)
    train_loader = data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True) 
     
    ### define the optimiser ###
    h_net_opt = ScheduledOptim(
            optim.Adam(
            filter(lambda p: p.requires_grad, h_net.parameters()), 
            betas=(0.9, 0.98), eps=1e-09, lr=args.h_lr, weight_decay=args.weight_decay, amsgrad=True),
            args.n_warmup_steps)

    ### manage directory to save the checkpoint ###
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint') 
    if not os.path.isdir('runs'):
        os.mkdir('runs')         
    exp_num = f'h_tau_{args.tau}_hdim{args.hdim}' 
        
    if vis == True:
        writer = SummaryWriter(f'runs/{exp_num}')
        
    for epoch in range(1, args.epochs + 1):
        train(epoch, args, train_loader, h_net, h_net_opt) 
        torch.save(
            {'model': h_net.state_dict(), 'args': args},
             f'checkpoint/{exp_num}.pt')
    
    if vis == True:
        writer.close()
        
