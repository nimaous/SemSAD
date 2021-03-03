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

import torch
import torch.nn as nn
import torch.nn.functional as F

norm = torch.nn.PairwiseDistance(p=2)

def best_indices_(inp1, inp2):    
    inp1_normalized = (inp1/norm(inp1, torch.zeros_like(inp1)).view(-1, 1)).squeeze()
    inp2_normalized = (inp2/norm(inp2, torch.zeros_like(inp2)).view(-1, 1)).squeeze()
    similarity = (torch.matmul(inp1_normalized, inp2_normalized.transpose(1, 0)))
    _, best_indices = similarity.topk(1, dim=1) 
    best_indices = best_indices.view(inp1_normalized.size(0))
    return best_indices

def best_match_(z_test, z_train):
    bs, L, C = z_train.size()
    test_repeat = z_test.repeat_interleave(z_train.size(1), dim=0)
    similar_h = torch.matmul(z_train.reshape(bs*L, C).unsqueeze(1), test_repeat.unsqueeze(2)).reshape(bs, L)
    inp1_normalized = F.normalize(z_train.reshape(bs*L, C), dim=1, p=2)
    inp2_normalized = F.normalize(test_repeat, dim=1, p=2)
    similar_h = torch.matmul(inp1_normalized.unsqueeze(1), inp2_normalized.unsqueeze(2)).reshape(bs, L)
    best_indx = F.one_hot(similar_h.argmax(dim=1), num_classes = L).float() #[bs, L]
    best_indx = best_indx.unsqueeze(-1) #[bs, L, 1]
    best_match = (z_train.view(bs, L, C) * best_indx).sum(dim=1)  #[bs, L, 1]  
    return best_match
     
def get_features(args, h_net, x):
    with torch.no_grad():        
        t_1 = 50 #choosing one time step to be fixed for all of the sequences
        z = h_net.module.encoder(x) #[bs, hdim, L]
        z = z.transpose(1, 2) #[bs, L, hdim] 
        z_t1 = z[:, t_1, :].squeeze() #[bs, hdim]
        z_mean = torch.mean(z, dim=1) #[bs, hdim] taking average over L
    return z, z_t1, z_mean

def matrix_preparation_train(args, max_size, data_loader, h_net):
    z_mean_lst = []
    z_lst = []
    for i, (x, label) in enumerate(data_loader):
        x = x.float().unsqueeze(1).to(args.device)
        z, _, z_mean = get_features(args, h_net, x)
        z_mean_lst.append(z_mean)
        z_lst.append(z)
    z_tensor = torch.cat(z_lst, dim=0)
    z_mean_tensor = torch.cat(z_mean_lst, dim=0)
    return  z_tensor, z_mean_tensor

def matrix_preparation(args, max_size, data_loader, h_net):
    label_lst = []
    z_lst = []
    for i, (x, label) in enumerate(data_loader):
        x = x.float().unsqueeze(1).to(args.device)
        _, z_t1, _ = get_features(args, h_net, x)
        z_lst.append(z_t1)
        label_lst.append(label)
    label_tensor = torch.cat(label_lst, dim=0) 
    z_tensor = torch.cat(z_lst, dim=0)
    return  z_tensor, label_tensor

