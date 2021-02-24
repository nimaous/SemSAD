
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

import wget
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3"
import argparse
from tqdm import tqdm
from transformers import BertTokenizer,BertModel
import torch
import torchtext
from torch import nn, optim
from torch import nn
from torch.utils.data import DataLoader
from text_dataset import TextDataset
from utils import nce,  count_parameters
from encoder import  ConvEncoder

"""  if you get tensorflow related error 
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
"""

import numpy as np
vis = True
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    vis = False 

wt = 0.001
bert_dev =  [0,1,2,3]
h_dev= [1,2,3]


def get_bert_output(x):
    with torch.no_grad():
            x_dict = tokenizer.batch_encode_plus(x, truncation=True, 
                                                  max_length= args.max_length, 
                                                  padding='max_length',
                                                  return_tensors="pt").to(f'cuda:{bert_dev[0]}')  
            out = bert(**x_dict)
            hs = out[2] #[13, bs, L, 768]
            hs = hs[-1] + hs[-2] + hs[-3] + hs[-4]
            tokens_sum = hs[:,1:].sum(dim=1).unsqueeze(1) #[bs, 1 , 768]
            cls = hs[:,0].unsqueeze(1) #[bs, 1, 768]            
            return torch.cat((tokens_sum, cls), dim=1)#[bs,2,768] 

def train(epoch, loader, h_net_opt):
    loader = tqdm(loader)
    h_net.train()
    bert.eval()
    for i , (x1 , x2 , _ , _, _ , label) in enumerate(loader):
        h_net_opt.zero_grad()
        with torch.no_grad():
            x1_embedd = get_bert_output(x1) 
            x2_embedd = get_bert_output(x2)
        
        h1 = h_net(x1_embedd.to(f'cuda:{h_dev[0]}'))
        h2 = h_net(x2_embedd.to(f'cuda:{h_dev[0]}'))
        loss, accuracy = nce(h1, h2, temprature=args.temprature)
        loss.backward()
        h_net_opt.step()
        loader.set_description(
            (
                f' Epoch: {epoch + 1};  Iteration:{i};   Loss: {loss.item()}  '         
            )
        )
        
    if vis == True:
        with torch.no_grad(): 
            writer.add_scalar("NCE Loss", loss.item(), global_step=epoch, walltime=wt) 
            writer.add_scalar("Accuracy", accuracy.item(), global_step=epoch, walltime=wt) 
            if epoch % 10 == 0: 
                f_meta = []
                for sent , l in zip(x1, label):
                    f_meta.append(f"{l} - {sent}")
                writer.add_embedding(tag="h", mat=h1.cpu().numpy(), metadata=f_meta , global_step=epoch)         
                
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='AG_NEWS')
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=100)    
    parser.add_argument('--h_lr', type=float, default=1e-4)   
    parser.add_argument('--h_dim', type=int, default=512)    
    parser.add_argument('--weight_decay', type=float, default=1e-6) 
    parser.add_argument('--th_sus_prt', type=float, default=0.5) 
    parser.add_argument('--tf_idf_pr', type=float, default=0.3)    
    parser.add_argument('--temprature', type=float, default=0.25)
    parser.add_argument('--max_length', type=int, default=100)    
    parser.add_argument('--sync_aug', type=bool, default=True)        
    

    args = parser.parse_args()
    h_net = ConvEncoder(2 , 768 , args.h_dim, h_net=True)    
    print("h_net size" , count_parameters(h_net))
    h_net = nn.DataParallel(h_net, device_ids=h_dev)
    h_net = h_net.to(f'cuda:{h_dev[0]}')     
    
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert =  BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True,).to(f'cuda:{bert_dev[0]}')
    bert = nn.DataParallel(bert, device_ids=bert_dev)
    
    h_net_opt = optim.Adam(h_net.parameters(), lr=args.h_lr, weight_decay=args.weight_decay)    
    tp = "tokens_train" if os.path.isfile("tokens_train") else None
    dsp = "tf-idf-dict_train" if os.path.isfile("tf-idf-dict_train") else None
    ds = TextDataset(dataset_name=args.dataset_name, 
                     sufix_name = "train",
                     th_sus_prt = args.th_sus_prt,
                     tf_idf_pr = args.tf_idf_pr,  
                     tokens_path = tp,   
                     data_stats_path = dsp,                     
                     sync_aug= args.sync_aug,
                     out_cls=[0])
    loader = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=24)
    
    exp_name = f'{args.dataset_name}_hdim_{args.h_dim}'
    
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint') 
    if not os.path.isdir('runs'):
        os.mkdir('runs')      
    
    if vis == True:
        writer = SummaryWriter(f'runs/{exp_name}') 
        
    for epoch in range(args.epoch):
        train(epoch, loader, h_net_opt)
        torch.save(
             {'model': h_net.module.state_dict(), 'args': args},
            f'checkpoint/{exp_name}_h_net.pt',
        )       
    writer.close()
    
    