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
from utils import nce, count_parameters
import numpy as np
from encoder import   ConvEncoder
try:    
    from torch.utils.tensorboard import SummaryWriter
    vis = True
except:
    vis = False 
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

wt = 0.001
norm = torch.nn.PairwiseDistance(p=2) 
bert_dev =  [0, 1, 2, 3]
h_dev= [1, 2, 3]

def get_bert_output(x):
    with torch.no_grad():
            x_dict = tokenizer.batch_encode_plus(x, truncation=True, 
                                                  max_length=h_args.max_length, 
                                                  padding='max_length',
                                                  return_tensors="pt").to(f'cuda:{bert_dev[0]}')  
            out = bert(**x_dict)
            hs = out[2] #[13, bs, L, 768]
            hs = hs[-1] + hs[-2] + hs[-3] + hs[-4]
            tokens_sum = hs[:, 1:].sum(dim=1).unsqueeze(1) #[bs, 1 , 768]
            cls = hs[:, 0].unsqueeze(1) #[bs, 1, 768]            
            return torch.cat((tokens_sum, cls), dim=1)#[bs,768,2] 


def calculate_accuracy(x, label, s, exp_name):
    label.to(x.device)
    x = x/norm(x, torch.zeros_like(x)).view(-1,1)
    A = torch.matmul(x, x.transpose(1,0))
    A = A + (-1e+8) * torch.eye(A.size(0)).to(x.device)
    _, indicies = A.topk(1, dim=1)
    equals = torch.eq(label[indicies].squeeze(), label).long()    
    acc = equals.sum() / float(A.size(0))   
    w_pred_indices = torch.nonzero(1-equals).view(-1).tolist()
    s_p = [s[i] for i in indicies.view(-1).tolist()]
    not_match = [(s[i],s_p[i]) for i in w_pred_indices]
    with open(f'{exp_name}.txt', 'w') as outfile:
        outfile.write(f"Accuracy is {acc}")
        outfile.write("\n")
        for i, item in enumerate(not_match):
            outfile.write(f"{i}_{item[0]} \n {i}-->{item[1]}")
            outfile.write("\n\n\n")
    return acc 

   
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='AG_NEWS')    
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=500)    
    parser.add_argument('--h_lr', type=float, default=3e-4)   
    parser.add_argument('--h_dim', type=int, default=512)    
    parser.add_argument('--h_net_path', type=str, default='')    
    args = parser.parse_args()
    
    h_net_ckpt = torch.load(args.h_net_path)
    h_args = h_net_ckpt['args']    
    exp_name = f'h_results'     
    if vis:
        writer = SummaryWriter(f'runs/{exp_name}') 
    else:
        raise NotImplemented    
        
    h_net = ConvEncoder(2, 768, h_args.h_dim, h_net=True) 
    h_net.load_state_dict(h_net_ckpt['model'])   
    h_net = nn.DataParallel(h_net, device_ids=h_dev)    
    h_net = h_net.to(f'cuda:{h_dev[0]}')
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert =  BertModel.from_pretrained('bert-base-uncased', 
                                      output_hidden_states=True,).to(f'cuda:{bert_dev[0]}')
    bert = nn.DataParallel(bert, device_ids=bert_dev) 
    print("loading h net state dictionary")
    h_net.eval()
    bert.eval()    
    tp = "tokens_test" if os.path.isfile("tokens_test") else None
    dsp = "tf-idf-dict_test" if os.path.isfile("tf-idf-dict_test") else None    
    test_ds = TextDataset(dataset_name=args.dataset_name,
                            train=False,
                            sufix_name="0-out_test", 
                            tokens_path=tp,
                            data_stats_path=dsp,
                            tf_idf_pr=0.0,
                            th_sus_prt=0.0, 
                            out_cls=[0])
    test_loader = DataLoader(test_ds, batch_size=args.bs, 
                             shuffle=False, num_workers=24)        
    test_h_lst = []
    test_label_lst = []
    test_x = []    
    for i , (x, _, _, _, _, label) in enumerate(test_loader):
        with torch.no_grad():
            x_rep = get_bert_output(x)  
            h1 = h_net(x_rep.to(f'cuda:{h_dev[0]}')     )                       
            test_h_lst.append(h1)
            test_label_lst.extend(label.tolist()) 
            test_x.extend(x)
    
    f_meta = []
    for sent, l in zip(test_x, test_label_lst):
        f_meta.append(f"{l} - {sent}")
    print(set(test_label_lst))
    writer.add_embedding(tag="h", 
                         mat=torch.cat(test_h_lst, dim=0).cpu().numpy(), 
                         metadata=f_meta, 
                         global_step=i)  
    writer.close()                             
    test_acc = calculate_accuracy(torch.cat(test_h_lst, dim=0),
                                    torch.tensor(test_label_lst),
                                    test_x,
                                    exp_name)                                   
    print(f"Test Accuracy is: {test_acc}")                                      
                                      
