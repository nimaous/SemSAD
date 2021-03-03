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
from torch import nn, optim
from torch.utils.data import DataLoader
from text_dataset import TextDataset
from utils import  count_parameters, ListDataset, dist_plot, calculate_aucroc
from encoder import  ConvEncoder
from simple_tokenization import FullTokenizer

import numpy as np
vis = True
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    vis = False 
    

wt = 0.001
bert_dev =  [0, 1, 2, 3]
s_dev= [2, 3]
h_dev= [1, 2, 3]
norm = torch.nn.PairwiseDistance(p=2) 
uda_w_tokenizer = FullTokenizer(vocab_file=None)

def get_bert_output(x):
    with torch.no_grad():
            x_dict = tokenizer.batch_encode_plus(x, truncation=True, 
                                                  max_length= s_args.max_length, 
                                                  padding='max_length',
                                                  return_tensors="pt").to(f'cuda:{bert_dev[0]}')  
            out = bert(**x_dict)
            hs = out[2] #[13, bs, L, 768]
            hs = hs[-1] + hs[-2] + hs[-3] + hs[-4]
            tokens_sum = hs[:, 1:].sum(dim=1).unsqueeze(1) #[bs, 1 , 768]
            cls = hs[:, 0].unsqueeze(1) #[bs, 1, 768]            
            return torch.cat((tokens_sum, cls), dim=1)#[bs,2,768]  

def generate_normalized_h(loader, bert, h_net, h_args, data_size):    
    with torch.no_grad():
        norm_h = [] 
        x_lst = []
        label_lst = []
        for i , (x, _, _, _, _, label) in enumerate(loader):
            x_embedd = get_bert_output(x)   
            h = h_net(x_embedd.to(f'cuda:{h_dev[0]}') ).view(-1, h_args.h_dim)
            norm_h.append(h/norm(h, torch.zeros_like(h)).view(-1, 1))
            x_lst.extend(list(x))            
            label_lst.append(label)
        return torch.cat(norm_h, dim=0), x_lst, torch.cat(label_lst, dim=0)

def find_best_match(train_x , train_l, test_h, train_h):
    test_h = test_h.to('cpu')
    train_h = train_h.to('cpu')
    l = train_h.size(0)
    with torch.no_grad():
        score_mat =  torch.matmul(test_h, train_h.transpose(1,0))
        _, best_indices = score_mat.topk(1, dim=1) 
        best_indices = best_indices.view(test_h.size(0))
        return [train_x[i] for i in best_indices.tolist()], train_l[best_indices]
    
    
def calculate_score(x_ref, ref_label, x_cond , s_args, exp_name, title, save=False):
    tmp_ds = ListDataset(x_ref, x_cond)
    tmp_loader = DataLoader(tmp_ds, batch_size=200, shuffle=False, num_workers=24, drop_last=False) 
    z_f = [] 
    with torch.no_grad():    
        for i, (x1,x2) in enumerate(tmp_loader) :
            x1_rep = get_bert_output(x1)
            x2_rep = get_bert_output(x2)
            z = s_net(torch.cat((x1_rep, x2_rep), dim=1).to(f'cuda:{s_dev[0]}')).mean(-1)
            z_f.append(z) 
        z_f = torch.cat(z_f, dim=0)  
    return z_f.cpu()

 
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='AG_NEWS')    
    parser.add_argument('--bs', type=int, default=300)
    parser.add_argument('--epoch', type=int, default=500)       
    parser.add_argument('--h_net_path', type=str, default='')    
    parser.add_argument('--s_net_path', type=str, default='')    
    parser.add_argument('--weight_decay', type=float, default=1e-6)        
    args = parser.parse_args()
    
    h_net_ckpt = torch.load(args.h_net_path)    
    s_net_ckpt = torch.load(args.s_net_path)
    h_args = h_net_ckpt['args']
    s_args = s_net_ckpt['args']     
    
    
    h_net = ConvEncoder(2, 768 , h_args.h_dim, h_net=True)
    h_net.load_state_dict(h_net_ckpt['model'])
    h_net= nn.DataParallel(h_net, device_ids=h_dev)            
    h_net = h_net.to(f'cuda:{h_dev[0]}') 
         

    s_net = ConvEncoder(4, 768 , s_args.s_dim)
    s_net.load_state_dict(s_net_ckpt['model'])
    s_net = nn.DataParallel(s_net, device_ids=s_dev)    
    s_net = s_net.to(f'cuda:{s_dev[0]}') 
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert =  BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True,).to(f'cuda:{bert_dev[0]}')
    bert = nn.DataParallel(bert, device_ids=bert_dev) 
    
    s_net.eval()
    h_net.eval()
    bert.eval()    
    
    tp = "tokens_train" if os.path.isfile("tokens_train") else None
    dsp = "tf-idf-dict_train" if os.path.isfile("tf-idf-dict_train") else None
    train_ds = TextDataset(dataset_name=args.dataset_name,  
                             sufix_name="train", 
                             tokens_path=tp,
                             data_stats_path=dsp,
                             tf_idf_pr=0.0,
                             th_sus_prt=0.0, 
                             rnd_tf_idf_pr=0.0,
                             rnd_th_sus_prt=0.0,                            
                             out_cls=[0])
    train_loader = DataLoader(train_ds, 
                              batch_size=args.bs,
                              shuffle=False, 
                              num_workers=24, 
                              drop_last=False)
    
    test_tp = "tokens_test" if os.path.isfile("tokens_test") else None
    test_dsp = "tf-idf-dict_test" if os.path.isfile("tf-idf-dict_test") else None    
    test_ds = TextDataset(dataset_name=args.dataset_name, 
                            train=False,
                            sufix_name="test", 
                            tokens_path=test_tp,
                            data_stats_path=test_dsp,                          
                            tf_idf_pr=0.0,
                            th_sus_prt=0.0, 
                            rnd_tf_idf_pr=0.0,
                            rnd_th_sus_prt=0.0,                           
                            out_cls=[0])
    test_loader = DataLoader(test_ds, 
                             batch_size=args.bs,
                             shuffle=False, 
                             num_workers=24,
                             drop_last=False)   
    
    ood_tp = "tokens_ood" if os.path.isfile("tokens_ood") else None
    ood_dsp = "tf-idf-dict_ood" if os.path.isfile("tf-idf-dict_ood") else None      
    ood_ds = TextDataset(dataset_name=args.dataset_name,
                            train=False,
                            sufix_name="ood", 
                            tokens_path=ood_tp,
                            data_stats_path=ood_dsp,                            
                            tf_idf_pr=0.0,
                            th_sus_prt=0.0, 
                            rnd_tf_idf_pr=0.0,
                            rnd_th_sus_prt=0.0,                          
                            out_cls=[1, 2, 3])
    ood_loader = DataLoader(ood_ds, 
                            batch_size=args.bs,
                            shuffle=False, 
                            num_workers=24, 
                            drop_last=False)    
    
    train_aug = TextDataset(dataset_name=args.dataset_name,  
                                sufix_name="train", 
                                tokens_path=tp,
                                data_stats_path=dsp,
                                tf_idf_pr=s_args.tf_idf_pr,
                                th_sus_prt=s_args.th_sus_prt, 
                                rnd_tf_idf_pr=s_args.rnd_tf_idf_pr,
                                rnd_th_sus_prt=s_args.rnd_th_sus_prt,                             
                                sync_aug=s_args.sync_aug,
                                out_cls=[0]) 
   
    
    train_aug_loader = DataLoader(train_aug, 
                                  batch_size=len(test_ds),
                                  shuffle=False, 
                                  num_workers=24,
                                  drop_last=False)     



     
    train_h, train_x, train_l = generate_normalized_h(train_loader, bert, 
                                                      h_net, h_args, len(train_ds))
    print("train normalization done.....")
    test_h, test_x, test_l = generate_normalized_h(test_loader, bert, 
                                                   h_net, h_args, len(test_ds))
    print("test normalization done.....")    
    ood_h, ood_x, ood_l = generate_normalized_h(ood_loader, bert, 
                                                h_net, h_args, len(ood_ds))
    print("ood normalization done.....")    
    
    print(len(train_x))
    print(len(test_x))
    print(len(ood_x))
    print(train_h.size())
    print(test_h.size())
    print(ood_h.size())
    
    best_match_test, best_match_test_l = find_best_match(train_x, train_l,
                                                         test_h, train_h)
    best_match_ood, best_match_ood_l  = find_best_match(train_x, train_l, 
                                                        ood_h, train_h)  
    best_match_train, best_match_train_l = find_best_match(train_x[len(test_ds):], 
                                                           train_l[len(test_ds):],
                                                           train_h[0:len(test_ds)], 
                                                           train_h[len(test_ds):])     
    

    exp_name = f"discriminator_evaluation_results"
    if not os.path.isdir(exp_name):
        os.mkdir(exp_name)

    itr = iter(train_aug_loader)
    x_ref, x_pos, x_neg, _, _, l_pos = next(itr)

    
    
    z_pos = calculate_score(x_ref, l_pos, x_pos, 
                              s_args, exp_name,
                              "z_pos", save=False)
    z_neg = calculate_score(x_ref, l_pos, x_neg, 
                               s_args, exp_name, 
                               "z_neg", save=False)    
    z_test = calculate_score(best_match_test, 
                                best_match_test_l,
                                test_x, s_args, exp_name, 
                                "z_test", save=True)
    z_ood = calculate_score(best_match_ood,  
                              best_match_ood_l,
                              ood_x, s_args, exp_name, 
                              "z_ood", save=True)

    calculate_aucroc(z_test, z_ood, exp_name+"/", "test-ood")
    dist_plot(exp_name+"/", [z_test, z_pos, z_neg, z_ood],
                             ["test", "train_pos", "train_neg","ood"])    

    

    
