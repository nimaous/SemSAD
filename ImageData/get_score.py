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
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from torchvision.utils import make_grid
from PIL import Image, ImageFilter
from torchvision import transforms


class Get_Score(object):
    def __init__(self, models, loaders, args, div, train_hist=True): 
        self.loader, self.ind_loader, self.ood_loader = loaders
        self.h_net, self.s_net = models  
        self.max_size1, self.max_size2 = int(args.train_size), int(args.test_size)
        self.norm = torch.nn.PairwiseDistance(p=2)        
        self.s_dev = args.s_dev
        self.h_dev = args.h_dev
        self.dir = args.save_path
        self.args = args
        self.ind_name = args.dataset
        self.ood_name = args.ood_dataset
        self.div = div
        self.train_hist = train_hist
        
    def matrix_preparation(self, max_size, data_loader):        
        x_t1_lst = []
        x_t2_lst = []
        x_o_lst = []
        x_n_lst = []
        h_lst = []
        size_ = 0
        for i, (x, _) in enumerate(data_loader):
            x_n, x_o, x_t1, x_t2 = x
            h = self.h_net(x_o.to(f'cuda:{self.h_dev[0]}'))
            h_normalized = h/self.norm(h, torch.zeros_like(h)).view(-1, 1)
            h_lst.append(h_normalized)
            size_ += x_t1.size(0) 
            x_t1_lst.append(x_t1)
            x_t2_lst.append(x_t2)
            x_o_lst.append(x_o)
            x_n_lst.append(x_n)
            if size_ == max_size:
                break
        _, h_dim = h.size()
        bs, c, width, height = x_t1.size()  

        x_t1_total = torch.cat(x_t1_lst, dim=0)
        x_t2_total = torch.cat(x_t2_lst, dim=0)
        x_o_total = torch.cat(x_o_lst, dim=0)
        x_n_total = torch.cat(x_n_lst, dim=0)
        h_total = torch.cat(h_lst, dim=0)
        return h_total, x_o_total, x_n_total, x_t1_total, x_t2_total
        
    def score_function(self,):
        best_match_ind_train, best_match_ind, best_match_ood = self.find_best_match()
        self.score_ind_train = self.s_net(torch.cat((best_match_ind_train,
                                                     self.x_ind_train), 
                                                    dim=1).to(f'cuda:{self.s_dev[0]}') ).sum(-1)
        self.score_ind = self.s_net(torch.cat((best_match_ind, 
                                               self.x_ind), 
                                              dim=1).to(f'cuda:{self.s_dev[0]}')).sum(-1)
        self.score_ood = self.s_net(torch.cat((best_match_ood,
                                               self.x_ood),
                                              dim=1).to(f'cuda:{self.s_dev[0]}')).sum(-1)
        
        l = len(self.x_ind_train)
        rnd_idx = list(range(l))
        random.shuffle(rnd_idx)
        self.x_pos_ref = torch.cat((self.x_ind_train[0:l//self.div], 
                                    self.x_t1[l//self.div:]), dim=0)[rnd_idx] 
        self.x_pos_match = torch.cat((best_match_ind_train[0:l//self.div],
                                      self.x_t2[l//self.div:]), dim=0)[rnd_idx]
        
        self.score_ind_train_pos = self.s_net(torch.cat((self.x_pos_ref, 
                                                         self.x_pos_match), 
                                                        dim=1).to(f'cuda:{self.s_dev[0]}')).sum(-1)
        self.score_ind_train_neg = self.s_net(torch.cat((self.x_pos_ref,
                                                         self.x_ind_train_neg), 
                                                        dim=1).to(f'cuda:{self.s_dev[0]}')).sum(-1)
                
        self.calculate_aucroc(self.score_ind, self.score_ood, 
                              self.ind_name+'_'+self.ood_name)
        self.dist_plot()
        return self.score_ind_train, self.score_ind, self.score_ood 
    
    def find_best_match(self,): 
        h, x, x_n, self.x_t1, self.x_t2 = self.matrix_preparation(self.max_size1, 
                                                                  self.loader)
        print("training sample size", h.size())
        h_d, x_d = h[self.max_size2:], x[self.max_size2:]
        print("shrunk sample size", h_d.size())
        self.h_ind_train, self.x_ind_train = h[0:self.max_size2], x[0:self.max_size2] 
        self.h_ind, self.x_ind, _, _, _ = self.matrix_preparation(self.max_size2,
                                                                  self.ind_loader)
        self.h_ood, self.x_ood, _, _, _ = self.matrix_preparation(self.args.ood_size, 
                                                                  self.ood_loader)
        
        self.x_ind_train_neg = x_n[self.max_size2:self.max_size2*2]
        
        best_indx_ind_train = self._best_indices(self.h_ind_train, h_d)
        best_indx_ind = self._best_indices(self.h_ind, h)        
        best_indx_ood = self._best_indices(self.h_ood, h)
                
        best_match_ind_train = x_d[best_indx_ind_train]
        best_match_ind = x[best_indx_ind]
        best_match_ood = x[best_indx_ood]
                
        self.best_match_plot(self.x_ind, best_match_ind, self.ind_name)
        self.best_match_plot(self.x_ood, best_match_ood, self.ood_name)            
        return best_match_ind_train, best_match_ind, best_match_ood
    
    def _best_indices(self, inp1, inp2):
        print(inp1.size())
        print(inp2.size())
        similarity = (torch.matmul(inp1, inp2.transpose(1, 0)))
        print(similarity.size())
        _, best_indices = similarity.topk(1, dim=1) 
        best_indices = best_indices.view(inp1.size(0))
        return best_indices
            
    def calculate_aucroc(self, s_1, s_2, name): 
        file = open(self.dir + name+"_auroc.txt","a+")    
        l1 = torch.zeros(s_1.size(0))
        l2 = torch.ones(s_2.size(0))
        label = torch.cat((l1, l2),dim=0).view(-1, 1).cpu()
        scores = torch.cat((s_1, s_2), dim=0).cpu()
        FPR, TPR, _ = roc_curve(label, scores, pos_label=0)
        file.write("AUC :{} \r\n".format(auc(FPR, TPR)))        
        file.close()  
                
    def best_match_plot(self, x_ref, x_best, title):
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.imshow(make_grid(x_ref[0:40].cpu()).permute(1, 2, 0))
        plt.title('x')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(make_grid(x_best[0:40].cpu()).permute(1, 2, 0))
        plt.title(f'best_neighbour')
        plt.axis('off')
        plt.savefig(self.dir + title + '.png') 
        
    def dist_plot(self,):
        plt.figure()
        num_bins = 100  
        if self.train_hist:
            plt.figure()
            plt.hist(self.score_ind_train.cpu(), bins=num_bins, alpha=0.8)
            plt.hist(self.score_ind.cpu(), bins=num_bins, alpha=0.8)
            plt.hist(self.score_ood.cpu(), bins=num_bins, alpha=0.8)
            plt.hist(self.score_ind_train_pos.cpu(), 
                     bins=num_bins, alpha=0.8) 
            plt.hist(self.score_ind_train_neg.cpu(), 
                     bins=num_bins, alpha=0.8)                    
            plt.xlabel("OOD score", fontsize=10)
            plt.ylabel("Frequency", fontsize=10)
            plt.legend([self.ind_name+'-train', self.ind_name+'-test', 
                        self.ood_name+'ood-test', self.ind_name+'-train_pos', 
                        self.ind_name+'-train_neg'], loc='upper right')
            plt.savefig(self.dir+'train_hist.png')
        plt.figure()
        plt.hist(self.score_ind.cpu(), bins=num_bins, alpha=0.8)
        plt.hist(self.score_ood.cpu(), bins=num_bins, alpha=0.8) 
        plt.xlabel("OOD score", fontsize=10)
        plt.ylabel("Frequency", fontsize=10)
        plt.legend([self.ind_name+'-test', self.ood_name+'-test'], loc='upper right')
        plt.savefig(self.dir+'hist.png')       
