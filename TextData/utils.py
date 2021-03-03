
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

import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import sys
import os
from sklearn.metrics import roc_curve, auc
from torchvision.utils import make_grid

norm = torch.nn.PairwiseDistance(p=2)  
criterion = torch.nn.CrossEntropyLoss()

def calculate_aucroc(s_1, s_2, fold_path, name): 
    file = open(fold_path+  f"{name}_auroc.txt", "a+")    
    l1 = torch.zeros(s_1.size(0))
    l2 = torch.ones(s_2.size(0))
    label = torch.cat((l1, l2), dim=0).view(-1, 1).cpu()
    scores = torch.cat((s_1, s_2), dim=0).cpu()
    FPR, TPR, _ = roc_curve(label, scores, pos_label = 0)
    file.write("AUC :{} \r\n".format(auc(FPR, TPR)))        
    file.close()  

def dist_plot(fold_path,
              score_lst,
              name_lst): 
    with torch.no_grad():
        plt.figure()
        num_bins=100 
        plt.figure()
        print("plotting")
        for score in score_lst:
            plt.hist(score.cpu().tolist(), bins=num_bins, alpha=0.8)                  
        plt.xlabel("OOD score", fontsize=10)
        plt.ylabel("Frequency", fontsize=10)
        plt.legend(name_lst, loc='upper right')        
        plt.savefig(fold_path+'dist_hist.png')


def nce(h, h_tild, temprature=0.5):
    bs, h_dim = h.size()      
    label = torch.arange(bs).to(h.device)    
    h_norm = h/norm(h, torch.zeros_like(h)).view(-1, 1) #[bs, h]
    h_tild_norm = h_tild/norm(h_tild, 
                              torch.zeros_like(h)).view(-1, 1) #[bs, h]
    logits = torch.matmul(h_norm.view(-1, h_dim), 
                          h_tild_norm.view(-1, h_dim).transpose(-1, -2))/temprature 
    loss = criterion(logits, label)
    with torch.no_grad():
        accuracy = torch.eq(
            torch.argmax(torch.softmax(logits, dim = 1), dim = 1),
            label).float().mean()            
    return loss, accuracy 



            
            
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



class ListDataset(Dataset):
    def __init__(self, lst1, lst2):
        self.lst1 = lst1
        self.lst2 = lst2

    def __getitem__(self, index):
        return self.lst1[index], self.lst2[index]

    def __len__(self):
        return len(self.lst1)
