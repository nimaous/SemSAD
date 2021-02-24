
import torch
from torch import nn
import torch.distributions as tdist
import numpy as np
import torch.nn.functional as F
import sys
import os

norm = torch.nn.PairwiseDistance(p=2)  
criterion = torch.nn.CrossEntropyLoss()

def nce(h, h_tild, temprature=0.5):
    bs, h_dim = h.size()      
    label = torch.arange(bs).to(h.device)    
    h_norm = h/norm(h, torch.zeros_like(h)).view(-1,1) #[bs, h]
    h_tild_norm = h_tild/norm(h_tild, torch.zeros_like(h)).view(-1,1) #[bs, h]
    logits = torch.matmul( h_norm.view(-1, h_dim), 
                         h_tild_norm.view(-1, h_dim).transpose(-1, -2))/temprature 
    loss = criterion(logits, label)
    with torch.no_grad():
        accuracy = torch.eq(
            torch.argmax(torch.softmax(logits, dim = 1), dim = 1),
            label).float().mean()            
    return loss, accuracy 


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


            
            
            
           