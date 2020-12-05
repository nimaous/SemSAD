
import torch
from torch import nn
import torch.distributions as tdist
import numpy as np
import torch.nn.functional as F
import sys
import os

def avg_NCE_loss(h, 
                 h_tild , 
                 h_neg=None, 
                 h_neg_tild=None, 
                 similarity=None,
                 temprature=1,):
    nce_loss_tild =  NCE_Loss(h_tild,  
                              h.detach(), 
                              h_neg_tild.detach(), 
                              similarity=similarity,
                              temprature=temprature,
                              )  

    nce_loss =  NCE_Loss(h,  
                         h_tild.detach(), 
                         h_neg.detach(), 
                         similarity=similarity,
                         temprature=temprature,
                         )  

    return nce_loss + nce_loss_tild 


def NCE_Loss(out, target, neg=None, self_neg=False,  similarity=None, temprature=1):
    """
    out = [bs, latent_dim]
    target = [bs, latent_dim]
    neg = [bs, Num_neg, latent_dim]    
    """
    neg_size = neg.size(1)
    bs = out.size(0)
    if similarity is None:
        class_1 = torch.matmul(out.unsqueeze(1), target.unsqueeze(2)).view(bs,1) #[bs,1]
        class_2 = torch.matmul(out.unsqueeze(1), neg.permute(0,2,1)).squeeze() #[bs, neg_size]
    elif isinstance(similarity, nn.CosineSimilarity):
        class_1 = similarity(out, target).view(bs,1)
        class_2 = similarity(out.unsqueeze(-1), neg.permute(0,2,1))#[bs, neg_size]
    elif isinstance(similarity, nn.MSELoss):
        class_1 = -similarity(out, target).sum(dim=1).view(bs,1)
        class_2 = -similarity(out.repeat_interleave(neg_size, dim=0),neg.view(bs*neg_size,-1)).sum(dim=-1)
        class_2 = class_2.view(bs, neg_size)
    else:
        raise NotImplemented 
    data = torch.cat((class_1/temprature, class_2/temprature), dim=1)    
    log_softmax = torch.log_softmax(data, dim=1)+ np.log(float(neg_size+1)) 
    infoNCE_loss = -torch.mean(log_softmax[:,0])  
    return infoNCE_loss



def count_parameters(model):
    """
     returns number of parameters of a given model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


            
            
            
           
