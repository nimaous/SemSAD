import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, in_channel, in_dim , out_dim, h_net=False):
        super(ConvEncoder, self).__init__()
        if h_net:
            self.trunk = nn.Sequential(
                    nn.Conv1d(in_channel , 2*in_channel , 
                              kernel_size=3, padding=1),
                    nn.BatchNorm1d(2*in_channel),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(2*in_channel , 4*in_channel ,
                              kernel_size=3, padding=1),
                    nn.BatchNorm1d(4*in_channel),
                    nn.ReLU(inplace=True),  
                    nn.Conv1d(4*in_channel , 8*in_channel , 
                              kernel_size=3, padding=1),
                    nn.BatchNorm1d(8*in_channel),
                    nn.ReLU(inplace=True), 
                    nn.Conv1d(8*in_channel , 16*in_channel ,
                              kernel_size=3, padding=1),
                    nn.BatchNorm1d(16*in_channel),
                    nn.ReLU(inplace=True),              
                )
            self.head = nn.Linear(16*in_channel*in_dim , out_dim)
        else:
            self.trunk = nn.Sequential(
                    nn.Conv1d(in_channel , 2*in_channel ,
                              kernel_size=3, padding=1),
                    nn.BatchNorm1d(2*in_channel, track_running_stats=False),
                    nn.ELU(inplace=True),
                    nn.Conv1d(2*in_channel , 4*in_channel , 
                              kernel_size=3, padding=1),
                    nn.BatchNorm1d(4*in_channel, track_running_stats=False),
                    nn.ELU(inplace=True),  
                    nn.Conv1d(4*in_channel , 8*in_channel ,
                              kernel_size=3, padding=1),
                    nn.BatchNorm1d(8*in_channel, track_running_stats=False),
                    nn.ELU(inplace=True), 
              
                )  
            self.head = nn.Linear(8*in_channel*in_dim , out_dim)
    def forward(self, x):
        bs = x.size(0)
        trunk = self.trunk(x)
        out = self.head(trunk.view(bs, -1))       
        return out
    

 
    
