"""
cnn+rnn for 1-d signal data, pytorch version
borrowed from: Shenda Hong, Jan 2020
https://github.com/hsd1503/resnet1d
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        n_classes: number of classes  
    """

    def __init__(self, in_channels, out_channels, n_len_seg, n_classes, num_layers=1, verbose=False):
        super(CNN, self).__init__()
        
        self.n_len_seg = n_len_seg
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.verbose = verbose

        # (batch, channels, length)
        self.cnn = nn.Conv1d(in_channels=self.in_channels, 
                            out_channels=self.out_channels, 
                            kernel_size=16, 
                            stride=2)
        # (batch, channels, length)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.out_channels, 
            nhead=8,
            dim_feedforward=128,
            dropout=0.5,)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers) 
        self.dense = nn.Linear(out_channels, n_classes)
        
    def forward(self, x):

        self.n_channel, self.n_length = x.shape[-2], x.shape[-1]
        assert (self.n_length % self.n_len_seg == 0), "Input n_length should divided by n_len_seg"
        self.n_seg = self.n_length // self.n_len_seg

        out = x
        if self.verbose:
            print(out.shape)

        # (n_samples, n_channel, n_length) -> (n_samples, n_length, n_channel)
        out = out.permute(0, 2, 1)
        if self.verbose:
            print(out.shape)
        # (n_samples, n_length, n_channel) -> (n_samples*n_seg, n_len_seg, n_channel)
        out = out.reshape(-1, self.n_len_seg, self.n_channel)
        if self.verbose:
            print(out.shape)
        # (n_samples*n_seg, n_len_seg, n_channel) -> (n_samples*n_seg, n_channel, n_len_seg)
        out = out.permute(0, 2, 1)
        if self.verbose:
            print(out.shape)
        # cnn
        out = self.cnn(out)
        if self.verbose:
            print(out.shape)
        # global avg, (n_samples*n_seg, out_channels)
        out = out.mean(-1)
        if self.verbose:
            print(out.shape)
        out = out.view(-1, self.n_seg, self.out_channels)
        if self.verbose:
            print(out.shape)
        out = self.transformer_encoder(out)
        if self.verbose:
            print(out.shape)
        out = out.mean(-2)
        if self.verbose:
            print(out.shape)
        out = self.dense(out)
        if self.verbose:
            print(out.shape)
        
        return out
