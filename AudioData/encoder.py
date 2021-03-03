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


class Encoder(nn.Module):
    def __init__(self, seq_len, output_channel):
        super(Encoder, self).__init__()
        self.seq_len = seq_len
        self.similarity = nn.CosineSimilarity(dim=1)
        self.norm = torch.nn.PairwiseDistance(p=2)
        self.ch = output_channel
        self.encoder = nn.Sequential(
            nn.Conv1d(1, self.ch, kernel_size=10, stride=5, padding=3, bias=False),
            nn.BatchNorm1d(self.ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.ch, self.ch, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(self.ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.ch, self.ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(self.ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.ch, self.ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(self.ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.ch, self.ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(self.ch),
            nn.ReLU(inplace=True)
        )

        self.lsoftmax = nn.LogSoftmax(dim=1) 

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x, tau):
        bs = x.size()[0]
        t_1 = torch.randint(int(self.seq_len/160-tau), (1,)).long() 
        t_2 = t_1 + tau
        z = self.encoder(x) #[bs, 256, 128]
        z = z.transpose(1, 2) #[bs, 128, 256]         
        z_t1 = z[:, t_1, :].squeeze() 
        z_t1_normalized = (z_t1/self.norm(z_t1, torch.zeros_like(z_t1)).view(-1, 1))
        z_t2 = z[:, t_2, :].squeeze() 
        z_t2_normalized = (z_t2/self.norm(z_t2, torch.zeros_like(z_t2)).view(-1, 1))
        total = torch.mm(z_t1_normalized, torch.transpose(z_t2_normalized, 0, 1)) 
        nce = self.lsoftmax(total).diag().sum() 
        nce /= -1.*bs
        return nce
