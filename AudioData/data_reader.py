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

import numpy as np
import torch
from torch.utils import data
import h5py
from random import randint

        
class RawDataset(data.Dataset):
    def __init__(self, raw_file, list_file, audio_window, train=True):
        self.raw_file  = raw_file 
        self.audio_window = audio_window 
        self.utts = []
        self.train = train
        
        with open(list_file) as f:
            temp = f.readlines()
        self.utts = [x.strip() for x in temp]
        
        self.h5f = h5py.File(self.raw_file, 'r')
        
    def __len__(self):
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index] 
        utt_len = self.h5f[utt_id].shape[0] 
        index = np.random.randint(utt_len - self.audio_window + 1) 
        speaker = int(utt_id.split('-')[0])
        spk_id  = torch.tensor(speaker)
        if self.train:
            return self.h5f[utt_id][index:index+self.audio_window] 
        else:
            return self.h5f[utt_id][:self.audio_window], spk_id
