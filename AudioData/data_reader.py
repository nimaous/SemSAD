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