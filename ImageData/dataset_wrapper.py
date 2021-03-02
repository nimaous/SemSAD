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
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFilter
from torchvision.datasets import ImageFolder
from torchvision import transforms
import random


class DataSetWrapper(object):
    def __init__(self, dataset_name, 
                 dataset_directory, 
                 batch_size1, 
                 batch_size2, 
                 num_workers,
                 mode,
                 train = True,
                 type = 'trainset',                 
                 shuffle= True,
                 ti_for_ci = False,):  

        self.type = type
        self.ti_for_ci = ti_for_ci
        self.ds_name = dataset_name
        self.ds_dir = dataset_directory
        self.bs1 = batch_size1  
        if batch_size2 == None:
            self.bs2 = batch_size1
        else:
            self.bs2 = batch_size2
        self.nw = num_workers
        self.mode = mode
        self.train = train
        self.shuffle = shuffle

        
    def get_loaders(self):      

        t_neg , t_org, t_pos = self._get_transformations()

        if self.ds_name == 'svhn':
            ds1 = datasets.SVHN(root=self.ds_dir, split='test' if not self.train else 'train',
                                transform = SampleTransform(t_neg, t_org, t_pos, mode=self.mode), 
                                download=True)        
        
        if self.ds_name == 'cifar10':
            ds1 = datasets.CIFAR10(root=self.ds_dir, train=self.train, 
                                   transform = SampleTransform(t_neg, t_org, t_pos, mode=self.mode), 
                                   download=True)  
            
        if self.ds_name == 'cifar100':
            ds1 = datasets.CIFAR100(root=self.ds_dir, train=self.train,
                                    transform = SampleTransform(t_neg, t_org, t_pos, mode=self.mode), 
                                    download=True)
            
        if self.ds_name == 'tiny_imagenet':   
            ds1 = ImageFolder(self.ds_dir, 
                              transform=SampleTransform(t_neg, t_org, 
                                                        t_pos, mode=self.mode))
            
        print(f"Dataset size is: {len(ds1)} ")        
        if self.mode == 'auroc':            
            loader1 = DataLoader(ds1, batch_size=self.bs1, 
                                 shuffle=self.shuffle, 
                                 num_workers=self.nw, 
                                 drop_last=False)        
            loader2 = DataLoader(ds1, batch_size=self.bs2,
                                 shuffle=self.shuffle, 
                                 num_workers=self.nw,
                                 drop_last=False) 
        else:
            loader1 = DataLoader(ds1, batch_size=self.bs1, 
                                 shuffle=self.shuffle,
                                 num_workers=self.nw, 
                                 drop_last=True,
                                  pin_memory=True)        
            loader2 = DataLoader(ds1, batch_size=self.bs2, 
                                 shuffle=self.shuffle, 
                                 num_workers=self.nw,
                                 drop_last=True,
                                  pin_memory=True)        
        return loader1, loader2

    
    def _get_transformations(self): 

        if self.ds_name in ['cifar10', 'cifar100', 'svhn']:
            size = 32
            s = 0.5
        if self.ds_name in ['tiny_imagenet']:
            if self.ti_for_ci:
                size = 32
            else:
                size = 64
            s = 0.5
        print(self.ds_name)
        print(size)
        if self.mode in ['discriminator', 'auroc']:              
            colorJitter = transforms.ColorJitter(
                0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
            ) 
            
            shared_T =  transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([colorJitter], p=0.8),
            ])            
            t_neg = transforms.Compose([
                transforms.RandomResizedCrop(size),
                shared_T,             
                GaussianBlur(),
                transforms.ToTensor(),        
            ])   
            t_pos = transforms.Compose([
                transforms.RandomResizedCrop(size, scale=(0.4, 1)),
                shared_T,
                transforms.ToTensor(),
            ])

            t_org =  transforms.Compose([
                transforms.RandomResizedCrop(size, scale=(1, 1)),
                transforms.ToTensor(),
            ])        
            return t_neg, t_org, t_pos
        
        if self.mode == 'encoder':
            t_org =  transforms.Compose([
                transforms.RandomResizedCrop(size, scale=(1, 1)),
                transforms.ToTensor(),
            ])

            colorJitter = transforms.ColorJitter(
                0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
            ) 
            if self.ds_name in ['cifar10', 'cifar100', 'svhn', 'tiny_imagenet']:                        
                t_1 = transforms.Compose([
                    transforms.RandomResizedCrop(size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([colorJitter], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),        
                ]) 
            if self.ds_name in ['imagenet' ]:
                t_1 = transforms.Compose([
                    transforms.RandomResizedCrop(size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([colorJitter], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    GaussianBlur(),
                    transforms.ToTensor(),        
                ])                

            return t_1, t_org, t_1



class GaussianBlur(object):
    """Gaussian blur augmentation: https://github.com/facebookresearch/moco/"""

    def __init__(self, sigma=[.1, 2], p=0.5 ):
        self.sigma = sigma
        self.p = p 

    def __call__(self, x):
        rnd = random.uniform(0, 1)
        if rnd > self.p:
            return x 
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
class SampleTransform(object):


    def __init__(self, t_neg, t_org, t_pos, mode=None):            
        self.t_neg = t_neg
        self.t_pos = t_pos
        self.t_org = t_org
        self.mode = mode 

    def __call__(self, sample):            
        if self.mode in ['discriminator', 'auroc']:
            return self.t_neg(sample), self.t_org(sample), 
                   self.t_pos(sample), self.t_pos(sample)
        if self.mode == 'encoder':
            return self.t_org(sample), self.t_pos(sample), self.t_pos(sample)
