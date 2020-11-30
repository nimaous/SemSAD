import os
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter
from torchvision import  transforms


import random


class DataSetWrapper(object):

    def __init__(self, dataset_name, 
                 dataset_directory, 
                 batch_size1, 
                 batch_size2 , 
                 num_workers,
                 mode,
                 train = True,
                 shuffle= True):        
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
            ds1 = datasets.SVHN(root=self.ds_dir, split='test' if not self.train else 'train' ,
                               transform = SampleTransform(t_neg, t_org, t_pos , mode=self.mode), download=True)        
        
        if self.ds_name == 'cifar10':
            ds1 = datasets.CIFAR10(root=self.ds_dir, train=self.train, 
                               transform = SampleTransform(t_neg, t_org, t_pos , mode=self.mode), download=True)  
            
        if self.ds_name == 'cifar100':
            ds1 = datasets.CIFAR100(root=self.ds_dir, train=self.train,
                                    transform = SampleTransform(t_neg, t_org, t_pos , mode=self.mode), download=True)             
                
        if self.mode == 'auroc':            
            loader1 = DataLoader(ds1, batch_size=self.bs1, shuffle=self.shuffle, num_workers=self.nw, drop_last=False)        
            loader2 = DataLoader(ds1, batch_size=self.bs2, shuffle=self.shuffle, num_workers=self.nw, drop_last=False) 
        else:
            loader1 = DataLoader(ds1, batch_size=self.bs1, shuffle=self.shuffle, num_workers=self.nw, drop_last=True)        
            loader2 = DataLoader(ds1, batch_size=self.bs2, shuffle=self.shuffle, num_workers=self.nw, drop_last=True)        
        return loader1, loader2

    
    def _get_transformations(self): 
        if self.mode in ['discriminator', 'auroc']:
            if self.ds_name in ['cifar10', 'cifar100', 'svhn']:
                size = 32
                s = 0.5
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
                transforms.RandomResizedCrop(size, scale=(0.4,1)),
                shared_T,
                transforms.ToTensor(),
                #transforms.Normalize(mean, std)
            ])

            t_org =  transforms.Compose([
                transforms.ToTensor(),
            ])        
            return t_neg, t_org, t_pos
        
        if self.mode == 'encoder':
            if self.ds_name in ['cifar10', 'cifar100']:
                size = 32
                s = 0.5
            t_org =  transforms.Compose([
                transforms.ToTensor(),
            ])

            colorJitter = transforms.ColorJitter(
                0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
            )            
            t_1 = transforms.Compose([
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([colorJitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
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
    def __init__(self, t_neg, t_org, t_pos , mode=None):            
        self.t_neg = t_neg
        self.t_pos = t_pos
        self.t_org = t_org
        self.mode = mode 
    def __call__(self, sample):            
        if self.mode in ['discriminator', 'auroc']:
            return self.t_neg(sample), self.t_org(sample), self.t_pos(sample), self.t_pos(sample)
        if self.mode == 'encoder':
            return self.t_org(sample), self.t_pos(sample), self.t_pos(sample)
