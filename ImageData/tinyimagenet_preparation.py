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
import os
import argparse
from shutil import copy, copytree
import numpy as np
import random
import shutil
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


def get_annotations_map(val_annotaion_path):
    valAnnotationsFile = open(val_annotaion_path, 'r')
    valAnnotationsContents = valAnnotationsFile.read()
    valAnnotations = {}
    for line in valAnnotationsContents.splitlines():
        pieces = line.strip().split()
        valAnnotations[pieces[0]] = pieces[1]
    return valAnnotations

def download_and_unzip(zipurl, root_dir):
    with urlopen(zipurl) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(root_dir)


if __name__=='__main__':  

    """
    Tiny-ImageNet has 200 classes. Each image label has 500 training images(totally 100,000), 50 validation images(totally
    10,000), and 50 test images (totally 10,000). The test images are unlabeled.
    Since test images are not labeled we use validation images for known/unknown splits.
    """    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='.', help='Where the data is located')
    args = parser.parse_args()
 
    if not os.path.exists(f'{args.root_dir}/tiny-imagenet-200'):
        print('tiny-imagenet-200 is downloading ...')
        download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip', args.root_dir)
        print('done!')
        
    train_path = os.path.join(os.path.join(args.root_dir,
                                          'tiny-imagenet-200'), 'train')
    val_path = os.path.join(os.path.join(args.root_dir,
                                          'tiny-imagenet-200'), 'val')
    test_path = os.path.join(os.path.join(args.root_dir,
                                          'tiny-imagenet-200'), 'test') 
        
    print('categorize validation images ...')
    ### get  training annotations ###
    annotations = {}
    j = 0
    for sChild in os.listdir(train_path):
        sChildPath = os.path.join(os.path.join(train_path,sChild),'images')
        annotations[sChild]=j
        j+=1
    
    ### manage directory to save categorized validation images ###
    catval_path = os.path.join(os.path.join(args.root_dir,
                                            'tiny-imagenet-200'), 'categorized_val/') 
    if not os.path.isdir(catval_path):
        os.mkdir(catval_path)

    val_annotations_map = get_annotations_map(os.path.join(val_path, 
                                            'val_annotations.txt'))    
    for sChild in os.listdir(os.path.join(val_path, 'images')):
        if val_annotations_map[sChild] in annotations.keys():
            dst = catval_path + val_annotations_map[sChild] 
            if not os.path.isdir(dst):
                os.mkdir(dst)
                os.mkdir(dst+'/images')
            sChildPath = os.path.join(os.path.join(val_path, 'images'), sChild)
            src = os.path.join(val_path, 'images') + '/' + str(sChild)
            copy(src, dst+'/images')
            
    print('done!')        

    print('randomly sample 20 classes as known and the remaining classes as unknown ...')
    classes = np.arange(0,200)    
    random.shuffle(classes)    
    random_known_classes = classes[180:] 
    random_unknown_classes = classes[:180]  
    
    ### manage directory to save splitted datasets ###    
    known_train_path = os.path.join(os.path.join(args.root_dir,
                                                 'tiny-imagenet-200'), 'trainset/')
    eval_path = os.path.join(os.path.join(args.root_dir,
                                                 'tiny-imagenet-200'), 'validation/')
    ood_path = os.path.join(os.path.join(args.root_dir,
                                                  'tiny-imagenet-200'), 'ood/')

    if not os.path.isdir(known_train_path):
        os.makedirs(known_train_path)

    if not os.path.isdir(eval_path):    
        os.makedirs(eval_path) 

    if not os.path.isdir(ood_path):    
        os.mkdir(ood_path)
        
    ### copy from source to destination ###
    ### known train_set
    for key in annotations:
        if annotations[key] in random_known_classes:
            src = train_path + '/' + key
            dst = known_train_path + '/' + key
            copytree(src, dst)
            
    ### known eval_set                   
    for key in annotations:
        if annotations[key] in random_known_classes:
            src = catval_path + key
            dst = eval_path +'/' + key
            copytree(src, dst)
            
    ### unknown eval_set        
    for key in annotations:
        if annotations[key] in random_unknown_classes:
            src = catval_path + key
            dst = ood_path + '/' + key
            copytree(src, dst)    
            
    ### remove unnecessary folders
    if os.path.isdir(train_path):
        shutil.rmtree(train_path)
        
    if os.path.isdir(val_path):
        shutil.rmtree(val_path) 
    
    if os.path.isdir(catval_path):
        shutil.rmtree(catval_path) 
        
    os.rename(os.path.join(os.path.join(args.root_dir,'tiny-imagenet-200'), 'trainset/'), 
              os.path.join(os.path.join(args.root_dir,'tiny-imagenet-200'), 'train/')) 
