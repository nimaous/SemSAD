import os
import matplotlib.pyplot as plt
import torch
import argparse
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torchvision import models

from dataset_wrapper import DataSetWrapper
from get_score import Get_Score
from torchvision.models.resnet import resnet34, resnet18
from change_resnet import modify_resnet_model
device = 'cuda'


def load_models(args,):
    h_args = torch.load(args.h_net_path)['args']  
    h_net = resnet18(pretrained=False, progress=False, num_classes=h_args.h_dim)
    h_net = modify_resnet_model(h_net, args, mode='encoder')        
    h_net.load_state_dict(torch.load(args.h_net_path)['model'])
    h_net = nn.DataParallel(h_net)
    h_net = h_net.to(device)
    h_net.train()
         
    s_args = torch.load(args.s_net_path)['args'] 
    div = s_args.div
    s_net = resnet34(pretrained=False, progress=False, num_classes=s_args.s_dim) 
    s_net = modify_resnet_model(s_net, args, mode='discriminator')         
    s_net.load_state_dict(torch.load(args.s_net_path)['model'])
    s_net = nn.DataParallel(s_net)
    s_net = s_net.to(device)
    s_net.train()
    return h_net, s_net, div
        
def main(args):    
    ind_train_ds = DataSetWrapper(args.dataset, args.ind_dir, args.batch_size,
                                  args.batch_size, args.nw, mode='auroc', train=True, shuffle=False)
    ind_train_loader, _ = ind_train_ds.get_loaders() 
    
    ind_test_ds = DataSetWrapper(args.dataset, args.ind_dir, 
                                 args.batch_size, args.batch_size, args.nw, mode='auroc', train=False, shuffle=False)
    ind_test_loader, _ = ind_test_ds.get_loaders()
            
    ood_test_ds = DataSetWrapper(args.ood_dataset, args.ood_dir,
                                 args.batch_size, args.batch_size, args.nw, mode='auroc', train=False, shuffle=False)
    ood_test_loader, _ = ood_test_ds.get_loaders()
    
    loaders = [ind_train_loader, ind_test_loader, ood_test_loader]    
    h_net, s_net, div = load_models(args,)    
    with torch.no_grad():
        get_score = Get_Score([h_net, s_net], loaders, args, div, train_hist=True)
        score_ind_train, score_ind_test, score_ood = get_score.score_function()                                             
 

if __name__=='__main__':     
    parser = argparse.ArgumentParser(description='out-of-distribution-detection')
    parser.add_argument('--train_size', default=5e4, type=int)
    parser.add_argument('--test_size', default=1e4, type=int)
    parser.add_argument('--nw', type=int, default=6) 
    parser.add_argument('--batch_size', default=5000, type=int, help='number of images in each mini-batch')
    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar100 or cifar10')
    parser.add_argument('--ood_dataset', type=str, default='cifar100', help='cifar100 or cifar10 or svhn')
    parser.add_argument('--ind_dir', type=str, default='.') 
    parser.add_argument('--ood_dir', type=str, default='.')
    parser.add_argument('--h_net_path', type=str, )
    parser.add_argument('--s_net_path', type=str, )

    args = parser.parse_args()
    args.device = device  
    save_path = args.dataset + '_' + args.ood_dataset + '/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    args.save_path = save_path
    main(args)


