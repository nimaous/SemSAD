import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3"
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
h_dev = [0,1,2,3]
s_dev = [1,2,3]

def load_models(args,):
    h_args = torch.load(args.h_net_path)['args']  
    h_net = resnet18(pretrained=False, progress=False, num_classes=h_args.h_dim)
    h_net = modify_resnet_model(h_net, args, mode='encoder')        
    h_net.load_state_dict(torch.load(args.h_net_path)['model'])
    h_net = nn.DataParallel(h_net, device_ids=h_dev)
    h_net = h_net.to(f'cuda:{h_dev[0]}') 
    h_net.eval()
         
    s_args = torch.load(args.s_net_path)['args'] 
    div = s_args.div
    s_net = resnet34(pretrained=False, progress=False, num_classes=s_args.s_dim) 
    s_net = modify_resnet_model(s_net, args, mode='discriminator')         
    s_net.load_state_dict(torch.load(args.s_net_path)['model'])
    s_net = nn.DataParallel(s_net, device_ids=s_dev)
    s_net = s_net.to(f'cuda:{s_dev[0]}') 
    s_net.eval()
    return h_net, s_net, div
        
def main(args):    
    ind_train_ds = DataSetWrapper(args.dataset, args.train_dir, args.batch_size,
                                  args.batch_size, args.nw, mode='auroc', 
                                  train=True, shuffle=False)
    ind_train_loader, _ = ind_train_ds.get_loaders() 
    
    ind_test_ds = DataSetWrapper(args.dataset, args.test_dir, 
                                 args.batch_size, args.batch_size, args.nw, mode='auroc',
                                 train=False, shuffle=False)
    ind_test_loader, _ = ind_test_ds.get_loaders()
            
    ood_test_ds = DataSetWrapper(args.ood_dataset, args.ood_dir,
                                 args.batch_size, args.batch_size, args.nw, mode='auroc', 
                                 train=False, shuffle=False, ti_for_ci=True if args.dataset != 'tiny_imagenet' else False)
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
    parser.add_argument('--ood_size', default=1e4, type=int)
    parser.add_argument('--nw', type=int, default=6) 
    parser.add_argument('--batch_size', default=5000, type=int, help='number of images in each mini-batch')
    parser.add_argument('--dataset', type=str, default='cifar10', help='tiny_imagenet cifar100 or cifar10')
    parser.add_argument('--ood_dataset', type=str, default='tiny_imagenet', help='tiny_imagenet cifar100 or cifar10 or svhn')
    parser.add_argument('--train_dir', type=str, default='.') 
    parser.add_argument('--test_dir', type=str, default='.')     
    parser.add_argument('--ood_dir', type=str, default='tiny-imagenet-200/test/')
    parser.add_argument('--h_net_path', type=str, default='checkpoint/h_net_cifar10.pt' )
    parser.add_argument('--s_net_path', type=str, default='checkpoint/5th_s_cifar10_T8_sdim64_bs128_slr5e-05_resnet18ELU_div_32_grad_True_scheduler_4neighbours_gammaNeg1-10/s_net_500.pt' )

    args = parser.parse_args()
    args.s_dev = s_dev
    args.h_dev = h_dev  
    save_path = args.dataset + '_' + args.ood_dataset + '/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    args.save_path = save_path
    main(args)


