import os 
import torch
from torch import nn, optim
import random
import argparse
from tqdm import tqdm
import numpy as np
from dataset_wrapper import DataSetWrapper
from torchvision.models.resnet import resnet18
from utils import  nce, count_parameters
from change_resnet import modify_resnet_model


"""  if you get tensorflow related error 
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
"""

vis = True
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    vis = False 
    
wt = 0.001
similarity = nn.CosineSimilarity(dim=1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(epoch, args, loader,  h_net, h_net_opt):        
    loader = tqdm(loader)    
    h_net.train()              
    for itr, ((img_norm, img_t1, img_t2), label) in enumerate(loader): 
        h_net.zero_grad()        
        img_t1, img_t2  = img_t1.to(device), img_t2.to(device)            
        h1 = h_net(img_t1)
        h2 = h_net(img_t2)
        nce_loss, accuracy = nce(h1, h2, temprature = args.temprature)
        nce_loss.backward()
        h_net_opt.step()                                                   
        loader.set_description(
            (
                f' Epoch: {epoch + 1};  NCELoss: {nce_loss.item()}  ' #Enc Losses: {enc_losses.item():.3f};                 
            )
        )
        if vis == True:
            writer.add_scalar("NCE Loss", nce_loss.item(), global_step=epoch, walltime=wt) 
            writer.add_scalar("Accuracy", accuracy.item(), global_step=epoch, walltime=wt) 
    if vis == True:
        with torch.no_grad(): 
            if epoch % 100 == 0:
                label_lst = label.tolist()
                label_lst = [f'{i}' for i in label_lst]            
                writer.add_embedding(tag="h1", mat=h_net(img_norm), metadata=label_lst, label_img=img_norm , global_step=epoch)            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=2048)
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--ds_dir', type=str, default='.')    
    parser.add_argument('--epoch', type=int, default=1500)    
    parser.add_argument('--h_lr', type=float, default=3e-4)   
    parser.add_argument('--h_dim', type=int, default=128)    
    parser.add_argument('--temprature', type=float, default=0.5)   
    parser.add_argument('--weight_decay', type=float, default=1e-6)                                                   
    parser.add_argument('--nw', type=int, default=6)         
    args = parser.parse_args()
           
    ### initialisation of the encoder ### 
    torch.manual_seed(1)
    h_net = resnet18(pretrained=False, progress=False, num_classes=args.h_dim)
    h_net = modify_resnet_model(h_net, args , mode='encoder')    
    print("Number of h net Paramters: ", count_parameters(h_net))
    
    ### defining the optimiser #####
    h_net_opt = optim.Adam(h_net.parameters(), lr=args.h_lr, weight_decay=args.weight_decay)    
    h_net = nn.DataParallel(h_net)    
    h_net = h_net.to(device)
    
    ### get the dataloaders ######
    ds_warped = DataSetWrapper(args.dataset, args.ds_dir, args.bs, None , args.nw, mode='encoder')
    loader , _ = ds_warped.get_loaders()
            
    ### manage directory to save the checkpoint   ./checkpoint/experiment_name/h_net.pt     
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint') 
    if not os.path.isdir('runs'):
        os.mkdir('runs')         
    exp_num = f'h_{args.dataset}_{args.h_dim}_bs{args.bs}_hlr{args.h_lr}_temp{args.temprature}'        
    
    if vis == True:
        writer = SummaryWriter(f'runs/{exp_num}') 
    
    for epoch in range(args.epoch):
        train(epoch, args, loader, h_net, h_net_opt)
        torch.save(
             {'model': h_net.module.state_dict(), 'args': args},
            f'checkpoint/{exp_num}_h_net.pt',
        )

    if vis == True:
        writer.close()

 
    

    