import os
import torch
import torch.nn as nn
from torch.utils import data
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from data_reader import RawDataset
from encoder import Encoder
from Transformer import CNN
from get_score import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='audio dataset')
    parser.add_argument('--data_root', default='dataset/train-Librispeech.h5')
    parser.add_argument('--train_list', default='list/male_train_split.txt')
    parser.add_argument('--test_list', default='list/male_test_split.txt')
    parser.add_argument('--ood_list', default='list/female_test_split.txt')
    parser.add_argument('--h_net_path', type=str, default='checkpoint/h_tau_5_hdim256.pt')
    parser.add_argument('--s_net_path', type=str, default='checkpoint/s_tau_5_hdim256_sdim64_gammaNeg1-10.pt')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--audio_window', type=int, default=20480)
    parser.add_argument('--sdim', type=int, default=64)
    parser.add_argument('--hdim', type=int, default=256)
    parser.add_argument('--len_seg', type=int, default=16)
    parser.add_argument('--input_channel', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=4)
    args = parser.parse_args()
    args.device = device
    
    ### get the dataloaders ###
    training_set = RawDataset(args.data_root, args.train_list, args.audio_window, train=False)
    test_set = RawDataset(args.data_root, args.test_list, args.audio_window, train=False)
    ood_set = RawDataset(args.data_root, args.ood_list, args.audio_window, train=False)
    train_loader = data.DataLoader(training_set, batch_size=args.batch_size, shuffle=False,  drop_last=False)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,  drop_last=False)
    ood_loader = data.DataLoader(ood_set, batch_size=args.batch_size, shuffle=False,  drop_last=False)
    
    ### load the encoder checkpoint ###
    assert args.h_net_path is not None
    h_net_ckpt = torch.load(args.h_net_path)
    h_net = Encoder(args.audio_window, args.hdim) 
    h_net.load_state_dict(h_net_ckpt['model'])
    h_net = nn.DataParallel(h_net)
    h_net = h_net.to(device)
    h_net.eval()  
    
    ### load the discriminator checkpoint ###
    assert args.s_net_path is not None
    s_net_ckpt = torch.load(args.s_net_path)
    s_net = CNN(args.input_channel, 256, args.len_seg, args.sdim, args.num_layers)
    s_net.load_state_dict(s_net_ckpt['model'])
    s_net = nn.DataParallel(s_net)
    s_net = s_net.to(args.device)
    s_net.eval()    
        
    with torch.no_grad():
        z_train, zmean_train = matrix_preparation_train(args, len(training_set), train_loader, h_net)
        z_test, label_test = matrix_preparation(args, len(test_set), test_loader, h_net)
        print('# male speakers in test_set:', len(torch.unique(label_test)))
        z_ood, label_ood = matrix_preparation(args, len(ood_set), ood_loader, h_net)
        print('#female speakers in ood_set:', len(torch.unique(label_ood)))

        best_indx_test = best_indices_(z_test, zmean_train)
        best_indx_ood = best_indices_(z_ood, zmean_train)

        best_match_test = best_match_(z_test, z_train[best_indx_test]) 
        best_match_ood = best_match_(z_ood, z_train[best_indx_ood]) 
        
        s_test = s_net(torch.cat((best_match_test.unsqueeze(1), z_test.unsqueeze(1)), dim=1)).sum(-1)
        s_ood = s_net(torch.cat((best_match_ood.unsqueeze(1), z_ood.unsqueeze(1)), dim=1)).sum(-1)
        
        file = open("auroc.txt","a+")
        l1 = torch.zeros(s_test.size(0))
        l2 = torch.ones(s_ood.size(0))
        label = torch.cat((l1, l2),dim=0).view(-1,1).cpu()
        scores = torch.cat((s_test, s_ood), dim=0).cpu()
        FPR, TPR, _ = roc_curve(label, scores, pos_label = 0)
        print('AUC:', auc(FPR, TPR))
        file.write("AUC :{} \r\n".format(auc(FPR, TPR)))        
        file.close() 
