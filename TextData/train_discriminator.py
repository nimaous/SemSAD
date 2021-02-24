import wget
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "2,3,0,1"
import argparse
from tqdm import tqdm

from transformers import BertTokenizer,BertModel
import torch
import torchtext
from torch import nn, optim
from torch.utils.data import DataLoader
from text_dataset import TextDataset
from utils import  count_parameters
from encoder import  ConvEncoder
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

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
log_sigmoid = nn.LogSigmoid()


bert_dev =  [0,1,2,3]
s_dev= [1,2,3]
norm = torch.nn.PairwiseDistance(p=2) 

def avg_loss(args, a_pos, a_neg, GN):                   
    loss = torch.mean(-(log_sigmoid(a_pos) + GN*log_sigmoid(-a_neg)), dim=0) # average over samples
    min_loss, min_dim = loss.topk(1, largest=False) # select smallest dim  
    loss_avg = loss.mean() #average over dimensions
    return loss_avg, min_dim

def get_bert_output(x):
    with torch.no_grad():
            x_dict = tokenizer.batch_encode_plus(x, truncation=True, 
                                                  max_length= args.max_length, 
                                                  padding='max_length',
                                                  return_tensors="pt").to(f'cuda:{bert_dev[0]}')  
            out = bert(**x_dict)
            hs = out[2] #[13, bs, L, 768]
            hs = hs[-1] + hs[-2] + hs[-3] + hs[-4]
            tokens_sum = hs[:,1:].sum(dim=1).unsqueeze(1) #[bs, 1 , 768]
            cls = hs[:,0].unsqueeze(1) #[bs, 1, 768]            
            return torch.cat((tokens_sum, cls), dim=1)#[bs,2,768]   


def train(epoch, loader, s_net_opt, s_net):
    loader = tqdm(loader)
    s_net.train()
    bert.eval()

    for i , (x1 , x2 , x_rnd, _, _ , label) in enumerate(loader):
        if args.gamma_neg == -1 :
            GN = np.random.randint(args.min_gamma_neg, args.max_gamma_neg)
        else:
            GN = args.gamma_neg        
        s_net_opt.zero_grad()
        with torch.no_grad():
            x1_rep = get_bert_output(x1) #[bs,768, 300,]             
            x2_rep = get_bert_output(x2)            
            x_rnd_rep = get_bert_output(x_rnd)

        z_pos = s_net(torch.cat((x1_rep, x2_rep), 
                                dim=1).to(f'cuda:{s_dev[0]}'))
        z_neg = s_net(torch.cat((x1_rep, x_rnd_rep), 
                                dim=1).to(f'cuda:{s_dev[0]}'))            
        a_pos = z_pos - np.log(GN)
        a_neg = z_neg - np.log(GN)              
        loss, best_dim = avg_loss(args, a_pos, a_neg, GN)
        loss.backward()
        s_net_opt.step()
        loader.set_description(
            (
                f' Epoch: {epoch + 1};  Iteration:{i};   Loss: {loss.item()}  '         
            )
        )
        
    if vis == True:
        with torch.no_grad(): 
            writer.add_histogram('Positive Pairs', a_pos.cpu().numpy(), global_step=epoch, walltime=wt)
            writer.add_histogram('Negative Pairs', a_neg.cpu().numpy(), global_step=epoch, walltime=wt)           
            writer.add_scalar("CE Loss", loss.item(), global_step=epoch, walltime=wt)        
            writer.add_scalar("Min Dim", best_dim.item(), global_step=epoch, walltime=wt)             
        
                
              
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='AG_NEWS')
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--div', type=int, default=0)     
    parser.add_argument('--epoch', type=int, default=150)    
    parser.add_argument('--s_lr', type=float, default=5e-5)   
    parser.add_argument('--s_dim', type=int, default=512)    
    parser.add_argument('--weight_decay', type=float, default=1e-6) 
    parser.add_argument('--th_sus_prt', type=float, default=0.5) 
    parser.add_argument('--tf_idf_pr', type=float, default=0.0) 
    parser.add_argument('--rnd_th_sus_prt', type=float, default=0.5) 
    parser.add_argument('--rnd_tf_idf_pr', type=float, default=0.0)     
    parser.add_argument('--gamma_neg', type=int, default=-1)                 
    parser.add_argument('--min_gamma_neg', type=int, default=1)                 
    parser.add_argument('--max_gamma_neg', type=int, default=10)  
    parser.add_argument('--max_length', type=int, default=100)    
    parser.add_argument('--out_cls', type=int, default=0)   
    parser.add_argument('--sync_aug', type=bool, default=True)   

    
       
    args = parser.parse_args()   
    s_net = ConvEncoder(4, 768, args.s_dim)
    s_net = nn.DataParallel(s_net, device_ids=s_dev)
    s_net = s_net.to(f'cuda:{s_dev[0]}') 
    s_net_opt = optim.Adam(s_net.parameters(), lr=args.s_lr, weight_decay=args.weight_decay)  
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert =  BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True,).to(f'cuda:{bert_dev[0]}')
    bert = nn.DataParallel(bert, device_ids=bert_dev)    
    
    
    
    
    lambda1 = lambda epoch: 1 if epoch <70 else (0.5 if epoch <100 else 0.01 )
    scheduler = LambdaLR(s_net_opt, lr_lambda= lambda1)  
    
    tp = "tokens_train" if os.path.isfile("tokens_train") else None
    dsp = "tf-idf-dict_train" if os.path.isfile("tf-idf-dict_train") else None
    
    ds = TextDataset(dataset_name=args.dataset_name, 
                     sufix_name = "train",
                     th_sus_prt = args.th_sus_prt,
                     tf_idf_pr = args.tf_idf_pr,   
                     rnd_th_sus_prt = args.rnd_th_sus_prt,
                     rnd_tf_idf_pr = args.rnd_tf_idf_pr,                       
                     tokens_path = tp,   
                     data_stats_path = dsp,
                     sync_aug = args.sync_aug,
                     out_cls=[0])
    
    loader = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=24)
    
    exp_name = f'{args.dataset_name}_S_{args.s_dim}' 
    
    
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint') 
    if not os.path.isdir('runs'):
        os.mkdir('runs')      
    
    if vis == True:
        writer = SummaryWriter(f'runs/{exp_name}') 
        
    for epoch in range(args.epoch):
        train(epoch, loader, s_net_opt, s_net)
        scheduler.step()
        torch.save(
             {'model': s_net.module.state_dict(), 'args': args},
            f'checkpoint/{exp_name}_s_net.pt',
        )       
    writer.close()    
    