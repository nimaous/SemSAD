Required Python Packages
    Python 3.6 
    pytorch >= 1.7 (pip3.6 install torch==1.7.0 torchtext==0.8.0  compatible with CUDA 10.2)
    transformers  (pip3.6 install transformers)
    matplotlib (pip3.6 install matplotlib)
    sklearn (pip3.6 install -U scikit-learn)    
    random 
    argparse
    numpy
    tqdm (pip3.6 install tqdm)
    re

    
Recommend Packages
    tensorboard >= 2.0
    
    

Required Steps for Training:

(* for training using BERT model  requires about 4 GPU, e.g. GTX1080ti (40 Gig RAM))

1- Training the encoder h     
 (* we observed moderate variations in AUROC score for different training runs of h. We trained h 3 times and report the evaluation with best h.
  for AG-NEWS datasets you find the checkpoint for the best trained h in the trained_model directory under 
         1-AG_NEWS_h_net_.pt   


   -python train_encoder.py  

   
notice that:
  1- checkpoint is saved under ./checkpoint/AG_NEWS_hdim_{args.h_dim}_h_net.pt 
  2- hyperparameters can be changed and passed as arguments 
   
   
2- Training the discriminator s 
 (* which requires at least 4 GPUs)   
 (* most of the variation in AUROC comes from variation in s for different training runs
 
   -python train_discriminator.py 


  
  
3- Evaluating the performance of the discriminator s
    (* requires at least 4 GPUs)
    
    - python evaluate_discriminator.py --h_net_path=<path to the trained h_net>  --s_net_path=<path to the trained s_net>
  

  
