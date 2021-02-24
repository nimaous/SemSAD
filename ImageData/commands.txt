Required Python Packages
    Python 3.6 
    pytorch >= 1.2 (pip3.6 install torch==1.2.0 torchvision==0.4.0  compatible with CUDA 10.0)
    matplotlib (pip3.6 install matplotlib)
    sklearn (pip3.6 install -U scikit-learn)    
    random 
    argparse
    numpy
    tqdm (pip3.6 install tqdm)
    PIL (pip3.6 install pillow)
    re

    
Recommend Packages
    tensorboard >= 2.0
    
    
For tiny_imagenet please at first run 
    - python tinyimagenet_preparation.py
    after running the above command the data samples can be access under tiny-imagenet-200/  in the main directory
    

Required Steps for Training

1- Training the encoder h   
 (* for training h by contrastive learning a large batch size needed, which requires about 4 GPU, e.g. GTX1080ti (40 Gig RAM)) 
 (* we observed moderate variations in AUROC score for different training runs of h. We trained h 3 times and report the evaluation with best h.
  for cifar10 and cifar100 datasets you find the checkpoints for the best trained h in the trained_model directory under 
         1- h_net_cifar10.pt    for cifar10
         2- h_net_cifar100.pt    for cifar100

   1.1 cifar10:  python train_encoder.py --dataset=cifar10 

   1.2 cifar100: python train_encoder.py --dataset=cifar100 
   
   1.3 tiny_imagenet: python train_encoder.py --dataset=tiny_imagenet --ds_dir=tiny-imagenet-200/train/ 
   
notice that:
  1- checkpoint is saved under ./checkpoint/{args.dataset}_{args.h_dim}_bs{args.bs}_hlr{args.h_lr}_temp{args.temprature}_h_nt.pt 
  2- hyperparameters can be changed and passed as arguments 
   
   
2- Training the discriminator s 
 (* which requires about 2 GPUs)   
 (* it's necessary to pass the path to the trained h as an argument)
 (* most of the variation in AUROC comes from variation in s for different training runs
 
  2.1 cifar10: python train_discriminator.py --dataset=cifar10  --h_net_path=<path to the trained h corresponding to the required dataset> 

  2.2 cifar100: python train_discriminator.py --dataset=cifar100  --h_net_path=<path to the trained h corresponding to the required dataset>
  
  2.3 tiny_imagenet:  python train_discriminator.py --dataset=tiny_imagenet --ds_dir=tiny-imagenet-200/train/ --h_net_path=<path to the trained h_net> 
  
  
3- Evaluating the performance of the discriminator s
    (* requires about 2 GPUs)
    

  3.1 cifar10 vs svhn:  python roc.py --dataset=cifar10 --ood_dataset=svhn  --h_net_path=<path to encoder h trained on cifar10> --s_net_path=<path to discriminator s trained on cifar10 >
  
  3.2 cifar100 vs svhn:  python roc.py --dataset=cifar100 --ood_dataset=svhn  --h_net_path=<path to encoder h trained on cifar100> --s_net_path=<path to discriminator s trained on cifar100 >
  
  3.3 cifar10 vs cifar100:  python roc.py --dataset=cifar10 --ood_dataset=cifar100  --h_net_path=<path to encoder h trained on cifar10> --s_net_path=<path to discriminator s trained on cifar10 >
  
  3.4 cifar100 vs cifar10:  python roc.py --dataset=cifar100 --ood_dataset=cifar10  --h_net_path=<path to encoder h trained on cifar100> --s_net_path=<path to discriminator s trained on cifar100 >  
  
  3.5 cifar100 vs tiny_iamgenet:  python roc.py --dataset=cifar100 --ood_dataset=tiny_imagenet --ood_dir=tiny-imagenet-200/test/  --h_net_path=<path to encoder h trained on cifar100> --s_net_path=<path to discriminator s trained on cifar100 > 
  
  3.6 cifar10 vs tiny_iamgenet:  python roc.py --dataset=cifar10 --ood_dataset=tiny_imagenet --ood_dir=tiny-imagenet-200/test/  --h_net_path=<path to encoder h trained on cifar10> --s_net_path=<path to discriminator s trained on cifar10> 
  
  3.7 tiny_iamgenet vs tiny_iamgenet:  python roc.py --dataset=tiny_iamgenet --ood_dataset=tiny_imagenet --ood_size=9000 --test_size=1000 --ood_dir=tiny-imagenet-200/ood/ --train_dir=tiny-imagenet-200/train/ --test_dir=tiny-imagenet-200/validation/ --h_net_path=<path to encoder h > --s_net_path=<path to discriminator s>   
