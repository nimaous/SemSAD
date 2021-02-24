import torch.nn as nn
from torchvision.models.resnet import ResNet


def replace_relu(module, name):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == nn.ReLU:
            new_bn = nn.ELU(inplace=True)
            setattr(module, attr_str, new_bn)    
    for name, immediate_child_module in module.named_children():
        replace_relu(immediate_child_module, name) 



def replace_bn(module, name):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == nn.BatchNorm2d:
            new_bn = nn.BatchNorm2d(target_attr.num_features,
                                    track_running_stats=False)
            setattr(module, attr_str, new_bn)
    for name, immediate_child_module in module.named_children():
        replace_bn(immediate_child_module, name)         
        
def modify_resnet_model(model, args , mode='encoder'):
    """
     Modify resnet to fit for dataset with small image size
     and to accept 2*channel_size input for discriminator
    """
    assert isinstance(model, ResNet), "model must be a ResNet instance"          
    
    if args.dataset in ['cifar10', 'cifar100']:
        if mode == 'encoder':
            conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            
        if mode == 'discriminator':
            conv1 = nn.Conv2d(2*3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            replace_relu(model, 'model') 
            replace_bn(model, 'model')    

        nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
        model.conv1 = conv1
        model.maxpool = nn.Identity()
    if args.dataset in ['tiny_imagenet']:
        if mode == 'encoder':
            conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=1, bias=False)
            
        if mode == 'discriminator':
            conv1 = nn.Conv2d(2*3, 64, kernel_size=5, stride=2, padding=1, bias=False)                
            replace_relu(model, 'model') 
            replace_bn(model, 'model')
        nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
        model.conv1 = conv1 
    #
    return model
