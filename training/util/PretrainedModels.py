import fastai
from fastai.vision import *
import pretrainedmodels
import torch.nn as nn
import geffnet
import kornia

# Helpers for viewing the model layers

def arch_summary(arch):
    model = arch(False)
    tot = 0
    for i, l in enumerate(model.children()):
        n_layers = len(flatten_model(l))
        tot += n_layers
        print(f'({i}) {l.__class__.__name__:<12}: {n_layers:<4}layers (total: {tot})')

def get_groups(model, layer_groups):
    group_indices = [len(g) for g in layer_groups]
    curr_i = 0
    group = []
    for layer in model:
        group_indices[curr_i] -= len(flatten_model(layer))
        group.append(layer.__class__.__name__)
        if group_indices[curr_i] == 0:
            curr_i += 1
            print(f'Group {curr_i}:', group)   
            group = []
        
class FakeData:
    def __init__(self):
        self.c = 2
        self.path = ''    
        self.device = None
        self.loss_func = CrossEntropyFlat(axis=1)

        
def convert_MP_to_blurMP(model, layer_type_old):
    conversion_count = 0
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_MP_to_blurMP(module, layer_type_old)
        if type(module) == layer_type_old:
            layer_old = module
            layer_new = kornia.contrib.MaxBlurPool2d(3, True)
            model._modules[name] = layer_new
    return model

################################################
#### Models wrapped to Fastai's format #########
################################################
        
# we wrap cadene model to pytorch models format
def se_resnext50_32x4d(pretrained=True):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained=pretrained)
    all_layers = list(model.children())
    return nn.Sequential(*all_layers)

# we wrap cadene model to pytorch models format
def se_resnext50_32x4d_blurmaxpool(pretrained=True):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained=pretrained)
    model = convert_MP_to_blurMP(model, nn.MaxPool2d)
    #model = convert_MP_to_blurMP(model, nn.AdaptiveAvgPool2d)
    all_layers = list(model.children())
    return nn.Sequential(*all_layers)

def se_resnext101_32x4d(pretrained=True):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.__dict__['se_resnext101_32x4d'](pretrained=pretrained)
    all_layers = list(model.children())
    return nn.Sequential(*all_layers)

def xception(pretrained=True):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.__dict__['xception'](pretrained=pretrained)
    all_layers = list(model.children())
    return nn.Sequential(*all_layers)

def xception_blurmaxpool(pretrained=True):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.__dict__['xception'](pretrained=pretrained)
    model = convert_MP_to_blurMP(model, nn.MaxPool2d)
    all_layers = list(model.children())
    return nn.Sequential(*all_layers).cuda()

# we wrap model to fastai models format
def effnetb2(pretrained=True):
    m = geffnet.efficientnet_b2(pretrained=pretrained, drop_rate=0.25, drop_connect_rate=0.2, as_sequential=True)
    return m

def effnetb0(pretrained=True):
    m = geffnet.efficientnet_b0(pretrained=pretrained, drop_rate=0.25, drop_connect_rate=0.2, as_sequential=True)

def effnetb1(pretrained=True):
    m = geffnet.efficientnet_b1(pretrained=pretrained, drop_rate=0.25, drop_connect_rate=0.2, as_sequential=True)
    return m

def effnetb3(pretrained=True):
    m = geffnet.efficientnet_b3(pretrained=pretrained, drop_rate=0.25, drop_connect_rate=0.2, as_sequential=True)
    return m

def effnetb4(pretrained=True):
    m = geffnet.efficientnet_b4(pretrained=pretrained, drop_rate=0.25, drop_connect_rate=0.2, as_sequential=True)
    return m

def effnetb6(pretrained=True):
    m = geffnet.efficientnet_b6(pretrained=pretrained, drop_rate=0.25, drop_connect_rate=0.2, as_sequential=True)
    return m