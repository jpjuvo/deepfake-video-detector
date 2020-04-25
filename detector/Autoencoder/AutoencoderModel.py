import torch.nn as nn
import geffnet

class UpSample(nn.Module):
    def __init__(self,feat_in,feat_out,out_shape=None,scale=2):
        super().__init__()
        self.conv = nn.Conv2d(feat_in,feat_out,kernel_size=(3,3),stride=1,padding=1)
        self.out_shape,self.scale = out_shape,scale
        
    def forward(self,x):
        return self.conv(
            nn.functional.interpolate(
                x,size=self.out_shape,scale_factor=self.scale,mode='bilinear',align_corners=True))

def get_upSamp(feat_in,feat_out, out_shape=None, scale=2, act='relu'):
    
    upSamp = UpSample(feat_in,feat_out,out_shape=out_shape,scale=scale).cuda()
    
    layer = nn.Sequential(upSamp)
    
    if act == 'relu':
        act_f = nn.ReLU(inplace=True).cuda()
        bn = nn.BatchNorm2d(feat_out).cuda()
        layer.add_module('ReLU',act_f)
        layer.add_module('BN',bn)
    elif act == 'sig':
        act_f = nn.Sigmoid()
        layer.add_module('Sigmoid',act_f)
    return layer

def add_layer(m,feat_in,feat_out,name,out_shape=None,scale=2,act='relu'):
    upSamp = get_upSamp(feat_in,feat_out,out_shape=out_shape,scale=scale,act=act)
    m.add_module(name,upSamp)

def effnetb0(pretrained=True):
    m = geffnet.efficientnet_b0(pretrained=pretrained, drop_rate=0.25, drop_connect_rate=0.2)
    return m

def SimpleAEModel():
    encoder = effnetb0(True).cuda()

    model = nn.Sequential(encoder.conv_stem,
                       encoder.bn1,
                       encoder.act1,
                       encoder.blocks)

    code_sz = 64
    conv = nn.Conv2d(320, code_sz, kernel_size=(2,2)).cuda()
    model.add_module('CodeIn',conv)
    add_layer(model,code_sz,256,'CodeOut', out_shape=(10,10),scale=None)
    add_layer(model,256,128,'Upsample0')
    add_layer(model,128,64,'Upsample1')
    add_layer(model,64,32,'Upsample2')
    add_layer(model,32,3,'Upsample3',act='sig')
    return model