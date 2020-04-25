from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision import learner
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveModule(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.body = pretrained_model[:-2] # remove pool and linear layer
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                                  Flatten(), 
                                  nn.Linear(num_features_model(self.body), 256), 
                                  nn.ReLU(), 
                                  nn.Linear(256, 128))
    def forward(self,x):
        bs,replicate_length,ch,w,h = x.shape
        # body requires (bs*replicate_length, ch, w, h)
        out = x.view(bs*replicate_length,ch,w,h)
        out = self.body(out)
        out = self.head(out)
        # back to pairs
        out = out.view(bs,replicate_length,-1)
        return out