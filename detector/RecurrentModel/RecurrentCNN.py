from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision import learner
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from RecurrentModel.ImageSequence import ImageSequence
from RecurrentModel.ImageSequenceList import ImageSequenceList 

class VideoSequenceModel(nn.Module):
    def __init__(self, pretrained_classifier, 
                 hidden_dim = 1024,#2048,
                 emb_sz=2,
                 ps=0.6, 
                 bidirectional=True):
        super().__init__()
        self.hidden_dim, self.emb_sz, self.ps = hidden_dim, emb_sz, ps
        self.bidirectional=bidirectional
        
        hidden_dim_size = self.hidden_dim * (1+self.bidirectional)
        
        self.body = pretrained_classifier[:-1]
        self.maxpool = nn.MaxPool2d((10,10))
        self.lstm1 = nn.LSTM(num_features_model(self.body), 
                             self.hidden_dim,
                             batch_first=True,
                             num_layers=1,
                             #dropout=self.ps,
                             bidirectional=self.bidirectional) # True returns two times the outputs
        #self.batchnorm1 = nn.BatchNorm1d(hidden_dim_size)
        self.dropout1 = nn.Dropout(self.ps)
        
        self.linear1 = nn.Linear(hidden_dim_size, self.emb_sz)# hidden_dim_size//2)
        #self.batchnorm2 = nn.BatchNorm1d(hidden_dim_size//2)
        #self.act2 = nn.ReLU()
        #self.dropout2 = nn.Dropout(self.ps)
        
        #self.linear2 = nn.Linear(hidden_dim_size//2, self.emb_sz)
    
    def forward(self, x):
        self.lstm1.flatten_parameters() # GPU memory allocation
        # x.shape (bs, sequence_length, ch, w, h)
        bs,sequence_length,ch,w,h = x.shape
        # body requires (bs, ch, w, h) * sequence_length
        out = x.view(bs*sequence_length,ch,w,h)
        out = self.body(out)
        out = self.maxpool(out)
        #c_outs = torch.cat([self.maxpool(self.body(batch)).unsqueeze(dim=0) for batch in x.permute(1,0,2,3,4)],
        #                    dim=0)
        out = out.view(bs, sequence_length,-1)
        out,(h_n, h_c) = self.lstm1(out, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """
        out = out[:,-1,:] # take the last time step
        #out = self.batchnorm1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.linear1(out)
        #out = self.batchnorm2(out)
        #out = self.act2(out)
        #out = self.dropout2(out)
        #out = self.linear2(out)
        return out


class GRUVideoSequenceModel(nn.Module):
    def __init__(self, pretrained_classifier, 
                 hidden_dim = 128,
                 emb_sz=2,
                 ps=0.2,
                 bidirectional=True):
        super().__init__()
        self.hidden_dim, self.emb_sz, self.ps = hidden_dim, emb_sz, ps
        self.bidirectional=bidirectional
        
        hidden_dim_size = self.hidden_dim * (1+self.bidirectional)
        
        self.body = pretrained_classifier[:-1]
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.recurrent = nn.GRU(num_features_model(self.body), 
                                self.hidden_dim,
                                batch_first=True,
                                num_layers=1,
                                #dropout=self.ps/2,
                                bidirectional=self.bidirectional) # True returns two times the outputs
        #self.batchnorm1 = nn.BatchNorm1d(hidden_dim_size)
        self.dropout1 = nn.Dropout(self.ps/2)
        
        self.linear1 = nn.Linear(hidden_dim_size, hidden_dim_size//2)
        #self.batchnorm2 = nn.BatchNorm1d(hidden_dim_size//2)
        self.dropout2 = nn.Dropout(self.ps)
        self.linear2 = nn.Linear(hidden_dim_size//2, self.emb_sz)
    
    def forward(self, x):
        self.recurrent.flatten_parameters() # GPU memory allocation
        # x.shape (bs, sequence_length, ch, w, h)
        bs,sequence_length,ch,w,h = x.shape
        # body requires (bs, ch, w, h) * sequence_length
        out = x.view(bs*sequence_length,ch,w,h)
        out = self.body(out)
        out = self.pool(out)
        out = out.view(bs, sequence_length,-1)
        out, _ = self.recurrent(out, None)
        out = out[:,-1,:] # take the last time step
        out = F.elu(out)
        out = self.dropout1(out)
        out = self.linear1(out)
        out = F.elu(out)
        out = self.dropout2(out)
        out = self.linear2(out)
        return out

class ConvVideoSequenceModel(nn.Module):
    def __init__(self, pretrained_classifier, 
                 hidden_dim = 128,
                 emb_sz=2,
                 ps=0.2):
        super().__init__()
        self.hidden_dim, self.emb_sz, self.ps = hidden_dim, emb_sz, ps
        
        hidden_dim_size = self.hidden_dim
        
        self.body = pretrained_classifier[:-1]
        self.pool = nn.AvgPool2d((10,10))
        self.conv1d = nn.Conv1d(in_channels=num_features_model(self.body),
                                   out_channels=hidden_dim,
                                   kernel_size=5)
        #self.batchnorm1 = nn.BatchNorm1d(hidden_dim_size)
        self.dropout1 = nn.Dropout(self.ps/2)
        
        self.linear1 = nn.Linear(hidden_dim_size, hidden_dim_size//2)
        #self.batchnorm2 = nn.BatchNorm1d(hidden_dim_size//2)
        self.dropout2 = nn.Dropout(self.ps)
        self.linear2 = nn.Linear(hidden_dim_size//2, self.emb_sz)
    
    def forward(self, x):
        # x.shape (bs, sequence_length, ch, w, h)
        bs,sequence_length,ch,w,h = x.shape
        # body requires (bs, ch, w, h) * sequence_length
        out = x.view(bs*sequence_length,ch,w,h)
        out = self.body(out)
        out = self.pool(out)
        out = out.view(bs, sequence_length,-1)
        out = out.permute(0,2,1)
        out = self.conv1d(out)
        out = out.view(bs, -1)
        out = F.elu(out)
        out = self.dropout1(out)
        out = self.linear1(out)
        out = F.elu(out)
        out = self.dropout2(out)
        out = self.linear2(out)
        return out

class MixedVideoSequenceModel(nn.Module):
    def __init__(self, pretrained_classifier, 
                 hidden_dim = 64,
                 emb_sz=2,
                 ps=0.2,
                 bidirectional=True):
        super().__init__()
        self.hidden_dim, self.emb_sz, self.ps = hidden_dim, emb_sz, ps
        self.bidirectional=bidirectional
        
        hidden_dim_size = self.hidden_dim * (1+self.bidirectional)
        
        self.body = pretrained_classifier[:-1] 
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.recurrent = nn.GRU(num_features_model(self.body), 
                                self.hidden_dim,
                                batch_first=True,
                                num_layers=1,
                                #dropout=self.ps/2,
                                bidirectional=self.bidirectional) # True returns two times the outputs
        self.conv1d = nn.Conv1d(in_channels=num_features_model(self.body),
                                   out_channels=hidden_dim,
                                   kernel_size=5)
        #self.batchnorm1 = nn.BatchNorm1d(hidden_dim_size)
        self.dropout1 = nn.Dropout(self.ps/2)
        
        self.linear1 = nn.Linear(hidden_dim_size + hidden_dim, hidden_dim_size//2)
        #self.batchnorm2 = nn.BatchNorm1d(hidden_dim_size//2)
        self.dropout2 = nn.Dropout(self.ps)
        self.linear2 = nn.Linear(hidden_dim_size//2, self.emb_sz)
    
    def forward(self, x):
        self.recurrent.flatten_parameters() # GPU memory allocation
        # x.shape (bs, sequence_length, ch, w, h)
        bs,sequence_length,ch,w,h = x.shape
        # body requires (bs, ch, w, h) * sequence_length
        out = x.view(bs*sequence_length,ch,w,h)
        out = self.body(out)
        out = self.pool(out)
        out = out.view(bs, sequence_length,-1)
        
        gru_out, _ = self.recurrent(out, None)
        gru_out = gru_out[:,-1,:] # take the last time step

        conv_out = out.permute(0,2,1)
        conv_out = self.conv1d(conv_out)
        conv_out = conv_out.view(bs, -1)

        out = torch.cat([gru_out,conv_out],dim=1)
        out = F.elu(out)
        out = self.dropout1(out)
        out = self.linear1(out)
        out = F.elu(out)
        out = self.dropout2(out)
        out = self.linear2(out)
        return out