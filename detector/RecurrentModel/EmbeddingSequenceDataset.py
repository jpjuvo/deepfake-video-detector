from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision import learner
import random
import cv2
import numpy as np

class EmbeddingSequenceDataset(Dataset):
    def __init__(self, df, name, path, label, suffix, sequence_len=10):
        print('Checking files...')
        file_path_list = []
        ys = []
        for fname, y in zip(df[name].values,df[label].values):
            full_path = os.path.join(path, fname + suffix)
            if os.path.isfile(full_path):
                file_path_list.append(full_path)
                ys.append(1 if y=='FAKE' else 0)
        self.file_paths = file_path_list
        self.ys = ys
        print("Loaded {0} files".format(len(file_path_list)))
        self.sequence_len = sequence_len
        self.c = 2 # binary label
    
    def __len__(self):
        return len(self.file_paths)
    
    def get_emb_szs(self):
        return self.__getitem__(0)[0].shape
    
    def __getitem__(self, i):
        fname = self.file_paths[i]
        x = np.load(fname)
        
        # some sequence item can be missing, fill out the missing ones (reflect)
        n_seq = x.shape[0]
        if n_seq < self.sequence_len and n_seq >= self.sequence_len//2:
            new_x = np.zeros((self.sequence_len,x.shape[1]))
            new_x[:n_seq,:] = x
            for i, ind in enumerate(range(n_seq, self.sequence_len)):
                new_x[ind,:] = x[n_seq-1-i]
            x = new_x.copy()
        # if reflect fill is not possible, copy the last item to fill the sequence
        elif n_seq < self.sequence_len:
            new_x = np.zeros((self.sequence_len,x.shape[1]))
            new_x[:n_seq,:] = x
            for i, ind in enumerate(range(n_seq, self.sequence_len)):
                new_x[ind,:] = x[n_seq-1]
            x = new_x.copy()
        elif n_seq > self.sequence_len:
            x = x[:self.sequence_len,:]
        y = self.ys[i]
        
        # return float x and long y
        return torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.long)