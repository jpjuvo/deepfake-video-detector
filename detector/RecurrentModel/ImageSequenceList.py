from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision import learner
import random
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from RecurrentModel.ImageSequence import ImageSequence
from RecurrentModel.RecurrentModelConfig import RecurrentModelConfig

class ImageSequenceList(ImageList):
    def __init__(self,  
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hard coded sequence length and sampling indices
        self.len_sequence = 5
        self.min_frame_index = 10
        self.max_frame_index = 16
        
    def __len__(self)->int: return len(self.items) or 1 
    
    def get(self, i):
        fn = self.items[i]
        
        # sample consecutive frames randomly from the specified range 
        start_index = random.randint(self.min_frame_index, self.max_frame_index-self.len_sequence)
        im_paths = [fn + '_0_{0}.png'.format(i) for i in range(start_index, start_index+self.len_sequence)]
        
        # check that the paths exist
        found_paths = []
        for pth in im_paths:
            if os.path.exists(pth):
                found_paths.append(pth)
        if len(found_paths) < self.len_sequence:
            short_length = len(found_paths)
            if (short_length>=self.len_sequence//2):
                # reverse and copy frames
                for i, ind in enumerate(range(short_length, self.len_sequence)):
                    found_paths.append(found_paths[short_length-1-i])
            else:
                # copy last frame
                for _ in range(short_length, self.len_sequence):
                    found_paths.append(found_paths[short_length-1])
        
        np_imgs = [cv2.cvtColor(cv2.imread(fn),cv2.COLOR_BGR2RGB) for fn in found_paths]
        return ImageSequence(np_imgs)
    
    def reconstruct(self, t):
        np_imgs = [tensor_img.cpu().detach().numpy() for tensor_img in t]
        
        imagenet_mean=0.45
        imagenet_std=0.225
        np_imgs = np.array([img*imagenet_std + imagenet_mean for img in np_imgs])

        return ImageSequence(np_imgs*255)
    
    def show_xys(self, xs, ys, figsize:Tuple[int,int]=(9,10), **kwargs):
        rows = int(math.sqrt(len(xs)))
        fig, axs = plt.subplots(rows,rows,figsize=figsize)
        for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
            xs[i].to_one().show(ax=ax, y=ys[i], **kwargs)
        plt.tight_layout()
        
    def show_xyzs(self, xs, ys, zs, figsize:Tuple[int,int]=None, **kwargs):
        """Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`. 
        `kwargs` are passed to the show method."""
        figsize = ifnone(figsize, (6,3*len(xs)))
        fig,axs = plt.subplots(len(xs), 2, figsize=figsize)
        fig.suptitle('Ground truth / Predictions', weight='bold', size=14)
        for i,(x,y,z) in enumerate(zip(xs,ys,zs)):
            x.show(ax=axs[i,0], y=y, **kwargs)
            x.show(ax=axs[i,1], y=z, **kwargs)

class ImageSequenceListOfTwo(ImageList):
    def __init__(self,  
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hard coded sequence length and sampling indices
        self.len_sequence = 2
        self.min_frame_index = 10
        self.max_frame_index = 16
        
    def __len__(self)->int: return len(self.items) or 1 
    
    def get(self, i):
        fn = self.items[i]
        
        # sample consecutive frames randomly from the specified range 
        start_index = random.randint(self.min_frame_index, self.max_frame_index-self.len_sequence)
        im_paths = [fn + '_0_{0}.png'.format(i) for i in range(start_index, start_index+self.len_sequence)]
        
        # check that the paths exist
        found_paths = []
        for pth in im_paths:
            if os.path.exists(pth):
                found_paths.append(pth)
        if len(found_paths) < self.len_sequence:
            short_length = len(found_paths)
            if (short_length>=self.len_sequence//2):
                # reverse and copy frames
                for i, ind in enumerate(range(short_length, self.len_sequence)):
                    found_paths.append(found_paths[short_length-1-i])
            else:
                # copy last frame
                for _ in range(short_length, self.len_sequence):
                    found_paths.append(found_paths[short_length-1])
        
        np_imgs = [cv2.cvtColor(cv2.imread(fn),cv2.COLOR_BGR2RGB) for fn in found_paths]
        return ImageSequence(np_imgs)
    
    def reconstruct(self, t):
        np_imgs = [tensor_img.cpu().detach().numpy() for tensor_img in t]
        
        imagenet_mean=0.45
        imagenet_std=0.225
        np_imgs = np.array([img*imagenet_std + imagenet_mean for img in np_imgs])

        return ImageSequence(np_imgs*255)
    
    def show_xys(self, xs, ys, figsize:Tuple[int,int]=(9,10), **kwargs):
        rows = int(math.sqrt(len(xs)))
        fig, axs = plt.subplots(rows,rows,figsize=figsize)
        for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
            xs[i].to_one().show(ax=ax, y=ys[i], **kwargs)
        plt.tight_layout()
        
    def show_xyzs(self, xs, ys, zs, figsize:Tuple[int,int]=None, **kwargs):
        """Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`. 
        `kwargs` are passed to the show method."""
        figsize = ifnone(figsize, (6,3*len(xs)))
        fig,axs = plt.subplots(len(xs), 2, figsize=figsize)
        fig.suptitle('Ground truth / Predictions', weight='bold', size=14)
        for i,(x,y,z) in enumerate(zip(xs,ys,zs)):
            x.show(ax=axs[i,0], y=y, **kwargs)
            x.show(ax=axs[i,1], y=z, **kwargs)

class ImageSequenceListFirstFrames(ImageList):
    def __init__(self,  
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hard coded sequence length and sampling indices
        self.len_sequence = 5
        self.min_frame_index = 0
        self.max_frame_index = 9
        
    def __len__(self)->int: return len(self.items) or 1 
    
    def get(self, i):
        fn = self.items[i]
        
        # sample consecutive frames randomly from the specified range 
        start_index = random.randint(self.min_frame_index, self.max_frame_index-self.len_sequence)
        im_paths = [fn + '_0_{0}.png'.format(i) for i in range(start_index, start_index+self.len_sequence)]
        
        # check that the paths exist
        found_paths = []
        for pth in im_paths:
            if os.path.exists(pth):
                found_paths.append(pth)
        if len(found_paths) < self.len_sequence:
            short_length = len(found_paths)
            if (short_length>=self.len_sequence//2):
                # reverse and copy frames
                for i, ind in enumerate(range(short_length, self.len_sequence)):
                    found_paths.append(found_paths[short_length-1-i])
            else:
                # copy last frame
                for _ in range(short_length, self.len_sequence):
                    found_paths.append(found_paths[short_length-1])
        
        np_imgs = [cv2.cvtColor(cv2.imread(fn),cv2.COLOR_BGR2RGB) for fn in found_paths]
        return ImageSequence(np_imgs)
    
    def reconstruct(self, t):
        np_imgs = [tensor_img.cpu().detach().numpy() for tensor_img in t]
        
        imagenet_mean=0.45
        imagenet_std=0.225
        np_imgs = np.array([img*imagenet_std + imagenet_mean for img in np_imgs])

        return ImageSequence(np_imgs*255)
    
    def show_xys(self, xs, ys, figsize:Tuple[int,int]=(9,10), **kwargs):
        rows = int(math.sqrt(len(xs)))
        fig, axs = plt.subplots(rows,rows,figsize=figsize)
        for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
            xs[i].to_one().show(ax=ax, y=ys[i], **kwargs)
        plt.tight_layout()
        
    def show_xyzs(self, xs, ys, zs, figsize:Tuple[int,int]=None, **kwargs):
        """Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`. 
        `kwargs` are passed to the show method."""
        figsize = ifnone(figsize, (6,3*len(xs)))
        fig,axs = plt.subplots(len(xs), 2, figsize=figsize)
        fig.suptitle('Ground truth / Predictions', weight='bold', size=14)
        for i,(x,y,z) in enumerate(zip(xs,ys,zs)):
            x.show(ax=axs[i,0], y=y, **kwargs)
            x.show(ax=axs[i,1], y=z, **kwargs)

