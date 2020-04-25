from fastai import *
from fastai.vision import *
from fastai.callbacks import *
import random
import cv2
import numpy as np

class ImageSequence(ItemBase):
    def __init__(self, np_imgs):
        self.fastai_imgs = [vision.Image(px=pil2tensor(np_img/255., np.float32)) for np_img in np_imgs]
        # we still keep track of the initial object (usuall in an obj attribute) 
        # to be able to show nice representations later on.
        self.obj = np_imgs
        # The basis is to code the data attribute that is what will be given to the model
        self.data = self.convertImagesToData()
    
    def apply_tfms(self, tfms,*args, **kwargs):
        # keep random state to apply the same augmentations for all images
        random_state = random.getstate()
        for i in range(len(self.fastai_imgs)):
            random.setstate(random_state)
            self.fastai_imgs[i] = self.fastai_imgs[i].apply_tfms(tfms, *args, **kwargs)
        self.data = self.convertImagesToData()
        return self
    
    def __repr__(self):
        return f'{self.__class__.__name__}'
    
    def to_one(self):
        return Image(torch.cat([img.data for img in self.fastai_imgs],dim=0)
                     .transpose(1,0)
                     .transpose(1,2))
    
    def convertImagesToData(self):
        # returns torch.tensor of shape [sequence_length, c, h, w]
        imagenet_mean = torch.tensor(0.45)
        imagenet_std = torch.tensor(0.225)
        return torch.cat([((img.data - imagenet_mean)/imagenet_std).unsqueeze(dim=0) for img in self.fastai_imgs],dim=0)
