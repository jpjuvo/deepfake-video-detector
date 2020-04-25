from fastai import *
from fastai.vision import *
import torch

class ContrastiveImageTuple(ItemBase):
    def __init__(self, img1, img2):
        self.img1,self.img2 = img1,img2
        self.obj = (img1,img2)
        self.data = self.convertImagesToData()

    def apply_tfms(self, tfms,*args, **kwargs):
        self.img1 = self.img1.apply_tfms(tfms, *args, **kwargs)
        self.img2 = self.img2.apply_tfms(tfms, *args, **kwargs)
        self.data = self.convertImagesToData()
        return self
    
    def __repr__(self):
        return f'{self.__class__.__name__}'
    
    def to_one(self):
        return Image(torch.cat([img.data for img in [self.img1, self.img2]],dim=2))

    def convertImagesToData(self):
        # returns torch.tensor of shape [sequence_length, c, h, w]
        imagenet_mean = torch.tensor(0.45)
        imagenet_std = torch.tensor(0.225)
        return torch.cat([((img.data - imagenet_mean)/imagenet_std).unsqueeze(dim=0) for img in [self.img1,self.img2]],dim=0)