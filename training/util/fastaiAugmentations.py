import albumentations as albu
import PIL
import numpy as np
import fastai
from fastai.vision import *

"""
Fastai wrappers for Albumentation pixel level augmentations.

Fastai applies augmentations on tensors so to include pixel level augmentation,
we have to transform images back to numpy, apply transform, and then back to tensor.
"""

def JPEGAugment(quality_lower=60, quality_upper=100, always_apply=False, p=0.5):
    return alb_tfm2fastai(albu.JpegCompression(quality_lower=quality_lower,
                                               quality_upper=quality_upper,
                                               always_apply=always_apply,
                                               p=p))

def HueSaturationValueAugment(hue_shift_limit=20,sat_shift_limit=30,val_shift_limit=20,always_apply=False,p=0.5):
    return alb_tfm2fastai(albu.HueSaturationValue(hue_shift_limit=hue_shift_limit,
                                                  sat_shift_limit=sat_shift_limit,
                                                  val_shift_limit=val_shift_limit,
                                                  always_apply=always_apply,
                                                  p=p))

def BlurAugment(blur_limit=7,always_apply=False,p=0.5):
    return alb_tfm2fastai(albu.Blur(blur_limit=blur_limit,
                                    always_apply=always_apply,
                                    p=p))

def DownscaleAugment(scale_min=0.5,scale_max=0.9,interpolation=0,always_apply=False,p=0.65):
    return alb_tfm2fastai(albu.Downscale(always_apply=always_apply, 
                                         p=p, 
                                         scale_min=scale_min, 
                                         scale_max=scale_max, 
                                         interpolation=interpolation))

def tensor2np(x):
    np_image = x.cpu().permute(1, 2, 0).numpy()
    np_image = (np_image * 255).astype(np.uint8)
    
    return np_image

def alb_tfm2fastai(alb_tfm):

    def _alb_transformer(x):
        # tensor to numpy
        np_image = tensor2np(x)

        # apply albumentations
        transformed = alb_tfm(image=np_image)['image']

        # back to tensor
        tensor_image = pil2tensor(transformed, np.float32)
        tensor_image.div_(255)

        return tensor_image

    transformer = fastai.vision.transform.TfmPixel(_alb_transformer)
    
    return transformer()