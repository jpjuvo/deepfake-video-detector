from fastai import *
from fastai.vision import *
import torch

from selfsupervised.ContrastiveImageTuple import ContrastiveImageTuple

class ContrastiveImageTupleList(ImageList):
    def __init__(self, items, **kwargs):
        super().__init__(items, **kwargs)

    def get(self, i):
        img1 = super().get(i)
        img2 = super().get(i)
        return ContrastiveImageTuple(img1, img2)

    def reconstruct(self, t:Tensor):
        imagenet_mean=torch.tensor(0.45)
        imagenet_std=torch.tensor(0.225)
        return ContrastiveImageTuple(Image(t[0]*imagenet_std + imagenet_mean),Image(t[1]*imagenet_std + imagenet_mean))

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