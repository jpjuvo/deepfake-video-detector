import torch
import torch.nn as nn
import torch.nn.functional as F

def remove_diag(x):
    bs = x.shape[0]
    return x[~torch.eye(bs).bool()].reshape(bs,bs-1)

class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.3):
        super().__init__()
        self.temp = temp

    def forward(self, input, labels):
        # input shape is bs,2,feat. Concatenate pairs to end
        out = torch.cat((input[:,0,:], input[:,1,:]), dim=0)
        bs,feat = out.shape
        
        labels = torch.arange(bs).cuda().roll(bs//2)
        
        csim = F.cosine_similarity(out, out.unsqueeze(dim=1), dim=-1)/self.temp
        csim = remove_diag(csim) # remove self similarity
        labels =  remove_diag(torch.eye(labels.shape[0], device=out.device)[labels]).nonzero()[:,-1]
        return F.cross_entropy(csim, labels)