from enum import unique
from torch import nn
import torch
from torch import nn
from torch.autograd import Function
from torchvision import models

class SiameseTangentLayer(nn.Module):
    def __init__(self,num_classes,num_protos,feat_dim=None,dim=None):
        super().__init__()
        self.nall = num_protos *num_classes
        self.num_protos,self.num_classes =num_protos,num_classes
        self.feat_dim, self.subdim = feat_dim, dim

    def forward(self, x, protos):
            
        x = x.unsqueeze(1).expand(x.size(0), protos.size(0), x.size(-1))
        protos = protos.unsqueeze(0).expand(x.size(0), protos.size(0), x.size(-1))
        projectors =torch.eye(self.subspaces.shape[-2],device=x.device) - torch.bmm(self.subspaces,self.subspaces.permute([0,2,1]))
        diff = (x - protos)
        diff = diff.permute([1, 0, 2]) 
        diff = torch.bmm(diff, projectors) 
        diff = torch.norm(diff,2,dim=-1).T
        # diff = diff.reshape(diff.size(0),self.num_protos,self.num_classes) # wta prepare
        # diff,idx = torch.min(diff,dim=1) # wta 
        return diff 

    def init(self,siaprotos,data=None):
        if data is not None:
            subspaces = torch.cat([self.init_local_subspace(d[0],self.num_protos) for d in data])
        else:
            subspaces  = torch.randn([self.nall,self.feat_dim,self.subdim]) 
        self.subspaces = nn.Parameter(subspaces,requires_grad=True)
        self.protos = siaprotos


    def init_local_subspace(self,data,num_subspaces):
        data = data -  torch.mean(data,dim=0)      
        _,_,v = torch.svd(data)
        v = v[:,:self.subdim]
        return v.unsqueeze(0).repeat_interleave(num_subspaces,0)

    def orthogonalize_subspace(self):
        if self.subspaces is not None:
            with torch.no_grad():
                self.subspaces.copy_(orthogonalization(self.subspaces))

def orthogonalization(tensors):
    # orthogonalization via polar decomposition
        u, _, v = torch.svd(tensors,compute_uv=True)
        u_shape = tuple(list(u.shape))
        v_shape = tuple(list(v.shape))

        # reshape to (num x N x M)
        u = torch.reshape(u, (-1, u_shape[-2], u_shape[-1]))
        v = torch.reshape(v, (-1, v_shape[-2], v_shape[-1]))

        out = u @  v.permute([0, 2, 1])

        out = torch.reshape(out, u_shape[:-1] + (v_shape[-2],))

        return out


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def create_resnet50_features():
        model_resnet = models.resnet50(pretrained=True)
        conv1 = model_resnet.conv1
        bn1 = model_resnet.bn1
        relu = model_resnet.relu
        maxpool = model_resnet.maxpool
        layer1 = model_resnet.layer1
        layer2 = model_resnet.layer2
        layer3 = model_resnet.layer3
        layer4 = model_resnet.layer4
        avgpool = model_resnet.avgpool
        feature_layers = nn.Sequential(conv1, bn1, relu, maxpool, \
                                layer1, layer2, layer3, layer4, avgpool)
        return feature_layers