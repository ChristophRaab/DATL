import numpy as np
import torch
from datl.normalization import orthogonalization
from datl.metrics import ltangent_distance
from torch import nn


class SiameseTangentLayer(nn.Module):

    def __init__(self, num_classes, num_protos, feat_dim=None, dim=None):
        super().__init__()
        self.nall = num_protos * num_classes
        self.num_protos, self.num_classes = num_protos, num_classes
        self.feat_dim, self.subdim = feat_dim, dim

    def forward(self, x, y, omegas):
        return ltangent_distance(x, y, omegas)

    def init(self, siaprotos, data=None, labels=None, protoparams=True):
        if data is not None:
            subspaces = torch.cat(
                [self.init_local_subspace(d, self.num_protos) for d in data])
        else:
            subspaces = torch.randn([self.nall, self.feat_dim, self.subdim])
        self.subspaces = nn.Parameter(subspaces, requires_grad=True)
        if protoparams:
            self.protos = nn.Parameter(siaprotos, requires_grad=True)
        else:
            self.protos = siaprotos
        self.plabels = labels

    def init_local_subspace(self, data, num_subspaces):
        data = data - torch.mean(data, dim=0)
        try:
            _, _, v = torch.linalg.svd(data)
        except:
            _, _, v = torch.linalg.svd(data)
        v = v.T[:, :self.subdim]
        return v.unsqueeze(0).repeat_interleave(num_subspaces, 0)

    def orthogonalize_subspace(self):
        if self.subspaces is not None:
            with torch.no_grad():
                self.subspaces.copy_(orthogonalization(self.subspaces))
