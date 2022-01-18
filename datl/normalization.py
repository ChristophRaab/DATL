from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch


def orthogonalization(tensors):
    # orthogonalization via polar decomposition
    u, _, v = torch.svd(tensors, compute_uv=True)
    u_shape = tuple(list(u.shape))
    v_shape = tuple(list(v.shape))

    # reshape to (num x N x M)
    u = torch.reshape(u, (-1, u_shape[-2], u_shape[-1]))
    v = torch.reshape(v, (-1, v_shape[-2], v_shape[-1]))

    out = u @ v.permute([0, 2, 1])

    out = torch.reshape(out, u_shape[:-1] + (v_shape[-2], ))

    return out