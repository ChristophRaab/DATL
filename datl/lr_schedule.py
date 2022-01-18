import torch
import numpy as np


def inv_lr_scheduler(optimizer,
                     iter_num,
                     gamma,
                     power,
                     lr=0.001,
                     weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = lr * (1 + gamma * iter_num)**(-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        i += 1
    #print(lr)
    return optimizer


def cdan_lda_coeff(
        iter_num,
        high=1.0,
        low=0.0,
        alpha=10.0,
        max_iter=10000.0):  # CDAM Lambda Adjustments progress based.
    return np.float(2.0 * (high - low) /
                    (1.0 + np.exp(-alpha * iter_num / max_iter)) -
                    (high - low) + low)
