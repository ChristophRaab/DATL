import torch 
import torch.nn.functional as F

def entropy(p):
    p = F.softmax(p,dim=1)
    return -torch.mean(torch.sum(p * torch.log(p+1e-5), 1))


    
def mpce(mat, query_labels, reference_labels,t=1, invert_distance=True):
    # in parts from https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/losses/nca_loss.py
        if not invert_distance:
            mat = -mat
        same_labels = (query_labels.unsqueeze(1) == reference_labels.unsqueeze(0))

        exp = torch.nn.functional.softmax(mat / t, dim=1)
        exp = torch.sum(exp * same_labels, dim=1)
        non_zero = exp != 0
        loss = torch.mean(-torch.log(exp[non_zero]))
        return loss
    

