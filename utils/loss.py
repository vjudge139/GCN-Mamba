import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import utils.lovasz_loss as L

def loss(logits, labels):
    if len(labels.shape) == 4:
        labels = torch.argmax(labels, 1)
    if logits.shape == labels.shape:
        labels = torch.argmax(labels, dim=1)
    elif len(labels.shape) == 3:
        labels = labels
    else:
        assert False, "pred.shape not match label.shape"
        
    ce_loss_1 = torch.nn.CrossEntropyLoss()(logits, labels)
    lovasz_loss = L.lovasz_softmax(F.softmax(logits, dim=1), labels, ignore=255)
    main_loss = ce_loss_1 + lovasz_loss
    return main_loss

def ce_love(input, target, weight=None):
    return loss(input, target)
