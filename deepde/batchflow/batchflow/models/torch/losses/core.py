""" Core loss functions """
import torch
import torch.nn as nn


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """ Custom loss which casts target dtype if needed """
    def forward(self, input, target):
        target = target.to(dtype=torch.long)
        return super().forward(input, target)
