""" Contains pyramid layer """
import numpy as np
import torch
import torch.nn as nn

from .conv_block import ConvBlock
from .upsample import Upsample
from ..utils import get_shape


class PyramidPooling(nn.Module):
    """ Pyramid Pooling module
    Zhao H. et al. "`Pyramid Scene Parsing Network <https://arxiv.org/abs/1612.01105>`_"

    Parameters
    ----------
    inputs : torch.Tensor, torch.nn.Module, numpy.ndarray or tuple
        shape or an example of input tensor to infer shape
    layout : str
        sequence of operations in convolution layer
    filters : int
        the number of filters in pyramid branches
    kernel_size : int
        kernel size
    pool_op : str
        a pooling operation ('mean' or 'max')
    pyramid : tuple of int
        the number of feature regions in each dimension, default is (0, 1, 2, 3, 6).
        `0` is used to include `inputs` into the output tensor.

    Returns
    -------
    torch.nn.Module
    """
    def __init__(self, inputs, layout='cna', filters=None, kernel_size=1, pool_op='mean',
                 pyramid=(0, 1, 2, 3, 6), **kwargs):
        super().__init__()

        shape = get_shape(inputs)
        self.axis = -1 if kwargs.get('data_format') == 'channels_last' else 1
        filters = filters if filters else shape[self.axis] // len(pyramid)

        if None in shape[1:]:
            # if some dimension is undefined
            raise ValueError("Pyramid pooling can only be applied to a tensor with a fully defined shape.")

        item_shape = np.array(shape[2:] if self.axis == 1 else shape[1:-1])

        modules = nn.ModuleList()
        for level in pyramid:
            if level == 0:
                module = None
            else:
                pool_size = tuple(np.ceil(item_shape / level).astype(np.int32).tolist())
                pool_strides = tuple(np.floor((item_shape - 1) / level + 1).astype(np.int32).tolist())

                pool = ConvBlock(inputs, 'p', pool_op=pool_op, pool_size=pool_size,
                                 pool_strides=pool_strides, **kwargs)
                conv = ConvBlock(pool, layout, filters=filters, kernel_size=kernel_size, **kwargs)
                upsamp = Upsample(inputs=conv, factor=None, layout='b', shape=tuple(item_shape.tolist()), **kwargs)
                module = nn.Sequential(pool, conv, upsamp)
            modules.append(module)
        self.blocks = modules
        self.output_shape = shape


    def forward(self, x):
        levels = [block(x) if block else x for block in self.blocks]
        return torch.cat(levels, dim=self.axis)
