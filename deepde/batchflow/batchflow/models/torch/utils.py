""" Auxiliary functions for Torch models """
import numpy as np
import torch


def get_num_channels(inputs, axis=1):
    """ Return a number of channels """
    return get_shape(inputs)[axis]

def get_num_dims(inputs):
    """ Return a number of semantic dimensions (i.e. excluding batch and channels axis)"""
    shape = get_shape(inputs)
    dim = len(shape)
    return max(1, dim - 2)

def get_shape(inputs, shape=None):
    """ Return inputs shape """
    if inputs is None:
        pass
    elif isinstance(inputs, np.ndarray):
        shape = inputs.shape
    elif isinstance(inputs, torch.Tensor):
        shape = tuple(inputs.shape)
    elif isinstance(inputs, (torch.Size, tuple, list)):
        shape = tuple(inputs)
    elif isinstance(inputs, torch.nn.Module):
        shape = get_output_shape(inputs, shape)
    else:
        raise TypeError('inputs can be array, tensor, tuple/list or layer', type(inputs))
    return shape

def get_output_shape(layer, shape=None):
    """ Return layer shape if it is defined """
    if hasattr(layer, 'output_shape'):
        shape = tuple(layer.output_shape)
    elif isinstance(layer, torch.nn.Sequential):
        shape = get_output_shape(layer[-1])
    return shape
