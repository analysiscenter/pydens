""" PyTorch implementation of model for solving partial differential equations. Inspired by
Sirignano J., Spiliopoulos K. "`DGM: A deep learning algorithm for solving partial differential equations
<http://arxiv.org/abs/1708.07469>`_"
"""

import torch
import torch.nn as nn

#pylint: disable=no-name-in-module, import-error
from .batchflow.models.torch import TorchModel
from .batchflow.models.torch.layers import ConvBlock
from .syntax import get_num_parameters



class TorchDeep(TorchModel):
    """ Torch model of solving PDE's. """
    _ = torch, nn, ConvBlock, get_num_parameters
