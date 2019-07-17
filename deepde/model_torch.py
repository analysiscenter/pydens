""" PyTorch implementation of model for solving partial differential equations. Inspired by
Sirignano J., Spiliopoulos K. "`DGM: A deep learning algorithm for solving partial differential equations
<http://arxiv.org/abs/1708.07469>`_"
"""
from inspect import signature

import tensorflow as tf
from tqdm import tqdm_notebook, tqdm

from .batchflow.models.torch import TorchModel
from .batchflow.models.torch.layers import ConvBlock
from .syntax import get_num_parameters



class TorchDeep(TorchModel):
	pass
