""" PyDEns Init-file. """

#pylint: disable=no-name-in-module, import-error
from .model_torch import Solver, D, V, TorchModel, ConvBlockModel, SpatialTemporalFourierNetwork
from .model_torch import FourierNetwork, MultiscaleFourierNetwork
from batchflow.sampler import *

__version__ = '1.0.2'
