""" BatchFlow enables a fast processing of large dataset using flexible pipelines """

import sys

if sys.version_info < (3, 5):
    raise ImportError("BatchFlow module requires Python 3.5 or higher")

from .base import Baseset
from .batch import Batch
from .batch_image import ImagesBatch
from .config import Config
from .dataset import Dataset
from .pipeline import Pipeline
from .named_expr import NamedExpression, B, C, F, L, V, D, R, W, P
from .dsindex import DatasetIndex, FilesIndex
from .decorators import action, inbatch_parallel, parallel, any_action_failed, mjit
from .exceptions import SkipBatchException
from .sampler import Sampler, ConstantSampler, NumpySampler, HistoSampler, ScipySampler


__version__ = '0.3.0'
