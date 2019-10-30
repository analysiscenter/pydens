""" PyDEns Init-file. """

#pylint: disable=no-name-in-module, import-error
from .syntax_tree import *
from .tokens import add_tokens
from .letters import *
from .model_tf import TFDeepGalerkin
from .wrapper import Solver
from .plot_utils import *
from .batchflow.sampler import *

<<<<<<< HEAD
__version__ = '0.1.2'
=======
__version__ = '0.1.1'
>>>>>>> e0b6bf998f612fa1142cef6e10622d3f86c591dc
