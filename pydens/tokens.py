""" Functions for making mathematical tokens and adding them to desired namespace. """
import inspect

import numpy as np
import tensorflow as tf

try:
    import torch
except ImportError:
    pass

from .syntax_tree import SyntaxTreeNode
from .letters import TFLetters, TorchLetters, NPLetters

MATH_TOKENS = ['sin', 'cos', 'tan',
               'asin', 'acos', 'atan',
               'sinh', 'cosh', 'tanh',
               'asinh', 'acosh', 'atanh',
               'exp', 'sigmoid', 'log', 'pow',
               'sqrt', 'sign',
               ]

CUSTOM_TOKENS = ['D', 'P', 'V', 'C', 'R', 'grad', 'laplace', 'Δ', 'div']


def make_token(module='tf', name=None, namespaces=None):
    """ Make a mathematical tokens.

    Parameters
    ----------
    module : str
        Can be 'np' (stands for `numpy`) or 'tf'(stands for `tensorflow`). Either choice binds tokens to
        correspondingly named operations from a module. For instance, token 'sin' for module 'np' stands for
        operation `np.sin`.
    name : str
        name of module function used for binding tokens.

    Returns
    -------
    callable
        Function that can be applied to a parse-tree, adding another node in there.
    """
    # parse namespaces-arg
    if module in ['tensorflow', 'tf']:
        namespaces = namespaces or [tf.math, tf, tf.nn]
        module_ = TFLetters()
    elif module in ['numpy', 'np']:
        namespaces = namespaces or [np, np.math]
        module_ = NPLetters()
    elif module == 'torch':
        namespaces = namespaces or [torch, torch.nn]
        module_ = TorchLetters()

    # None of the passed modules are supported
    if namespaces is None:
        raise ValueError('Module ' + module + ' is not supported: you should directly pass namespaces-arg!')

    # make method
    method_ = getattr(module_, name) if name in CUSTOM_TOKENS else fetch_method(name, namespaces)

    # method_ = letters.get(name) or fetch_method(name, namespaces)
    method = (lambda *args, **kwargs: SyntaxTreeNode(name, *args, **kwargs)
              if isinstance(args[0], SyntaxTreeNode) else method_(*args, **kwargs))
    return method

def fetch_method(name, modules):
    """ Get function from list of modules. """
    for module in modules:
        if hasattr(module, name):
            return getattr(module, name)
    raise ValueError('Cannot find method ' + name + ' in ' + ', '.join([module.__name__ for module in modules]))


def add_tokens(var_dict=None, postfix='__', module='tf', names=None, namespaces=None):
    """ Add tokens to passed namespace.

    Parameters
    ----------
    var_dict : dict
        Namespace to add names to. Default values is the namespace from which the function is called.
    postfix : str
        If the passed namespace already contains item with the same name, then
        postfix is appended to the name to avoid naming collision.
    module : str
        Can be 'np' (stands for `numpy`) or 'tf'(stands for `tensorflow`). Either choice binds tokens to
        correspondingly named operations from a module. For instance, token 'sin' for module 'np' stands for
        operation `np.sin`.
    names : str
        Names of function to be tokenized from the given module.

    Notes
    -----
    This function is also called when anything from this module is imported inside
    executable code (e.g. code where __name__ = __main__).
    """
    names = names or (MATH_TOKENS + CUSTOM_TOKENS)

    if var_dict is None:
        frame = inspect.currentframe()
        try:
            var_dict = frame.f_back.f_locals
        finally:
            del frame

    for name in names:
        token = make_token(module=module, name=name, namespaces=namespaces)
        if name not in var_dict:
            name_ = name
        else:
            name_ = name + postfix
            msg = 'Name `{}` already present in current namespace. Added as {}'.format(name, name+postfix)
            print(msg)
        var_dict[name_] = token
