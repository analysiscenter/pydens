""" Functions for making mathematical tokens and adding them to desired namespace. """
import inspect

import numpy as np
import tensorflow as tf
import torch

try:
    import networkx as nx
except ImportError:
    pass

from .letters import TFLetters, NPLetters, TorchLetters



MATH_TOKENS = ['sin', 'cos', 'tan',
               'asin', 'acos', 'atan',
               'sinh', 'cosh', 'tanh',
               'asinh', 'acosh', 'atanh',
               'exp', 'log', 'pow',
               'sqrt', 'sign',
               ]

CUSTOM_TOKENS = ['D', 'P', 'V', 'C', 'R']

LABELS_MAPPING = {
    '__sub__': '-', '__rsub__': '-',
    '__mul__': '*', '__rmul__': '*',
    '__div__': '/', '__rdiv__': '/',
    '__truediv__': '/', '__rtruediv__': '/',
    '__add__': '+', '__radd__': '+',
    '__pow__': '^', '__rpow__': '^'
}


def add_binary_magic(cls):
    """ Add binary-magic operators to `SyntaxTreeNode`-class. Allows to create and parse syntax trees
    using binary operations like '+', '-', '*', '/'.

    Parameters
    ----------
    cls : class
        The class to be processed by the decorator.
    operators : sequence
        Sequence of magic-method names to be added to `cls`.

    Returns
    -------
    modified class.
    """
    operators = list(LABELS_MAPPING.keys())

    for magic_name in operators:
        def magic(self, other, magic_name=magic_name):
            return cls(LABELS_MAPPING.get(magic_name), self, other)

        setattr(cls, magic_name, magic)
    return cls


@add_binary_magic
class SyntaxTreeNode():
    """ Node of parse tree. Stores operation representing the node along with its arguments.

    Parameters
    ----------
    name : str
        name of the node. Used for creating a readable string-repr of a tree.
    *args:
        args[0] : method representing the node of parse tree.
        args[1:] : arguments of the method.
    """
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self._args = args
        self._kwargs = kwargs

    def __len__(self):
        return len(self._args)

    def __repr__(self):
        return tuple((self.name, *self._args, self._kwargs)).__repr__()

def get_num_parameters(form):
    """ Get number of unique parameters (created via `P` letter) in the passed form."""
    n_args = len(inspect.signature(form).parameters)
    tree = form(*[SyntaxTreeNode('_' + str(i)) for i in range(n_args)])
    return len(get_unique_parameters(tree))

def get_unique_parameters(tree):
    """ Get unique names of parameters-variables (those containing 'P' in its name) from a parse-tree.
    """
    # pylint: disable=protected-access
    if isinstance(tree, (int, float, str, tf.Tensor, tf.Variable)):
        return []
    if tree.name == 'P':
        return [tree._args[0]]
    if len(tree) == 0:
        return []

    result = []
    for arg in tree._args:
        result += get_unique_parameters(arg)

    return list(set(result))


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

    if not var_dict:
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



# Drawing functions. Require graphviz
def make_unique_node(graph, name):
    """ Add as much postfix-'_' to `name` as necessary to make unique name for new node in `graph`.

    Parameters
    ----------
    graph : nx.Graph
        graph, for which the node is created.
    name : str
        name of new node.

    Returns
    -------
    Resulting name. Composed from `name` and possibly several '_'-characters.
    """
    if name not in graph:
        return name
    ctr = 1
    while True:
        name_ = name + '_' * ctr
        if name_ not in graph:
            return name_
        ctr += 1

def _build_graph(tree, graph, parent_name, labels):
    """ Recursive graph-builder. Util-function.
    """
    #pylint: disable=protected-access
    if isinstance(tree, (float, int)):
        return
    if len(tree) == 0:
        return

    for child in tree._args:
        if isinstance(child, (float, int)):
            child_name = make_unique_node(graph, str(np.round(child, 2)))
            labels.update({child_name: str(np.round(child, 2))})
        else:
            child_name = make_unique_node(graph, child.name)
            labels.update({child_name: LABELS_MAPPING.get(child.name, child.name)})

        graph.add_edge(parent_name, child_name)
        _build_graph(child, graph, child_name, labels)

def build_graph(tree):
    """ Build graph from a syntax tree.
    """
    # boundary case: trees with no children
    graph = nx.DiGraph()
    if isinstance(tree, (float, int)):
        graph.add_node(str(np.round(tree, 2)))
        return graph

    parent_name = LABELS_MAPPING.get(tree.name, tree.name)
    graph.add_node(parent_name)
    if len(tree) == 0:
        return graph

    # process generic trees
    labels = {parent_name: parent_name}
    _build_graph(tree, graph, parent_name, labels)

    return graph, labels
