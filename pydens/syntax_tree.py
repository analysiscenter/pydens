""" Functions for building, working with and plotting syntax trees of functions. """

import inspect

import numpy as np
import tensorflow as tf

try:
    import networkx as nx
except ImportError:
    pass



LABELS_MAPPING = {
    '__sub__': '-', '__rsub__': '-',
    '__mul__': '*', '__rmul__': '*',
    '__div__': '/', '__rdiv__': '/',
    '__truediv__': '/', '__rtruediv__': '/',
    '__add__': '+', '__radd__': '+',
    '__pow__': '^', '__rpow__': '^',
    '__matmul__': '@', '__rmatmul__': '@'
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
    if isinstance(tree, (int, float, str, tf.Tensor, tf.Variable, list, np.ndarray)):
        return []
    if tree.name == 'P':
        return [tree._args[0]]
    if len(tree) == 0:
        return []

    result = []
    for arg in tree._args:
        result += get_unique_parameters(arg)

    return list(set(result))


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
