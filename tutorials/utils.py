"""Utility functions for tutorials. """

import matplotlib.pyplot as plt



def plot_loss(graph_lists, labels=None, ylabel='Loss', figsize=(8, 5), title=None):
    """ Plot losses.

    Parameters
    ----------
    graph_lists : sequence of arrays
        list of arrays to plot.

    labels : sequence of str
        labels for different graphs.

    ylabel : str
        y-axis label.

    figsize : tuple of int
        size of resulting figure.

    title : str
        title of resulting figure.
    """
    if not isinstance(graph_lists[0], (tuple, list)):
        graph_lists = [graph_lists]

    labels = labels or 'loss'
    labels = labels if isinstance(labels, (tuple, list)) else [labels]

    plt.figure(figsize=figsize)
    for arr, label in zip(graph_lists, labels):
        plt.plot(arr, label=label)
    plt.xlabel('Iterations', fontdict={'fontsize': 15})
    plt.ylabel(ylabel, fontdict={'fontsize': 15})
    plt.grid(True)
    if title:
        plt.title(title, fontdict={'fontsize': 15})
    plt.legend()
    plt.show()
