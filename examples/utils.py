""" Util functions for examples.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_loss(loss, color='powderblue'):
    """ Plot loss of a trained pydens-model.
    """
    plt.plot(loss, c=color)
    plt.xlabel('Iteration number', fontdict={'fontsize': 15})
    plt.ylabel('Loss',fontdict={'fontsize': 15})
    plt.title('Model loss', fontdict={'fontsize': 19})
    plt.show()


def plot_pair(solution, model, points=np.linspace(0, 1, 200).reshape(-1, 1),
              xlabel=r'$t$', ylabel=r'$\hat{u}$', confidence=None, alpha=0.4):
    """ Plot solution-approximation given by a pydens-model along with true solution.
    """
    approxs = model.solve(points)
    true = solution(points).reshape(-1)
    plt.plot(points, true, 'b', linewidth=4, label='True solution', alpha=alpha + 0.15)
    plt.plot(points, approxs, 'r--', linewidth=5, label='Network approximation')
    plt.xlabel(xlabel, fontdict={'fontsize': 16})
    plt.ylabel(ylabel, fontdict={'fontsize': 16})
    plt.title('Solution against approximation', fontdict={'fontsize': 17})

    if confidence is not None:
        plt.fill_between(points.reshape(-1), true - confidence, true + confidence, alpha=alpha,
                         label='Confidence')
    plt.legend()
    plt.show()
