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


def plot_pair(solution, model, points=np.linspace(0, 1, 200).reshape(-1, 1), n_params=0,
              xlabel=r'$t$', ylabel=r'$\hat{u} | u$', confidence=None, alpha=0.4,
              title='Solution against approximation', loc=1):
    """ Plot solution-approximation given by a pydens-model along with true solution.
    """
    approxs = model.solve(points)
    points = points if n_params == 0 else points[:, :points.shape[1] - n_params]
    true = solution(points).reshape(-1)
    plt.plot(points, true, 'b', linewidth=4, label='True solution', alpha=alpha + 0.15)
    plt.plot(points, approxs, 'r--', linewidth=5, label='Network approximation')
    plt.xlabel(xlabel, fontdict={'fontsize': 16})
    plt.ylabel(ylabel, fontdict={'fontsize': 16})
    plt.title(title, fontdict={'fontsize': 17})

    if confidence is not None:
        plt.fill_between(points.reshape(-1), true - confidence, true + confidence, alpha=alpha,
                         label='Confidence')
    plt.legend(loc=loc)
    plt.show()


def show_heatmap(model, fetches=None, grid=None, cmap='viridis',
                 title='Elliptic PDE in $\mathcal{R}^2$: approximate solution'):
    """
    Show heatmap of a model-prediction.
    """
    if grid is None:
        n_el = 100
        grid = cart_prod(np.linspace(0, 1, n_el), np.linspace(0, 1, n_el))

    approxs = model.solve(grid, fetches=fetches)
    plt.title(title, fontdict={'fontsize': 17})
    plt.imshow(approxs.reshape(n_el, n_el), cmap=cmap)
    plt.colorbar()
    plt.show()

def cart_prod(*arrs):
    """ Get array of cartesian tuples from arbitrary number of arrays.
    """
    grids = np.meshgrid(*arrs, indexing='ij')
    return np.stack(grids, axis=-1).reshape(-1, len(arrs))
