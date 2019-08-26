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


def plot_pair(solution, model, points=None, plot_coord=None,
              xlabel=r'$t$', ylabel=r'$\hat{u} | u$', confidence=None, alpha=0.4,
              title='Solution against approximation', loc=1, grid=True):
    """ Plot solution-approximation given by a pydens-model along with true solution.
    """
    points = points if points is not None else np.linspace(0, 1, 200).reshape(-1, 1)
    approxs = model.solve(points)
    points = points if plot_coord is None else points[:, plot_coord]
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
    plt.grid(grid)
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


def plot_sections(model, timestamps=(0, 0.2, 0.4, 0.6, 0.7, 0.9), grid_size=(2, 3), points=None,
                  fetches=None, ylim=(0, 0.3), title=r'Heat PDE in $\mathcal{R}$: $\hat{u}$'):
    """ Plot 1d-sections of an approximation to an evolution equation.
    """
    points = points if points is not None else np.linspace(0, 1, 100).reshape(-1, 1)
    n_sections = len(timestamps)
    fig, axes = plt.subplots(*grid_size, figsize=(5 * grid_size[1], 5))
    for i, t_ in enumerate(timestamps):
        points_ = np.concatenate([points.reshape(-1, 1), t_ * np.ones((points.shape[0], 1))], axis=1)
        wx, wy = i // grid_size[1], i % grid_size[1]
        axes[wx, wy].plot(points.reshape(-1), model.solve(points_, fetches=fetches))
        axes[wx, wy].set_ylim(*ylim)
        axes[wx, wy].set_title('$t=%.2f$' % t_, size=22)

    fig.tight_layout()
    fig.suptitle(title, size=27)
    fig.subplots_adjust(top=0.82)
    fig.show()
