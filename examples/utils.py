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


def show_heatmap(model, mode='imshow', fetches=None, grid=None, cmap='viridis',
                 title='Elliptic PDE in $\mathcal{R}^2$: approximate solution',
                 xlim=(0, 1), ylim=(0, 1)):
    """ Show heatmap of a model-prediction.
    """
    if mode == 'imshow':
        if grid is None:
            num_points = 80
            grid = cart_prod(np.linspace(*xlim, num_points), np.linspace(*ylim, num_points))

        approxs = model.solve(grid, fetches=fetches)
        plt.imshow(approxs.reshape(num_points, num_points), cmap=cmap)
    elif mode == 'contourf':
        if grid is None:
            num_points = 80
            xs = np.linspace(*xlim, num_points)
            ys = np.linspace(*ylim, num_points)
            xs_, ys_ = np.meshgrid(xs, ys)
            grid = cart_prod(xs, ys)

        zs_ = model.solve(grid).reshape(len(xs), len(ys))
        plt.contourf(xs_, ys_, zs_, cmap=cmap)

    plt.title(title, fontdict={'fontsize': 17})
    plt.colorbar()
    plt.show()


def cart_prod(*arrs):
    """ Get array of cartesian tuples from arbitrary number of arrays.
    """
    grids = np.meshgrid(*arrs, indexing='ij')
    return np.stack(grids, axis=-1).reshape(-1, len(arrs))


def plot_sections(model, mode='2d', timestamps=(0, 0.2, 0.4, 0.6, 0.7, 0.9), grid_size=(2, 3), points=None,
                  fetches=None, ylim=(0, 0.3), zlim=None, title=r'Heat PDE in $\mathcal{R}$: $\hat{u}$'):
    """ Plot sections of an approximation to an evolution equation.
    """
    if mode == '2d':
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

    elif mode == '3d':
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(grid_size[0]*7, grid_size[1]*3))
        x = points if points is not None else np.linspace(0, 1, 30)
        y = points if points is not None else np.linspace(0, 1, 30)
        X, Y = np.meshgrid(x, y)
        points = cart_prod(x, y)

        for i, t_ in enumerate(timestamps):
            points_ = np.concatenate([points, t_ * np.ones((points.shape[0], 1))], axis=1)
            ax = fig.add_subplot(*grid_size, i + 1, projection='3d')
            Z = model.solve(points_).reshape(len(x), len(y))
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                            cmap='viridis', edgecolor='none')
            ax.set_title('$t=%.2f$' % t_, size=17);
            ax.set_xlabel(r'$x$', fontdict={'fontsize': 14})
            ax.set_ylabel(r'$y$', fontdict={'fontsize': 14})
            ax.set_zlim(*zlim)

        fig.suptitle(title, size=20)
        plt.savefig('test.png', dpi=400)
        fig.show()
