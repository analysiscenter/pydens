""" Util functions for examples.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_loss(loss, color='powderblue'):
    """ Plot loss.
    """
    plt.plot(loss, c=color)
    plt.xlabel('Iteration number', fontdict={'fontsize': 15})
    plt.ylabel('Loss', fontdict={'fontsize': 15})
    plt.title('Model loss', fontdict={'fontsize': 19})
    plt.show()


def plot_pair_1d(model, solution=None, fetches='solution', points=None, plot_coord=None, xlabel=r'$t$',
                 ylabel=r'$\hat{u} | u$', confidence=None, alpha=0.4, title='Solution against approximation',
                 loc=1, grid=True, show=True, save_to=None):
    r""" Visualize solution-approximation to a 1d problem (e.g., ode in $\mathcal{R}$)
    along with true solution.
    """
    # calculate and plot approximate and true solution
    points = points if points is not None else np.linspace(0, 1, 200).reshape(-1, 1)
    approxs = model.solve(points, fetches=fetches)
    points = points if plot_coord is None else points[:, plot_coord]
    if solution is not None:
        true = solution(points).reshape(-1)
        plt.plot(points, true, 'b', linewidth=4, label='True solution', alpha=alpha + 0.15)
        if confidence is not None:
            plt.fill_between(points.reshape(-1), true - confidence, true + confidence, alpha=alpha,
                             label='Confidence')

    plt.plot(points, approxs, 'r--', linewidth=5, label='Network approximation')

    # additional visual elements
    plt.xlabel(xlabel, fontdict={'fontsize': 16})
    plt.ylabel(ylabel, fontdict={'fontsize': 16})
    plt.title(title, fontdict={'fontsize': 17})
    plt.legend(loc=loc)
    plt.grid(grid)

    if save_to is not None:
        plt.savefig(save_to, dpi=300)

    if show:
        plt.show()


def plot_2d(model, mode='imshow', fetches=None, grid=None, x_size=None, y_size=None,
            cmap='viridis', title=r'Elliptic PDE in $\mathcal{R}^2$: approximate solution',
            xlim=(0, 1), ylim=(0, 1), num_points=None, save_to=None):
    r""" Visualize solution-approximation to a 2d problem (e.g., poisson problem in $\mathcal{R}^2$-square).
    """
    # make sure that the grid is set up
    if grid is None:
        if num_points is None:
            if mode == 'imshow':
                num_points = 80
            elif mode == 'contourf':
                num_points = 80
            elif mode == '3d_view':
                num_points = 20
            else:
                raise ValueError("Plot-mode {} is not supported.".format(mode))

        xs = np.linspace(*xlim, num_points)
        ys = np.linspace(*ylim, num_points)
        grid = cart_prod(xs, ys)
        xs_grid, ys_grid = np.meshgrid(xs, ys)
        x_size, y_size = num_points, num_points

    # calculate approximate solution
    approxs = model.solve(grid, fetches=fetches).reshape(x_size, y_size)

    # plot approximate solution
    if mode == 'imshow':
        plt.imshow(approxs, cmap=cmap)
    elif mode == 'contourf':
        plt.contourf(xs_grid, ys_grid, approxs, cmap=cmap)
    else:
        from mpl_toolkits.mplot3d import Axes3D             # pylint: disable=unused-import, import-outside-toplevel
        fig = plt.figure()
        axis = fig.gca(projection='3d')
        axis.plot_surface(xs_grid, ys_grid, approxs, rstride=1, cstride=1,
                          cmap=cmap, edgecolor='none')
        axis.set_title(title, size=17)
        axis.set_xlabel(r'$x$', fontdict={'fontsize': 14})
        axis.set_ylabel(r'$y$', fontdict={'fontsize': 14})

    if mode in ('imshow', 'contourf'):
        plt.title(title, fontdict={'fontsize': 17})
        plt.colorbar()

    if save_to is not None:
        plt.savefig(save_to, dpi=300)

    if mode in ('3d_view', 'contourf'):
        plt.show()


def cart_prod(*arrs):
    """ Get array of cartesian tuples from arbitrary number of arrays.
    """
    grids = np.meshgrid(*arrs, indexing='ij')
    return np.stack(grids, axis=-1).reshape(-1, len(arrs))


def plot_sections_2d(model, timestamps=(0, 0.2, 0.4, 0.6, 0.7, 0.9), grid_size=(2, 3), points=None,
                     fetches=None, xlim=(0, 1), ylim=(0, 0.3), title=r'Heat PDE in $\mathcal{R}$: $\hat{u}$',
                     save_to=None):
    r""" Plot 1d-time-sections of approximate solution to 2d-evolution equation, that is, with one
    spatial coordinate.
    """
    # set up grid of points
    points = points if points is not None else np.linspace(*xlim, 100).reshape(-1, 1)
    fig, axes = plt.subplots(*grid_size, figsize=(5 * grid_size[1], 5))

    # loop over and plot time-sections
    for i, t in enumerate(timestamps):
        points_ = np.concatenate([points.reshape(-1, 1), t * np.ones((points.shape[0], 1))], axis=1)
        wx, wy = i // grid_size[1], i % grid_size[1]
        axes[wx, wy].plot(points.reshape(-1), model.solve(points_, fetches=fetches))
        axes[wx, wy].set_ylim(*ylim)
        axes[wx, wy].set_title('$t=%.2f$' % t, size=22)

    # add title, save if needed and show the plot
    fig.tight_layout()
    fig.suptitle(title, size=27)
    fig.subplots_adjust(top=0.82)

    if save_to is not None:
        plt.savefig(save_to, dpi=300)
    plt.show()


def plot_sections_3d(model, timestamps=(0, 0.2, 0.4, 0.6, 0.7, 0.9), grid_size=(2, 3), mode='3d_view',
                     fetches=None, xlim=(0, 1), ylim=(0, 1), zlim=(-0.2, 0.2), num_points=None,
                     title=r'Heat PDE in $\mathcal{R}^2$: $\hat{u}$', cmap='viridis',
                     save_to=None):
    """ Plot 2d-sections of approximate solution to 3d-evolution equation, that is, with two
    spatial coordinates.
    """
    # set up grid of points
    if num_points is None:
        if mode in ('imshow', 'contourf'):
            num_points = 80
        else:
            num_points = 20

    xs = np.linspace(*xlim, num_points)
    ys = np.linspace(*ylim, num_points)
    grid = cart_prod(xs, ys)
    xs_grid, ys_grid = np.meshgrid(xs, ys)

    # loop over and plot time-sections
    fig = plt.figure(figsize=(grid_size[1]*3.5, grid_size[0]*3.5))
    for i, t in enumerate(timestamps):
        grid_t = np.concatenate([grid, t * np.ones((grid.shape[0], 1))], axis=1)

        # add new canvas
        if mode == '3d_view':
            from mpl_toolkits.mplot3d import Axes3D         # pylint: disable=unused-import, import-outside-toplevel
            axis = fig.add_subplot(*grid_size, i + 1, projection='3d')
        else:
            axis = fig.add_subplot(*grid_size, i + 1)

        # calculate and plot approximate solution for a section
        zs_grid = model.solve(grid_t, fetches=fetches).reshape(len(xs), len(ys))
        if mode == 'imshow':
            axis.imshow(zs_grid, cmap=cmap)
        elif mode == 'contourf':
            axis.contourf(xs_grid, ys_grid, zs_grid, cmap=cmap)
        elif mode == '3d_view':
            axis.plot_surface(xs_grid, ys_grid, zs_grid, rstride=1, cstride=1, cmap=cmap, edgecolor='none')
            axis.set_zlim(*zlim)
        axis.set_title('$t=%.2f$' % t, size=17)
        axis.set_xlabel(r'$x$', fontdict={'fontsize': 14})
        axis.set_ylabel(r'$y$', fontdict={'fontsize': 14})

    # add title, save if needed and show the plot
    fig.tight_layout()
    fig.suptitle(title, size=20)
    if save_to is not None:
        plt.savefig(save_to, dpi=300)
    plt.show()
