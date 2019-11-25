""" PDE-Solver class. Wraps up `TFDeepGalerkin`-model. """
import tensorflow as tf
from tqdm import tqdm_notebook, tqdm

from .model_tf import TFDeepGalerkin


class Solver:
    """ Solver-class for PDE-problems. Wraps up `TFDeepGalerkin`-model for convenient
    initialization, training and inference.

    Parameters
    ----------
    model_class : class
        class to use when buliding the model. Should inherit `TFDeepGalerkin` class.
    config : dict
        Configuration of model. Supports all of the options from `model_class`.
    layer_size : int
        When neural-network architecture is not specified in `config`, the default one
        is used. This parameter regulates the width of the default architecture.
    path : str or None
        if supplied, `Solver`-instance is initialized from a saved model.
    """
    def __init__(self, config=None, model_class=None, layer_size=15, path=None):
        config = config or {}
        model_class = model_class or config.get('model_class') or TFDeepGalerkin
        config = self.build_config(config, layer_size, path)
        self.model = model_class(config)


    def build_config(self, config, layer_size, path):
        """ Build model-config. Add default neural network configuration if needed. """
        if path is not None:
            config = {'load': {'path': path}}
            return config

        n_dims = config['pde']['n_dims']

        if config.get('body') is None:
            block = 'fa' if n_dims == 1 else 'Rfa+'
            layout = 'fafa ' + block*(n_dims-1)
            units = [layer_size//1.5] + [layer_size]*(n_dims)
            default_body = dict(layout=layout,
                                units=units,
                                activation=tf.nn.tanh)
            config['body'] = default_body

        if config.get('head') is None:
            default_head = dict(layout='fa f',
                                units=[layer_size//1.5, 1],
                                activation=tf.nn.tanh)
            config['head'] = default_head

        return config


    def fit(self, batch_size, sampler, n_iters, train_mode='', fetches=None, bar=False):
        """ Train model on batches of sampled points.

        Parameters
        ----------
        sampler : Sampler
            Generator of training points. Must provide points from the same
            dimension, as PDE.
        batch_size : int
            Number of points in each training batch.
        n_iters : int
            Number of times to generate data and train on it.
        fetches : str or sequence of str
            `tf.Operation`s and/or `tf.Tensor`s to calculate.
        train_mode : str
            Name of train step to optimize.
        bar : str
            Whether to show progress bar during training.
        """
        if fetches is None:
            fetches = ['loss' + '_'*int(len(name) > 0) + name
                       for name in self.model.get_from_attr('train_steps').keys()]
        fetches = [fetches] if isinstance(fetches, str) else fetches

        for name in fetches:
            if not hasattr(self, name):
                setattr(self, name, [])

        iterator = range(n_iters)
        bars = {'notebook': tqdm_notebook,
                'tqdm': tqdm,
                True: tqdm,
                False: lambda x: x,
                None: lambda x: x}
        iterator = bars[bar](iterator)

        for _ in iterator:
            points = sampler.sample(batch_size)
            fetched = self.model.train(fetches=fetches, feed_dict={'points': points}, train_mode=train_mode)
            for tensor, name in zip(fetched, fetches):
                getattr(self, name).append(tensor)


    def solve(self, points=None, fetches=None):
        """ Predict values of function on array of points.

        Parameters
        ----------
        points : array-like
            Points to give solution approximation on.
        fetches : str or sequence of str
            `tf.Operation`s and/or `tf.Tensor`s to calculate.

        Returns
        -------
        Calculated values of tensors in `fetches` in the same order and structure.
        """
        if points is not None:
            return self.model.predict(fetches=fetches,
                                      feed_dict={'points': points})
        return self.model.predict(fetches=fetches)

    def save(self, path):
        """ Save trained Solver-model for later usage.

        Parameters
        ----------
        path : str
            folder where model-files (graph and trained weights) will be stored.
        """
        self.model.save(path=path)
