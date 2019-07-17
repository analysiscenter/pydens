""" W. """

from tqdm import tqdm_notebook, tqdm

from .model_tf import TFDeep
from .model_torch import TorchDeep



class DeepSolver:
    """ Wrapper around `TFDeep` and `TorchDeep` to surpass BatchFlow's syntax sugar.

    Parameters
    ----------
    model_class : class
        class to use when buliding the model. Should inherit `TFDeep` or `TorchDeep` class.
    config : dict
        Configuration of model. Supports all of the options from `model_class`.
    """
    def __init__(self, config, model_class=TFDeep):
        self.model = model_class(config)


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
