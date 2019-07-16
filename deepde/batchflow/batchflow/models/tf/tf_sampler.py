""" Contains tensorflow-sampler class. """

from copy import copy
import tensorflow as tf
from ...sampler import Sampler, _get_method_by_alias



class TfSampler(Sampler):
    """ Sampler based on a distribution from tf.distributions.

    Parameters
    ----------
    name : str
        name of a distribution (class from tf.distributions), or its alias.
    **kwargs
        additional keyword-args for distribution specification.
        E.g., `loc` for name='Normal'

    Attributes
    ----------
    name : str
        name of a distribution (class from tf.distributions).
    _params : dict
        dict of args for distribution specification.
    graph : tf.Graph
        graph in which sampling nodes are placed.
    sampler : tf.distributions
        instance of distributions' class.
    sess : tf.Session
        session used for running sample-tensor.
    """
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        name = _get_method_by_alias(name, 'tf', tf.distributions)
        self.name = name
        self._params = copy(kwargs)
        self.graph = tf.Graph()
        with self.graph.as_default():
            config = tf.ConfigProto(device_count={'GPU':0})
            self.sess = tf.Session(config=config)
            _ = kwargs.pop('dim', None)
            self.sampler = getattr(tf.distributions, self.name)(**kwargs)

    def sample(self, size):                 # pylint: disable=method-hidden
        """ Sampling method of ``TfSampler``.

        Generates random samples from distribution ``self.name``.

        Parameters
        ----------
        size : int
            the size of sample to be generated.

        Returns
        -------
        np.ndarray
            array of shape (size, Sampler's dimension).
        """
        with self.graph.as_default():
            _sample = self.sampler.sample(size)

        sample = self.sess.run(_sample)

        if len(sample.shape) == 1:                                          # pylint: disable=no-member
            sample = sample.reshape(-1, 1)                                  # pylint: disable=no-member
        return sample
