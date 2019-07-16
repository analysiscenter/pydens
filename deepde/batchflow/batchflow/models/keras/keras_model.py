 # pylint: disable=super-init-not-called
 # pylint: disable=unused-argument
""" Contains base class for keras models. """

import functools
import numpy as np
from keras.models import Model
from ..base import BaseModel


class KerasModel(Model, BaseModel):
    """ Base class for all keras models.

    Contains load, dump and compile methods which are shared between all
    keras models.
    Also implements train and predict methods.

    """
    def __init__(self, *args, **kwargs):
        """ Call __init__ of BaseModel not keras.models.Model. """
        BaseModel.__init__(self, *args, **kwargs)

    def build(self, *args, **kwargs):
        """ Must return inputs and outputs. """
        input_nodes, output_nodes = self._build(**self.config)
        Model.__init__(self, input_nodes, output_nodes)
        self.compile(loss=self.get('loss', self.config, default=None),
                     optimizer=self.get('optimizer', self.config, default='sgd'))

    def _build(self, *args, **kwargs):
        """ Must return inputs and outputs. """
        raise NotImplementedError("This method must be implemented in ancestor model class")

    def train(self, x=None, y=None, **kwargs):
        """ Wrapper for keras.models.Model.train_on_batch.

        Checks whether feed_dict is None and unpacks it as kwargs
        of keras.models.Model.train_on_batch method.

        Parameters
        ----------
        x : ndarray(batch_size, ...)
            x argument of keras.models.Model.train_on_batch method, input of
            neural network.
        y : ndarray(batch_size, ...)
            y argument of keras.models.Model.predict_on_batch method.

        Returns
        -------
        ndarray(batch_size, ...)
            predictions of keras model.

        Raises
        ------
        ValueError if 'x' or 'y'  is None.
        """
        if x is None or y is None:
            raise ValueError("Arguments 'x' and 'y' must not be None")

        prediction = np.asarray(self.train_on_batch(x=x, y=y))
        return prediction

    def predict(self, x=None, **kwargs):
        """ Wrapper for keras.models.Model.predict_on_batch.

        Checks whether feed_dict is None and unpacks it
        as kwargs of keras.models.Model.predict_on_batch method.

        Parameters
        ----------
        x : ndarray(batch_size, ...)
            x argument of keras.models.Model.predict_on_batch method, input of
            neural network.

        Returns
        -------
        ndarray(batch_size, ...)
            predictions of keras model.

        Raises
        ------
        ValueError if 'x' argument is None.
        """
        if x is not None:
            return Model.predict_on_batch(self, x=x)
        raise ValueError("Argument 'x' must not be None")

    @functools.wraps(Model.load_weights)
    def load(self, *args, **kwargs):
        """ Wrapper for keras.models.Model.load_weights. """
        return Model.load_weights(self, *args, **kwargs)

    @functools.wraps(Model.save_weights)
    def save(self, *args, **kwargs):
        """ Wrapper for keras.models.Model.save_weights. """
        return Model.save_weights(self, *args, **kwargs)
