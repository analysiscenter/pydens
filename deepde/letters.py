""" Custom mathematical tokens for `deepde` syntax. """

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import torch

try:
    from autograd import grad
    import autograd.numpy as autonp
except ImportError:
    pass

#pylint: disable=no-name-in-module, import-error
from .batchflow.models.tf.layers import conv_block
from .batchflow.models.torch.layers import ConvBlock



class Letters(ABC):
    """ Abstract class for custom letters. Defines which letters should be implemented. """
    @abstractmethod
    def D(self, *args, **kwargs):
        """ `D` letter: taking gradient of the first argument with respect to the second. """

    @abstractmethod
    def P(self, *args, **kwargs):
        """ `P` letter: controllable from the outside perturbation. """

    @abstractmethod
    def R(self, *args, **kwargs):
        """ `R` letter: dynamically generated random noise. """

    @abstractmethod
    def V(self, *args, **kwargs):
        """ `V` letter: adjustable variation of the coefficient. """

    @abstractmethod
    def C(self, *args, **kwargs):
        """ `C` letter: small neural network inside equation. """



class TFLetters(Letters):
    """ TensorFlow implementations of custom letters. """
    @staticmethod
    def D(*args, **kwargs):
        _ = kwargs
        return tf.gradients(args[0], args[1])[0]


    @staticmethod
    def P(*args, **kwargs):
        _ = kwargs
        if len(args) != 1:
            raise ValueError('`P` is reserved to create exactly one perturbation at a time. ')
        return tf.identity(args[0])


    @staticmethod
    def R(*args, **kwargs):
        if len(args) > 2:
            raise ValueError('`R` works with 2 arguments at most. ')
        if len(args) == 2:
            inputs, scale = args
            shape = tf.shape(inputs)
        else:
            scale = args[0] if len(args) == 1 else 1
            # If model graph is already build, we can infer batch size from shape of one of the tensors
            # Otherwise, we use placeholder
            try:
                points = TFLetters.tf_check_tensor('inputs', 'concat', ':0')
                shape = (tf.shape(points)[0], 1)
            except KeyError:
                shape = ()

        distribution = kwargs.pop('distribution', 'normal')

        if distribution == 'normal':
            noise = tf.random.normal(shape=shape, stddev=scale)
        if distribution == 'uniform':
            noise = tf.random.uniform(shape=shape, minval=-scale, maxval=scale)
        return noise


    @staticmethod
    def V(*args, prefix='addendums', **kwargs):
        # Parsing arguments
        _ = kwargs
        *args, name = args
        if not isinstance(name, str):
            raise ValueError('`W` last positional argument should be its name. Instead got {}'.format(name))
        if len(args) > 1:
            raise ValueError('`W` can work only with one initial value. ')
        x = args[0] if len(args) == 1 else 0.0

        # Try to get already existing variable with the given name from current graph.
        # If it does not exist, create one
        try:
            var = TFLetters.tf_check_tensor(prefix, name)
            return var
        except KeyError:
            var_name = prefix + '/' + name
            var = tf.Variable(x, name=var_name, dtype=tf.float32, trainable=True)
            var = tf.identity(var, name=var_name + '/_output')
            return var


    @staticmethod
    def C(*args, prefix='addendums', **kwargs):
        *args, name = args
        if not isinstance(name, str):
            raise ValueError('`C` last positional argument should be its name. Instead got {}'.format(name))

        defaults = dict(layout='faf',
                        units=[15, 1],
                        activation=tf.nn.tanh)
        kwargs = {**defaults, **kwargs}

        try:
            block = TFLetters.tf_check_tensor(prefix, name)
            return block
        except KeyError:
            block_name = prefix + '/' + name
            points = tf.concat(args, axis=-1, name=block_name + '/concat')
            block = conv_block(points, name=block_name, **kwargs)
            return block


    @staticmethod
    def tf_check_tensor(prefix=None, name=None, postfix='/_output:0'):
        """ Convenient wrapper around `get_tensor_by_name`. """
        tensor_name = tf.get_variable_scope().name + '/' + prefix + '/' + name + postfix
        graph = tf.get_default_graph()
        tensor = graph.get_tensor_by_name(tensor_name)
        return tensor



class NPLetters(Letters):
    """ NumPy implementations of custom letters. """
    #pylint: disable=abstract-method
    try:
        _ = grad, autonp
    except NameError:
        _ = np

    @staticmethod
    def D(*args, **kwargs):
        _ = kwargs
        return grad(args[0])(args[1])



class TorchLetters(Letters):
    """ PyTorch implementations of custom letters. """
    #pylint: disable=abstract-method
    _ = torch, ConvBlock

    @staticmethod
    def D(*args, **kwargs):
        _ = kwargs
        return torch.autograd.grad(args[0], args[1])[0]
