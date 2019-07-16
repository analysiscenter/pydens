""" Christian Szegedy et al. "`Rethinking the Inception Architecture for Computer Vision
<https://arxiv.org/abs/1512.00567>`_"
"""
import tensorflow as tf

from .layers import conv_block
from .inception_base import Inception


_DEFAULT_V3_ARCH = {
    'b': {'filters': [[64, 48, 96, 32], [64, 48, 96, 64], [64, 48, 96, 64]]},
    'r': {'filters': (64, 96, 96, 384)},
    'f': {'filters': [[192, 128],
                      [192, 160],
                      [192, 160],
                      [192, 192]]},
    'm': {'filters': (192, 320)},
    'e': {'filters': (320, 384, 448, 192)}
}

class Inception_v3(Inception):
    """ The base Inception_v3 model

    Notes
    -----
    Since the article misses some important details (e.g. the number of filters in all convolutions),
    this class is based on `Google's implementation
    <https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py>`_

    **Configuration**

    inputs : dict
        dict with 'images' and 'masks' (see :meth:`~.TFModel._make_inputs`)
    """
    @classmethod
    def default_config(cls):
        """ Define the default parameters to inception_v3 model.

        Returns
        -------
        config : dict
            default parameters to build network
        """
        config = Inception.default_config()
        config['initial_block'] = dict(layout='cna cna cnap cna cnap', filters=[32, 32, 64, 80, 192],
                                       kernel_size=[3, 3, 3, 1, 3], strides=[2, 1, 1, 1, 1],
                                       pool_size=3, pool_strides=2, padding='valid')
        config['body']['layout'] = 'bbbrffffmee'
        config['body']['arch'] = _DEFAULT_V3_ARCH
        config['head'].update(dict(layout='Vdf', dropout_rate=.8))
        config['loss'] = 'ce'

        return config

    @classmethod
    def block(cls, inputs, filters, layout='cna', name='block', **kwargs):
        """ Network building block.

        For details see figure 5 in the article.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : tuple of 4 ints
            number of output filters
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            branch_1 = conv_block(inputs, layout, filters[0], 1, name='conv_1', **kwargs)

            branch_1_3 = conv_block(inputs, layout*2, [filters[1], filters[0]], [1, 5], name='conv_1_3', **kwargs)

            branch_1_3_3 = conv_block(inputs, layout*3, [filters[0]]+[filters[2]]*2, [1, 3, 3], name='conv_1_3_3',
                                      **kwargs)

            branch_pool = conv_block(inputs, 'p'+layout, filters[3], 1, 'c_pool', **{**kwargs, 'pool_strides': 1})

            axis = cls.channels_axis(kwargs['data_format'])
            output = tf.concat([branch_1, branch_1_3, branch_1_3_3, branch_pool], axis=axis, name='output')
        return output

    @classmethod
    def mixed_block(cls, inputs, filters, layout='cna', name='mixed_block', **kwargs):
        """ Mixed block.

        This block is not described in the paper, but is used in `Google's implementation
        <https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py#L347:L364>`_.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : tuple of 2 ints
            number of output filters
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            branch_1 = conv_block(inputs, layout, filters[0], 1, name='conv_1', **kwargs)
            branch_1_3 = conv_block(branch_1, layout, filters[1], 3, name='conv_1_3', strides=2,
                                    padding='valid', **kwargs)

            branch_1_7 = conv_block(inputs, layout*3, filters[0], [1, (1, 7), (7, 1)], name='conv_1_7', **kwargs)
            branch_1_7_3 = conv_block(branch_1_7, layout, filters[0], 3, name='conv_1_7_3', strides=2,
                                      padding='valid', **kwargs)

            branch_pool = conv_block(inputs, layout='p', name='c_pool', padding='valid', pool_size=3,
                                     pool_strides=2, **kwargs)

            axis = cls.channels_axis(kwargs['data_format'])
            output = tf.concat([branch_1_3, branch_1_7_3, branch_pool], axis, name='output')
        return output

    @classmethod
    def factorization_block(cls, inputs, filters, layout='cna', name='factorization_block', **kwargs):
        """ 7x7 factorization block.

        For details see figure 6 in the article.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : tuple of 3 int
            number of output filters
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            branch_1 = conv_block(inputs, layout, filters[0], 1, name='conv_1', **kwargs)

            factor = [1, 7], [7, 1]
            kernel_size = [1, *factor]
            branch_1_3 = conv_block(inputs, layout * 3, [filters[1]] * 2 + [filters[0]], kernel_size,
                                    name='conv_1_3', **kwargs)

            kernel_size = [1, *factor[::-1] * 2]
            branch_1_7 = conv_block(inputs, layout * 5, [filters[1]] * 4 + [filters[0]], kernel_size,
                                    name='conv_1_7', **kwargs)

            branch_pool = conv_block(inputs, 'p'+layout, filters[0], 1, name='c_pool', **{**kwargs, 'pool_strides': 1})

            axis = cls.channels_axis(kwargs['data_format'])
            output = tf.concat([branch_1, branch_1_3, branch_1_7, branch_pool], axis=axis, name='output')
        return output

    @classmethod
    def expanded_block(cls, inputs, filters, layout='cna', name='expanded_block', **kwargs):
        """ Network building block.

        For details see figure 7 in the article.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : tuole of 4 ints
            number of output filters
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            axis = cls.channels_axis(kwargs['data_format'])
            branch_1 = conv_block(inputs, layout, filters[0], 1, name='conv_1', **kwargs)

            branch_pool = conv_block(inputs, 'p'+layout, filters[3], 1, name='c_pool',
                                     **{**kwargs, 'pool_strides': 1})

            branch_a1 = conv_block(inputs, layout, filters[1], 1, name='conv_a1', **kwargs)
            branch_a1_31 = conv_block(branch_a1, layout, filters[1], [3, 1], name='conv_1_31', **kwargs)
            branch_a1_13 = conv_block(branch_a1, layout, filters[1], [1, 3], name='conv_1_13', **kwargs)
            branch_a = tf.concat([branch_a1_31, branch_a1_13], axis=axis)

            branch_b13 = conv_block(inputs, layout*2, [filters[2], filters[1]], [1, 3], name='conv_b13', **kwargs)
            branch_b13_31 = conv_block(branch_b13, layout, filters[1], [3, 1], name='conv_b13_31', **kwargs)
            branch_b13_13 = conv_block(branch_b13, layout, filters[1], [1, 3], name='conv_b13_13', **kwargs)
            branch_b = tf.concat([branch_b13_31, branch_b13_13], axis=axis)

            output = tf.concat([branch_1, branch_pool, branch_a, branch_b], axis=axis, name='output')
        return output
