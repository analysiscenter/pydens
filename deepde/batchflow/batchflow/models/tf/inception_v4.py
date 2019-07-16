""" Christian Szegedy et al. "`Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
<https://arxiv.org/abs/1602.07261>`_" """
import tensorflow as tf

from .layers import conv_block
from .inception_base import Inception


_DEFAULT_V4_ARCH = {
    'A': {'filters': [[96, 64]]*4},
    'r': {'filters': (192, 224, 256, 384)},
    'B': {'filters': [[384, 192, 224, 256, 128]]*7},
    'G': {'filters': (192, 256, 320)},
    'C': {'filters': [[256, 384, 448, 512]]*3}}


class Inception_v4(Inception):
    """ Inception_v4

    **Configuration**

    inputs : dict
        dict with 'images' and 'masks' (see :meth:`~.TFModel._make_inputs`)
   """
    @classmethod
    def default_config(cls):
        """ Default parameters for Inception_v4 model.

        Returns
        -------
        config : dict
            default parameters to network
        """
        config = Inception.default_config()

        config['initial_block'] = dict(layout='cna', filters=[32, 64, 96, 192],
                                       pool_size=3, pool_strides=2)
        config['body']['layout'] = 'AAAArBBBBBBBGCCC'
        config['body']['arch'] = _DEFAULT_V4_ARCH
        config['head'].update(dict(layout='Vdf', dropout_rate=.8))
        config['loss'] = 'ce'

        return config

    @classmethod
    def initial_block(cls, inputs, name='initial_block', **kwargs):
        """Input network block.

        For details see figure 3 in the article.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            kwargs = cls.fill_params('initial_block', **kwargs)
            layout, filters = cls.pop(['layout', 'filters'], kwargs)
            axis = cls.channels_axis(kwargs['data_format'])

            x = conv_block(inputs, layout*2, filters[0]*2, 3, name='conv_3_3', padding='valid',
                           strides=[2, 1], **kwargs)
            x = conv_block(x, layout, filters[1], 3, name='conv_3_3_3', **kwargs)

            branch_3 = conv_block(x, layout, filters[2], 3, name='conv_3', strides=2, padding='valid', **kwargs)
            branch_pool = conv_block(x, layout='p', name='max_pool', padding='valid', **kwargs)
            x = tf.concat([branch_3, branch_pool], axis, name='concat_3_and_pool')

            branch_1 = conv_block(x, layout, filters[1], 1, name='conv_1', **kwargs)
            branch_1_3 = conv_block(branch_1, layout, filters[2], 3, name='conv_1_3', padding='valid', **kwargs)

            branch_1_7 = conv_block(x, layout*3, [filters[1]]*3, [1, [7, 1], [1, 7]], name='conv_1_7', **kwargs)
            branch_1_7_3 = conv_block(branch_1_7, layout, filters[2], 3, name='conv_1_7_3', padding='valid', **kwargs)
            x = tf.concat([branch_1_3, branch_1_7_3], axis, name='concat_1_3_and_1_7_3')

            branch_out_3 = conv_block(x, layout, filters[3], 3, name='conv_out_3', strides=2,
                                      padding='valid', **kwargs)
            branch_out_pool = conv_block(x, layout='p', name='out_max_pooling', padding='valid', **kwargs)

            output = tf.concat([branch_out_3, branch_out_pool], axis, name='output')
        return output

    @classmethod
    def inception_a_block(cls, inputs, filters, layout='cna', name='inception_a_block', **kwargs):
        """ Inception block A.

        For details see figure 4 in the article.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : tuple of 2 int
            number of output filters
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            branch_1 = conv_block(inputs, layout, filters[0], 1, name='conv_1', **kwargs)

            branch_1_3 = conv_block(inputs, layout*2, [filters[1], filters[0]], [1, 3], name='conv_1_3', **kwargs)

            branch_1_3_3 = conv_block(inputs, layout*3, [filters[1]]+[filters[0]]*2, [1, 3, 3], name='conv_1_3_3',
                                      **kwargs)

            branch_pool = conv_block(inputs, 'v'+layout, filters[0], 1, name='c_pool', **{**kwargs, 'pool_strides': 1})

            axis = cls.channels_axis(kwargs['data_format'])
            output = tf.concat([branch_1, branch_1_3, branch_1_3_3, branch_pool], axis, name='output')
        return output

    @classmethod
    def inception_b_block(cls, inputs, filters, layout='cna', name='inception_b_block', **kwargs):
        """ Inception block B.

        For details see figure 5 in the article.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : tuple of 5 int
            number of output filters
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            branch_1 = conv_block(inputs, layout, filters[0], 1, name='conv_1_3', **kwargs)

            factor = [[1, 7], [7, 1]]
            kernel_size = [1, *factor]
            branch_1_7 = conv_block(inputs, layout*3, [filters[1], filters[2], filters[3]], kernel_size,
                                    name='conv_1_7', **kwargs)

            kernel_size = [1, *factor*2]
            branch_1_7_7 = conv_block(inputs, layout*5, [filters[1]]*2+[filters[2]]*2+[filters[3]], kernel_size,
                                      name='conv_1_7_7', **kwargs)

            branch_pool = conv_block(inputs, 'v'+layout, filters[4], 1, name='c_pool', **{**kwargs, 'pool_strides': 1})

            axis = cls.channels_axis(kwargs['data_format'])
            output = tf.concat([branch_1, branch_1_7, branch_1_7_7, branch_pool], axis, name='output')
        return output

    @classmethod
    def reduction_grid_block(cls, inputs, filters, layout='cna', name='reduction_b_block', **kwargs):
        """ grid-reduction block.

        For details see figure 8 in the article.

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
            branch_1_3 = conv_block(branch_1, layout, filters[0], 3, name='conv_1_3', strides=2,
                                    padding='valid', **kwargs)

            branch_1_7 = conv_block(inputs, layout*3, [filters[1]]*2+[filters[2]], [1, [1, 7], [7, 1]],
                                    name='conv_1_7', **kwargs)
            branch_1_7_3 = conv_block(branch_1_7, layout, filters[2], 3, name='conv_1_7_3', strides=2,
                                      padding='valid', **kwargs)

            branch_pool = conv_block(inputs, layout='p', name='max_pooling', pool_size=3, pool_strides=2,
                                     padding='valid', **kwargs)

            axis = cls.channels_axis(kwargs['data_format'])
            output = tf.concat([branch_1_3, branch_1_7_3, branch_pool], axis, name='output')
        return output

    @classmethod
    def inception_c_block(cls, inputs, filters, layout='cna', name='inception_c_block', **kwargs):
        """ Inception block C.

        For details see figure 6 in the article.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : tuple of 4 int
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

            branch_1 = conv_block(inputs, layout, filters[1], 1, name='conv_1_3', **kwargs)
            branch_1_13 = conv_block(branch_1, layout, filters[0], [1, 3], name='conv_1_13', **kwargs)
            branch_1_31 = conv_block(branch_1, layout, filters[0], [3, 1], name='conv_1_31', **kwargs)
            branch_1_33 = tf.concat([branch_1_13, branch_1_31], axis)

            branch_1_13_31 = conv_block(inputs, layout*3, [filters[1], filters[2], filters[3]], [1, [1, 3], [3, 1]],
                                        name='conv_1_13_31', **kwargs)
            branch_1_3_13 = conv_block(branch_1_13_31, layout, filters[0], [1, 3], name='conv_1_3_13', **kwargs)
            branch_1_3_31 = conv_block(branch_1_13_31, layout, filters[0], [3, 1], name='conv_1_3_31', **kwargs)
            branch_1_5 = tf.concat([branch_1_3_13, branch_1_3_31], axis)

            branch_pool = conv_block(inputs, 'v'+layout, filters[0], 1, name='c_pool', **{**kwargs, 'pool_strides': 1})

            output = tf.concat([branch_1, branch_1_33, branch_1_5, branch_pool], axis, name='output')
        return output
