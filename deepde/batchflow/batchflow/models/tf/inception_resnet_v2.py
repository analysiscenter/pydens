""" Szegedy C. et al "`Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
<https://arxiv.org/abs/1602.07261>`_"
"""
import tensorflow as tf

from .inception_base import Inception
from .layers import conv_block


_DEFAULT_ARCH = {
    'A': {'filters': (32, 32, 32, 32, 48, 64, 384)},
    'B': {'filters': (192, 128, 160, 192, 1152)},
    'C': {'filters': (192, 192, 224, 256, 2144)},
    'a': {'filters': (384, 256, 256, 384)},
    'b': {'filters': (256, 384, 256, 288, 256, 288, 320)},
}


class InceptionResNet_v2(Inception):
    """ Inception-ResNet network, version 2

    **Configuration**

    inputs : dict
        dict with 'images' and 'labels' (see :meth:`~.TFModel._make_inputs`)

    body/arch : dict
        architecture: network layout, block layout, number of filters in each block, pooling parameters
    """
    @classmethod
    def default_config(cls):
        config = Inception.default_config()
        config['common']['layout'] = 'cna'
        config['initial_block'] = dict(layout='cna', filters=[32, 64, 96, 192],
                                       pool_size=3, pool_strides=2)
        config['body']['layout'] = 'A'*5 + 'a' + 'B'*10 +'b' + 'C'*5
        config['body']['arch'] = _DEFAULT_ARCH
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
    def body(cls, inputs, name='body', **kwargs):
        """ Base layers.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        layout : str
            a sequence of blocks
        arch : dict
            parameters for each block
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body', **kwargs)
        arch, layout = cls.pop(['arch', 'layout'], kwargs)

        with tf.variable_scope(name):
            x, inputs = inputs, None

            for i, block in enumerate(layout):

                block_args = {**kwargs, **arch[block]}

                if block == 'A':
                    x = cls.block_a(x, name='block_a-%d'%i, **block_args)
                elif block == 'B':
                    x = cls.block_b(x, name='block_b-%d'%i, **block_args)
                elif block == 'C':
                    x = cls.block_c(x, name='block_c-%d'%i, **block_args)
                elif block == 'a':
                    x = cls.reduction_a(x, name='reduction_a-%d'%i, **block_args)
                elif block == 'b':
                    x = cls.reduction_b(x, name='reduction_b-%d'%i, **block_args)

        return x


    @classmethod
    def block_a(cls, inputs, filters, layout='cna', name='block_a', **kwargs):
        """ Inception block A.

        For details see figure 16 in the article.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : tuple of 6 int
            number of output filters
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            x = tf.nn.relu(inputs)

            branch_1 = conv_block(x, layout, filters[0], 1, name='conv_1', **kwargs)
            branch_2 = conv_block(x, layout*2, [filters[1], filters[2]], [1, 3], name='conv_2', **kwargs)
            branch_3 = conv_block(x, layout*3, [filters[3], filters[4], filters[5]], [1, 3, 3], name='conv_3', **kwargs)

            axis = cls.channels_axis(kwargs['data_format'])
            branch_1 = tf.concat([branch_1, branch_2, branch_3], axis=axis)
            branch_1 = conv_block(branch_1, 'c', filters[6], 1, name='conv_1x1', **kwargs)

            x = x + branch_1

            x = tf.nn.relu(x)

        return x

    @classmethod
    def block_b(cls, inputs, filters, layout='cna', name='block_b', **kwargs):
        """ Inception block B.

        For details see figure 17 in the article.

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
            x = tf.nn.relu(inputs)

            branch_1 = conv_block(x, layout, filters[0], 1, name='conv_1', **kwargs)
            branch_2 = conv_block(x, layout*3, [filters[1], filters[2], filters[3]], [1, (1, 7), (7, 1)],
                                  name='conv_2', **kwargs)

            axis = cls.channels_axis(kwargs['data_format'])
            branch_1 = tf.concat([branch_1, branch_2], axis=axis)
            branch_1 = conv_block(branch_1, 'c', filters[4], 1, name='conv_1x1', **kwargs)

            x = x + branch_1

            x = tf.nn.relu(x)

        return x

    @classmethod
    def block_c(cls, inputs, filters, layout='cna', name='block_c', **kwargs):
        """ Inception block C.

        For details see figure 19 in the article.

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
            x = tf.nn.relu(inputs)

            branch_1 = conv_block(x, layout, filters[0], 1, name='conv_1', **kwargs)
            branch_2 = conv_block(x, layout*3, [filters[1], filters[2], filters[3]], [1, (1, 3), (3, 1)],
                                  name='conv_2', **kwargs)

            axis = cls.channels_axis(kwargs['data_format'])
            branch_1 = tf.concat([branch_1, branch_2], axis=axis)
            branch_1 = conv_block(branch_1, 'c', filters[4], 1, name='conv_1x1', **kwargs)

            x = x + branch_1

            x = tf.nn.relu(x)

        return x

    @classmethod
    def reduction_a(cls, inputs, filters, layout='cna', name='reduction_a', **kwargs):
        """ Reduction block A.

        For details see figure 7 and Table 1 in the article.

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
            x = tf.nn.relu(inputs)

            branch_1 = conv_block(x, 'p', pool_strides=2, name='max-pool', **kwargs)
            branch_2 = conv_block(x, layout, filters[0], 3, strides=2, name='conv_2', **kwargs)
            branch_3 = conv_block(x, layout*3, [filters[1], filters[2], filters[3]], [1, 3, 3],
                                  strides=[1, 1, 2], name='conv_3', **kwargs)

            axis = cls.channels_axis(kwargs['data_format'])
            x = tf.concat([branch_1, branch_2, branch_3], axis=axis)

        return x

    @classmethod
    def reduction_b(cls, inputs, filters, layout='cna', name='reduction_a', **kwargs):
        """ Reduction block B.

        For details see figure 18 in the article.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : tuple of 7 int
            number of output filters
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            x = inputs
            branch_1 = conv_block(x, 'p', pool_size=3, pool_strides=2, name='max-pool', **kwargs)
            branch_2 = conv_block(x, layout*2, [filters[0], filters[1]], [1, 3], strides=[1, 2],
                                  name='conv_2', **kwargs)
            branch_3 = conv_block(x, layout*2, [filters[2], filters[3]], [1, 3], strides=[1, 2],
                                  name='conv_3', **kwargs)
            branch_4 = conv_block(x, layout*3, [filters[4], filters[5], filters[6]], [1, 3, 3], strides=[1, 1, 2],
                                  name='conv_4', **kwargs)

            axis = cls.channels_axis(kwargs['data_format'])
            x = tf.concat([branch_1, branch_2, branch_3, branch_4], axis)

        return x
