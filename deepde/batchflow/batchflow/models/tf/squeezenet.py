""" Iandola F. et al. "`SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size"
<https://arxiv.org/abs/1602.07360>`_"
"""
import numpy as np
import tensorflow as tf

from . import TFModel
from .layers import conv_block


class SqueezeNet(TFModel):
    """ SqueezeNet neural network

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'labels' (see :meth:`~.TFModel._make_inputs`)

    body : dict
        layout : str
            A sequence of blocks:

            - f : fire block
            - m : max-pooling
            - b : bypass

            Default is 'fffmffffmf'.
    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()

        config['initial_block'] = dict(layout='cnap', filters=96, kernel_size=7, strides=2,
                                       pool_size=3, pool_strides=2)
        config['body/layout'] = 'fffmffffmf'
        #config['body/layout'] = 'ffbfmbffbffmbf'

        num_blocks = config['body/layout'].count('f')
        layers_filters = 16 * np.arange(1, num_blocks//2 + num_blocks%2 + 1)
        layers_filters = np.repeat(layers_filters, 2)[:num_blocks].tolist()
        config['body/filters'] = layers_filters

        config['head'] = dict(layout='dcnaV', kernel_size=1, strides=1, dropout_rate=.5)

        config['loss'] = 'ce'

        return config

    def build_config(self, names=None):
        config = super().build_config(names)
        if config.get('head/units') is None:
            config['head/units'] = self.num_classes('targets')
        if config.get('head/filters') is None:
            config['head/filters'] = self.num_classes('targets')
        return config

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Create base VGG layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        layout : str
            a sequence of block types
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body', **kwargs)
        block = kwargs.pop('block', {})
        layout = kwargs.pop('layout')
        filters = kwargs.pop('filters')
        if isinstance(filters, int):
            filters = [filters] * layout.count('f')

        x = inputs
        bypass = None
        block_no = 0
        with tf.variable_scope(name):
            for i, b in enumerate(layout):
                if b == 'b':
                    bypass = x
                    continue
                elif b == 'f':
                    x = cls.fire_block(x, filters=filters[block_no], name='fire-block-%d' % i, **{**kwargs, **block})
                    block_no += 1
                elif b == 'm':
                    x = conv_block(x, 'p', name='max-pool-%d' % i, **kwargs)

                if bypass is not None:
                    bypass_channels = cls.num_channels(bypass, kwargs.get('data_format'))
                    x_channels = cls.num_channels(x, kwargs.get('data_format'))

                    if x_channels != bypass_channels:
                        bypass = conv_block(bypass, 'c', x_channels, 1, name='bypass-%d' % i, **kwargs)
                    x = x + bypass
                    bypass = None
        return x

    @classmethod
    def fire_block(cls, inputs, filters, layout='cna', name='fire-block', **kwargs):
        """ A sequence of 3x3 and 1x1 convolutions followed by pooling

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : int
            the number of filters in each convolution layer

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            x, inputs = inputs, None
            x = conv_block(x, layout, filters, 1, name='squeeze-1x1', **kwargs)

            exp1 = conv_block(x, layout, filters*4, 1, name='expand-1x1', **kwargs)
            exp3 = conv_block(x, layout, filters*4, 3, name='expand-3x3', **kwargs)

            axis = cls.channels_axis(kwargs.get('data_format'))
            x = tf.concat([exp1, exp3], axis=axis)
        return x
