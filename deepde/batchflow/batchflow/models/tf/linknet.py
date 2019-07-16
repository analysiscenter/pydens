""" Chaurasia A., Culurciello E. "`LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation
<https://arxiv.org/abs/1707.03718>`_"

NOTE: The fourth section of the article describes the method of weighing imbalance datasets,
      it does not exist in this implementation.

"""
import tensorflow as tf
import numpy as np

from .layers import conv_block
from . import TFModel
from .resnet import ResNet


class LinkNet(TFModel):
    """ LinkNet

    **Configuration**

    inputs : dict
        dict with 'images' and 'masks' (see :meth:`~.TFModel._make_inputs`)

    body : dict
        num_blocks : int
            number of downsampling/upsampling blocks (default=4)

        filters : list of int
            number of filters in each block (default=[64, 128, 256, 512])

    head : dict
        num_classes : int
            number of semantic classes
    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()

        filters = 64   # number of filters in the first block

        config['initial_block'] = dict(layout='cnap', filters=filters, kernel_size=7, strides=2,
                                       pool_size=3, pool_strides=2)
        config['body/num_blocks'] = 4
        config['body/upsample'] = dict(layout='tna', factor=2, kernel_size=3)

        config['head/filters'] = filters // 2
        config['head/upsample1'] = dict(layout='tna cna', factor=2, kernel_size=3, strides=[2, 1])
        config['head/upsample2'] = dict(layout='t', factor=2)

        config['loss'] = 'ce'

        return config

    def build_config(self, names=None):
        config = super().build_config(names)
        if config.get('body/filters') is None:
            config['body/filters'] = 2 ** np.arange(config['body/num_blocks']) * config['initial_block/filters']
        if config.get('head/num_classes') is None:
            config['head/num_classes'] = self.num_classes('targets')
        config['head/targets'] = self.get_from_attr('targets')
        return config

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ LinkNet body

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
        kwargs = cls.fill_params('body', **kwargs)
        filters = kwargs.pop('filters')

        with tf.variable_scope(name):
            x = inputs
            encoder_outputs = []
            for i, ifilters in enumerate(filters):
                x = cls.encoder_block(x, filters=ifilters, name='encoder-'+str(i), **kwargs)
                encoder_outputs.append(x)

            for i, ifilters in enumerate(filters[::-1][1:]):
                x = cls.decoder_block(x, filters=ifilters, name='decoder-'+str(i), **kwargs)
                x = cls.crop(x, encoder_outputs[-i-2], data_format=kwargs.get('data_format'))
                x = tf.add(x, encoder_outputs[-2-i])
            x = cls.decoder_block(x, filters[0], 'decoder-'+str(i+1), **kwargs)
            x = cls.crop(x, inputs, data_format=kwargs.get('data_format'))
        return x

    @classmethod
    def encoder_block(cls, inputs, filters, name, **kwargs):
        """ Two ResNet blocks of two 3x3 convolution + shortcut

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : int
            number of output filters
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        return ResNet.double_block(inputs, filters=filters, name=name, downsample=True, **kwargs)

    @classmethod
    def decoder_block(cls, inputs, filters, name, **kwargs):
        """ 1x1 convolution, 3x3 transposed convolution with stride=2 and 1x1 convolution

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : int
            number of output filters
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        upsample_args = cls.pop('upsample', kwargs)

        num_filters = cls.num_channels(inputs, kwargs.get('data_format')) // 4
        with tf.variable_scope(name):
            x, inputs = inputs, None
            x = conv_block(x, 'cna', num_filters, kernel_size=1, name='conv_pre', **kwargs)
            x = cls.upsample(x, filters=num_filters, name='upsample', **upsample_args, **kwargs)
            x = conv_block(x, 'cna', filters, kernel_size=1, name='conv_post', **kwargs)
        return x

    @classmethod
    def head(cls, inputs, targets, num_classes, name='head', **kwargs):
        """ 3x3 transposed convolution, 3x3 convolution and 2x2 transposed convolution

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        targets : tf.Tensor
            target tensor
        num_classes : int
            number of classes (and number of filters in the last convolution)
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('head', **kwargs)
        filters = kwargs.pop('filters')
        upsample1_args = cls.pop('upsample1', kwargs)
        upsample2_args = cls.pop('upsample2', kwargs)

        with tf.variable_scope(name):
            x, inputs = inputs, None
            x = cls.upsample(x, filters=filters, name='upsample1', **upsample1_args, **kwargs)
            x = cls.upsample(x, filters=num_classes, name='upsample2', **upsample2_args, **kwargs)
            x = cls.crop(x, targets, data_format=kwargs.get('data_format'))
        return x
