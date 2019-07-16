"""  Ronneberger O. et al "`U-Net: Convolutional Networks for Biomedical Image Segmentation
<https://arxiv.org/abs/1505.04597>`_"
"""
import tensorflow as tf
import numpy as np

from .layers import conv_block
from . import TFModel

class UNet(TFModel):
    """ UNet

    **Configuration**

    inputs : dict
        dict with 'images' and 'masks' (see :meth:`~.TFModel._make_inputs`)

    body : dict
        num_blocks : int
            number of downsampling/upsampling blocks (default=4)

        filters : list of int
            number of filters in each block (default=[128, 256, 512, 1024])

        downsample : dict
            parameters for downsampling block

        encoder : dict
            encoder block parameters

        upsample : dict
            parameters for upsampling block

        decoder : dict
            decoder block parameters

    head : dict
        num_classes : int
            number of semantic classes
    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()

        config['common'] = dict(conv=dict(use_bias=False))
        config['body/num_blocks'] = 5
        config['body/filters'] = (2 ** np.arange(config['body/num_blocks']) * 64).tolist()
        config['body/downsample'] = dict(layout='p', pool_size=2, pool_strides=2)
        config['body/encoder'] = dict(layout='cnacna', kernel_size=3)
        config['body/upsample'] = dict(layout='tna', kernel_size=2, strides=2)
        config['body/decoder'] = dict(layout='cnacna', kernel_size=3)
        config['head'] = dict(layout='c', kernel_size=1, strides=1)

        config['loss'] = 'ce'

        return config

    def build_config(self, names=None):
        config = super().build_config(names)

        if self.config.get('body/filters') is None:
            config['body/filters'] = (2 ** np.arange(config['body/num_blocks']) * 64).tolist()
        if config.get('head/num_classes') is None:
            config['head/num_classes'] = self.num_classes('targets')
        return config

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : tuple of int
            number of filters in encoder blocks
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body', **kwargs)
        filters = kwargs.pop('filters')
        downsample = kwargs.pop('downsample')
        encoder = kwargs.pop('encoder')
        upsample = kwargs.pop('upsample')
        decoder = kwargs.pop('decoder')

        with tf.variable_scope(name):
            x, inputs = inputs, None
            encoder_outputs = []
            for i, ifilters in enumerate(filters):
                down = downsample if i > 0 else None
                x = cls.encoder_block(x, ifilters, down, encoder, name='encoder-'+str(i), **kwargs)
                encoder_outputs.append(x)

            for i, ifilters in enumerate(filters[-2::-1]):
                x = cls.decoder_block((x, encoder_outputs[-i-2]), ifilters, upsample, decoder,
                                      name='decoder-'+str(i), **kwargs)

        return x

    @classmethod
    def encoder_block(cls, inputs, filters, downsample=None, encoder=None, name='encoder', **kwargs):
        """ 2x2 max pooling with stride 2 and two 3x3 convolutions

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : int
            number of output filters
        downsample : dict or None
            parameters for downsampling block (skipped if None)
        encoder : dict
            parameters for encoding block
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        encoder = cls.fill_params('body/encoder', **encoder)
        with tf.variable_scope(name):
            if downsample:
                downsample = cls.fill_params('body/downsample', **downsample)
                inputs = conv_block(inputs, filters=filters, name='downsample', **{**kwargs, **downsample})
            x = conv_block(inputs, filters=filters, name='encoder', **{**kwargs, **encoder})
        return x

    @classmethod
    def decoder_block(cls, inputs, filters, upsample=None, decoder=None, name='decoder', **kwargs):
        """ 3x3 convolution and 2x2 transposed convolution

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : int
            number of output filters
        upsample : dict
            parameters for upsampling block
        decoder : dict
            parameters for decoding block
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        upsample = cls.fill_params('body/upsample', **upsample)
        decoder = cls.fill_params('body/decoder', **decoder)

        with tf.variable_scope(name):
            x, skip = inputs
            inputs = None
            x = cls.upsample(x, filters=filters, name='upsample', **{**kwargs, **upsample})
            x = cls.crop(x, skip, data_format=kwargs.get('data_format'))
            axis = cls.channels_axis(kwargs.get('data_format'))
            x = tf.concat((skip, x), axis=axis)
            x = conv_block(x, filters=filters, name='conv', **{**kwargs, **decoder})
        return x

    @classmethod
    def head(cls, inputs, num_classes, name='head', **kwargs):
        """ Conv block followed by 1x1 convolution

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        num_classes : int
            number of classes (and number of filters in the last 1x1 convolution)
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('head', **kwargs)
        x = conv_block(inputs, filters=num_classes, units=num_classes, name=name, **kwargs)
        return x
