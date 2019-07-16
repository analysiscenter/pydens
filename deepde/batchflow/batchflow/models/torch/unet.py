"""  Ronneberger O. et al "`U-Net: Convolutional Networks for Biomedical Image Segmentation
<https://arxiv.org/abs/1505.04597>`_"
"""
import numpy as np
import torch
import torch.nn as nn

from .layers import ConvBlock
from . import TorchModel
from .utils import get_shape


class UNet(TorchModel):
    """ UNet

    **Configuration**

    inputs : dict
        dict with 'images' and 'masks' (see :meth:`~.TorchModel._make_inputs`)

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
        config = TorchModel.default_config()

        config['common'] = {'conv/bias': False}
        config['body/num_blocks'] = 5
        config['body/filters'] = (2 ** np.arange(config['body/num_blocks']) * 64).tolist()
        config['body/downsample'] = dict(layout='p', pool_size=2, pool_strides=2)
        config['body/encoder'] = dict(layout='cna cna', kernel_size=3)
        config['body/upsample'] = dict(layout='tna', kernel_size=2, strides=2)
        config['body/decoder'] = dict(layout='cna cna', kernel_size=3)
        config['head'] = dict(layout='c', kernel_size=1)

        config['loss'] = 'ce'

        return config

    def build_config(self, names=None):
        config = super().build_config(names)

        if self.config.get('body/filters') is None:
            config['body/filters'] = (2 ** np.arange(config['body/num_blocks']) * 64).tolist()
        if config.get('head/num_classes') is None:
            config['head/num_classes'] = self.num_classes('targets')

        return config

    def body(self, inputs=None, **kwargs):
        """ A sequence of encoder and decoder blocks with skip connections

        Parameters
        ----------
        filters : tuple of int
            number of filters in encoder blocks

        Returns
        -------
        nn.Module
        """
        kwargs = self.get_defaults('body', kwargs)
        filters = kwargs.pop('filters')
        downsample = kwargs.pop('downsample')
        encoder = kwargs.pop('encoder')
        upsample = kwargs.pop('upsample')
        decoder = kwargs.pop('decoder')

        encoders = []
        x = inputs
        for i, ifilters in enumerate(filters):
            down = downsample if i > 0 else None
            x = self.encoder_block(x, ifilters, down, encoder, **kwargs)
            encoders.append(x)

        decoders = []
        for i, ifilters in enumerate(filters[-2::-1]):
            skip = encoders[-i-2]
            x = self.decoder_block(x, skip, ifilters, upsample, decoder, **kwargs)
            decoders.append(x)

        return UNetBody(encoders, decoders)

    @classmethod
    def encoder_block(cls, inputs, filters, downsample=None, encoder=None, **kwargs):
        """ 2x2 max pooling with stride 2 and two 3x3 convolutions

        Parameters
        ----------
        inputs
            input tensor or previous layer
        filters : int
            number of output filters
        downsample : dict
            parameters for downsampling blocks
        encoder : dict
            parameters for encoder blocks

        Returns
        -------
        nn.Module
        """
        if downsample:
            downsample = cls.get_defaults('body/downsample', downsample)
            down_block = ConvBlock(inputs, filters=filters, **{**kwargs, **downsample})
            inputs = down_block
        encoder = cls.get_defaults('body/encoder', encoder)
        enc_block = ConvBlock(inputs, filters=filters, **{**kwargs, **encoder})
        return nn.Sequential(down_block, enc_block) if downsample else enc_block

    @classmethod
    def decoder_block(cls, inputs, skip, filters, upsample=None, decoder=None, **kwargs):
        """ Takes inputs from a previous block and a skip connection

        Parameters
        ----------
        inputs
            input tensor or previous layer
        skip
            skip connection tensor or layer
        filters : int
            number of output filters
        upsample : dict
            parameters for upsample layers
        decoder : dict
            parameters for decoder kayers
        inputs
            previous block or tensor
        skip
            skip connection block or tensor
        kwargs
            common parameters for layers

        Returns
        -------
        nn.Module
        """
        upsample = cls.get_defaults('body/upsample', upsample)
        decoder = cls.get_defaults('body/decoder', decoder)
        return DecoderBlock(inputs, skip, filters, upsample, decoder, **kwargs)

    @classmethod
    def head(cls, inputs, num_classes, **kwargs):
        """ Conv block with 1x1 convolution

        Parameters
        ----------
        num_classes : int
            number of classes (and number of filters in the last 1x1 convolution)

        Returns
        -------
        nn.Module
        """
        kwargs = cls.get_defaults('head', kwargs)
        return ConvBlock(inputs, filters=num_classes, **kwargs)


class UNetBody(nn.Module):
    """ A sequence of encoder and decoder blocks with skip connections """
    def __init__(self, encoders, decoders):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.output_shape = self.decoders[-1].output_shape

    def forward(self, x):
        skip = []
        for encoder in self.encoders:
            x = encoder(x)
            skip.append(x)

        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip=skip[-i-2])

        return x

class DecoderBlock(nn.Module):
    """ An upsampling block aggregating a skip connection

        Parameters
        ----------
        filters : int
            number of output filters
        upsample : dict
            parameters for upsample layers
        decoder : dict
            parameters for decoder kayers
        inputs
            previous block or tensor
        skip
            skip connection block or tensor
        kwargs
            common parameters for layers
    """
    def __init__(self, inputs, skip, filters, upsample, decoder, **kwargs):
        super().__init__()
        _ = skip
        self.upsample = ConvBlock(inputs, filters=filters, **{**kwargs, **upsample})
        shape = list(get_shape(self.upsample))
        shape[1] *= 2
        shape = tuple(shape)
        self.decoder = ConvBlock(shape, filters=filters, **{**kwargs, **decoder})
        self.output_shape = self.decoder.output_shape

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.size() > skip.size():
            shape = [slice(None, c) for c in skip.size()[2:]]
            shape = tuple([slice(None, None), slice(None, None)] + shape)
            x = x[shape]

        x = torch.cat([skip, x], dim=1)
        x = self.decoder(x)
        return x
