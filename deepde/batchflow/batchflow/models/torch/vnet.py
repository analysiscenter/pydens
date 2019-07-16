"""  Milletari F. et al "`V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
<https://arxiv.org/abs/1606.04797>`_"
"""
import numpy as np
import torch
import torch.nn as nn

from .layers import ConvBlock
from . import TorchModel
from .utils import get_shape
from .resnet import ResNet


class VNet(TorchModel):
    """ VNet

    **Configuration**

    inputs : dict
        dict with 'images' and 'masks' (see :meth:`~.TFModel._make_inputs`)

    body : dict
        num_blocks : int
            number of downsampling blocks (default=5)

        filters : list of int
            number of filters in each block (default=[16, 32, 64, 128, 256])

    head : dict
        num_classes : int
            number of semantic classes
    """
    @classmethod
    def default_config(cls):
        config = TorchModel.default_config()

        filters = 16   # number of filters in the first block
        config['body/layout'] = ['cna', 'cna'*2] + ['cna'*3] * 3
        num_blocks = len(config['body/layout'])
        config['body/filters'] = (2 ** np.arange(num_blocks) * filters).tolist()
        config['body/kernel_size'] = 5
        config['body/upsample'] = dict(layout='tna', factor=2)
        config['head'] = dict(layout='c', kernel_size=1)

        config['loss'] = 'ce'

        return config

    def build_config(self, names=None):
        config = super().build_config(names)
        if config.get('head/num_classes') is None:
            config['head/num_classes'] = self.num_classes('targets')
        return config

    @classmethod
    def body(cls, inputs, **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs
            input tensor

        Returns
        -------
        nn.Module
        """
        kwargs = cls.get_defaults('body', kwargs)
        layout, filters = cls.pop(['layout', 'filters'], kwargs)

        x = inputs
        encoders = []
        for i, ifilters in enumerate(filters):
            x = cls.encoder_block(x, layout=layout[i], filters=ifilters, downsample=i > 0, **kwargs)
            encoders.append(x)

        decoders = []
        for i, ifilters in enumerate(filters[-2::-1]):
            x = cls.decoder_block(x, encoders[-i-2], layout=layout[-i-1], filters=ifilters*2, **kwargs)
            decoders.append(x)

        return VNetBody(encoders, decoders)

    @classmethod
    def encoder_block(cls, inputs, downsample=True, **kwargs):
        """ 5x5x5 convolutions and 2x2x2 max pooling with stride 2

        Parameters
        ----------
        inputs
            input tensor

        Returns
        -------
        nn.Module
        """
        kwargs = cls.get_defaults('body', kwargs)
        layout, kernel_size = cls.pop(['layout', 'kernel_size'], kwargs)

        x, inputs = inputs, None
        if downsample:
            x = ConvBlock(x, layout='cna', kernel_size=2, strides=2, **kwargs)
        x = ResNet.block(x, layout=layout, kernel_size=kernel_size, downsample=False, **kwargs)

        return x

    @classmethod
    def decoder_block(cls, inputs, skip, **kwargs):
        """ 2x2x2 transposed convolution + 5x5x5 convolutions

        Parameters
        ----------
        inputs
            input tensor
        skip
            skip connection

        Returns
        -------
        nn.Module
        """
        kwargs = cls.get_defaults('body', kwargs)
        layout, filters, kernel_size = cls.pop(['layout', 'filters', 'kernel_size'], kwargs)
        upsample_args = cls.pop('upsample', kwargs)

        x = cls.upsample(inputs, filters=filters, name='upsample', **upsample_args, **kwargs)
        x = cls.crop(x, skip, data_format=kwargs.get('data_format'))
        x = torch.cat([skip, x], dim=1)
        x = ResNet.block(x, layout=layout, filters=filters, kernel_size=kernel_size, downsample=0, **kwargs)

        return x

    @classmethod
    def head(cls, inputs, num_classes, **kwargs):
        """ 1x1 convolution

        Parameters
        ----------
        inputs
            input tensor
        num_classes : int
            number of classes (and number of filters in the last 1x1 convolution)

        Returns
        -------
        nn.Module
        """
        kwargs = cls.get_defaults('head', kwargs)
        x = ConvBlock(inputs, filters=num_classes, **kwargs)
        return x


class VNetBody(nn.Module):
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
