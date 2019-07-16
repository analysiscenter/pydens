""" Iandola F. et al. "`SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size"
<https://arxiv.org/abs/1602.07360>`_"
"""
import numpy as np
import torch
import torch.nn as nn

from . import TorchModel
from .layers import ConvBlock
from .utils import get_num_channels, get_shape


class SqueezeNet(TorchModel):
    """ SqueezeNet neural network

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'labels' (see :meth:`~.TorchModel._make_inputs`)

    body : dict
        layout : str
            A sequence of blocks:

            - f : fire block
            - m : max-pooling
            - b : bypass

            Default is 'fffmffffmf'.

        block : dict
            FireBlock parameters
    """
    @classmethod
    def default_config(cls):
        config = TorchModel.default_config()

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
    def body(cls, **kwargs):
        """ Create base blocks

        Parameters
        ----------
        layout : str
            a sequence of block types

        Returns
        -------
        nn.Module
        """
        kwargs = cls.get_defaults('body', kwargs)
        return SqueezeNetBody(**kwargs)


class SqueezeNetBody(nn.Module):
    """ A sequence of fire and pooling blocks

        Parameters
        ----------
        layout : str
            A sequence of blocks:

            - f : fire block
            - m : max-pooling
            - b : bypass

        filters : list of int
            The number of output filters for each fire block.

        block : dict
            FireBlock parameters
    """
    def __init__(self, layout, filters, inputs=None, **kwargs):
        super().__init__()

        self.layout = layout

        if isinstance(filters, int):
            filters = [filters] * layout.count('f')
        block = kwargs.pop('block', {})

        x = inputs
        bypass = None
        block_no = 0
        for i, b in enumerate(self.layout):
            if b == 'b':
                bypass = x
                continue
            elif b == 'f':
                x = FireBlock(x, filters=filters[block_no], **{**kwargs, **block})
                block_no += 1
                self.add_module('fire%d' % i, x)
            elif b == 'm':
                x = ConvBlock(x, 'p', **kwargs)
                self.add_module('pool%d' % i, x)

            if bypass is not None:
                bypass_channels = get_num_channels(bypass)
                x_channels = get_num_channels(x)

                if x_channels != bypass_channels:
                    bypass = ConvBlock(bypass, 'c', x_channels, kernel_size=1, **kwargs)
                else:
                    bypass = None
                self.add_module('bypass%d' % (i-1), bypass)
                bypass = None

        self.output_shape = get_shape(x)

    def forward(self, x):
        """ Make forward pass """
        bypass = None
        for i, block in enumerate(self.layout):
            if block == 'b':
                bypass = x
                continue
            elif block == 'f':
                x = getattr(self, 'fire%d' % i)(x)
            elif block == 'm':
                x = getattr(self, 'pool%d' % i)(x)

            if bypass is not None:
                bypass_block = getattr(self, 'bypass%d' % (i-1))
                if bypass_block is not None:
                    bypass = bypass_block(bypass)
                x = x + bypass
                bypass = None
        return x


class FireBlock(nn.Module):
    """ 1x1 convolutions followed by expanding branches with 3x3 and 1x1 convolutions

    Parameters
    ----------
    layout : str
        A sequence of layers (default is 'cna')
    filters : int
        the number of filters in the convolution layer

    Notes
    -----
    For other params see :class:`.ConvBlock`.
    """
    def __init__(self, inputs, layout='cna', filters=None, **kwargs):
        super().__init__()
        self.entry = ConvBlock(inputs, layout, filters, kernel_size=1, **kwargs)
        self.exp1 = ConvBlock(self.entry, layout, filters*4, kernel_size=1, **kwargs)
        self.exp3 = ConvBlock(self.entry, layout, filters*4, kernel_size=3, **kwargs)

        self.output_shape = list(get_shape(self.exp3))
        self.output_shape[1] = self.output_shape[1] * 2
        self.output_shape = tuple(self.output_shape)

    def forward(self, x):
        """ Make forward pass """
        x = self.entry(x)
        x = torch.cat([self.exp1(x), self.exp3(x)], dim=1)
        return x
