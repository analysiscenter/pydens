""" Simonyan K., Zisserman A. "`Very Deep Convolutional Networks for Large-Scale Image Recognition
<https://arxiv.org/abs/1409.1556>`_"
"""
import torch.nn as nn

from . import TorchModel
from .layers import ConvBlock


_VGG16_ARCH = [
    (2, 0, 64, 1),
    (2, 0, 128, 1),
    (3, 0, 256, 1),
    (3, 0, 512, 1),
    (3, 0, 512, 1)
]

_VGG19_ARCH = [
    (2, 0, 64, 1),
    (2, 0, 128, 1),
    (4, 0, 256, 1),
    (4, 0, 512, 1),
    (4, 0, 512, 1)
]

_VGG7_ARCH = [
    (2, 0, 64, 1),
    (2, 0, 128, 1),
    (2, 1, 256, 1)
]


class VGG(TorchModel):
    """ Base VGG neural network

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'labels' (see :meth:`~.TorchModel._make_inputs`)

    body/arch : list of tuple of int
        Each list item contains parameters for one network block as a tuple of 4 ints:

        - number of convolution layers with 3x3 kernel
        - number of convolution layers with 1x1 kernel
        - number of filters in each layer
        - whether to downscale the image at the end of the block with max_pooling (2x2, stride=2)
    """
    @classmethod
    def default_config(cls):
        config = TorchModel.default_config()
        config['common/conv/bias'] = False
        config['body/block'] = dict(layout='cna', pool_size=2, pool_strides=2)
        config['head'] = dict(layout='Vdf', dropout_rate=.8, units=2)

        config['loss'] = 'ce'

        return config

    def build_config(self, names=None):
        config = super().build_config(names)
        if isinstance(config['head/units'], list):
            config['head/units'][-1] = self.num_classes('targets')
        else:
            config['head/units'] = self.num_classes('targets')
        return config

    @classmethod
    def body(cls, inputs, **kwargs):
        """ Create base VGG layers

        Parameters
        ----------
        arch : list of tuples
            number of 3x3 conv, number of 1x1 conv, number of filters, whether to downscale

        Returns
        -------
        nn.Module
        """
        kwargs = cls.get_defaults('body', kwargs)
        arch = kwargs.pop('arch')
        if not isinstance(arch, (list, tuple)):
            raise TypeError("arch must be list or tuple, but {} was given.".format(type(arch)))
        block = kwargs.pop('block')
        block = {**block, **kwargs}

        blocks = []
        x = inputs
        for block_cfg in arch:
            x = cls.block(x, *block_cfg, **block)
            blocks.append(x)
        x = nn.Sequential(*blocks)
        return x

    @classmethod
    def block(cls, inputs, depth3, depth1, filters, downscale, **kwargs):
        """ A sequence of 3x3 and 1x1 convolutions followed by pooling

        Parameters
        ----------
        depth3 : int
            the number of convolution layers with 3x3 kernel
        depth1 : int
            the number of convolution layers with 1x1 kernel
        filters : int
            the number of filters in each convolution layer
        downscale : bool
            whether to decrease spatial dimension at the end of the block

        Returns
        -------
        nn.Module
        """
        kwargs = cls.get_defaults('body/block', kwargs)
        layout = kwargs.pop('layout') * (depth3 + depth1) + 'p' * downscale
        kernels = [3] * depth3 + [1] * depth1
        x = ConvBlock(inputs, layout, filters, kernels, **kwargs)
        return x


class VGG16(VGG):
    """ VGG16 network """
    @classmethod
    def default_config(cls):
        config = VGG.default_config()
        config['body/arch'] = _VGG16_ARCH
        return config


class VGG19(VGG):
    """ VGG19 network """
    @classmethod
    def default_config(cls):
        config = VGG.default_config()
        config['body/arch'] = _VGG19_ARCH
        return config


class VGG7(VGG):
    """ VGG7 network """
    @classmethod
    def default_config(cls):
        config = VGG.default_config()
        config['body/arch'] = _VGG7_ARCH
        return config
