"""
Kaiming He et al. "`Deep Residual Learning for Image Recognition
<https://arxiv.org/abs/1512.03385>`_"

Kaiming He et al. "`Identity Mappings in Deep Residual Networks
<https://arxiv.org/abs/1603.05027>`_"

Sergey Zagoruyko, Nikos Komodakis. "`Wide Residual Networks
<https://arxiv.org/abs/1605.07146>`_"

Xie S. et al. "`Aggregated Residual Transformations for Deep Neural Networks
<https://arxiv.org/abs/1611.05431>`_"
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvBlock
from . import TorchModel
from .utils import get_shape, get_num_dims, get_num_channels


class ResNet(TorchModel):
    """ The base ResNet model

    Notes
    -----
    This class is intended to define custom ResNets.
    For more convenience use predefined :class:`~.torch.ResNet18`, :class:`~.torch.ResNet34`,
    and others described down below.

    **Configuration**

    inputs : dict
        dict with 'images' and 'labels' (see :meth:`~.TorchModel._make_inputs`)

    initial_block : dict
        filters : int
            number of filters (default=64)

    body : dict
        num_blocks : list of int
            number of blocks in each group with the same number of filters.
        filters : list of int
            number of filters in each group

        block : dict
            bottleneck : bool
                whether to use bottleneck blocks (1x1,3x3,1x1) or simple (3x3,3x3)
            bottleneck_factor : int
                filter shrinking factor in a bottleneck block (default=4)
            post_activation : None or bool or str
                layout to apply after after residual and shortcut summation (default is None)
            zero_pad : bool
                whether to pad a shortcut with zeros when a number of filters increases
                or apply a 1x1 convolution (default is False)
            width_factor : int
                widening factor to make WideResNet (default=1)
            se_block : bool
                whether to use squeeze-and-excitation blocks (default=False)
            se_factor : int
                squeeze-and-excitation channels ratio (default=16)
            resnext : bool
                whether to use aggregated ResNeXt block (default=False)
            resnext_factor : int
                the number of aggregations in ResNeXt block (default=32)

    head : dict
        'Vdf' with dropout_rate=.4
    """
    @classmethod
    def default_config(cls):
        config = TorchModel.default_config()
        config['common/conv/bias'] = False
        config['initial_block'] = dict(layout='cnap', filters=64, kernel_size=7, strides=2,
                                       pool_size=3, pool_strides=2)

        config['body/block'] = dict(layout=None, post_activation=None, downsample=False,
                                    bottleneck=False, bottleneck_factor=4,
                                    width_factor=1, zero_pad=False,
                                    resnext=False, resnext_factor=32,
                                    se_block=False, se_factor=16)

        config['head'] = dict(layout='Vdf', dropout_rate=.4)

        config['loss'] = 'ce'

        return config

    @classmethod
    def default_layout(cls, bottleneck, **kwargs):
        """ Define conv block layout """
        _ = kwargs
        reps = 3 if bottleneck else 2
        return 'cna' * reps

    def build_config(self, names=None):
        config = super().build_config(names)

        if config.get('body/filters') is None:
            width = config['body/block/width_factor']
            num_blocks = config['body/num_blocks']
            filters = config['initial_block/filters']
            config['body/filters'] = (2 ** np.arange(len(num_blocks)) * filters * width).tolist()

        if config.get('head/units') is None:
            config['head/units'] = self.num_classes('targets')
        if config.get('head/filters') is None:
            config['head/filters'] = self.num_classes('targets')

        return config

    @classmethod
    def body(cls, inputs=None, **kwargs):
        """ Base layers

        Parameters
        ----------
        filters : list of int or list of list of int
            number of filters in each block group
        num_blocks : list of int
            number of blocks in each group
        bottleneck : bool
            whether to use a simple or bottleneck block
        bottleneck_factor : int
            filter number multiplier for a bottleneck block

        Returns
        -------
        nn.Module
        """
        kwargs = cls.get_defaults('body', kwargs)
        filters, block_args = cls.pop(['filters', 'block'], kwargs)
        block_args = {**kwargs, **block_args}

        x = inputs
        all_blocks = []
        for i, n_blocks in enumerate(kwargs['num_blocks']):
            for block in range(n_blocks):
                downsample = i > 0 and block == 0
                block_args['downsample'] = downsample
                bfilters = filters[i] if isinstance(filters[i], int) else filters[i][block]
                x = cls.block(x, filters=bfilters, **block_args)
                all_blocks.append(x)
        return nn.Sequential(*all_blocks)

    @classmethod
    def double_block(cls, inputs, downsample=True, **kwargs):
        """ Two ResNet blocks one after another

        Parameters
        ----------
        inputs
            input tensor
        downsample : bool
            whether to decrease spatial dimensions
        kwargs : dict
            block parameters

        Returns
        -------
        nn.Module
        """
        first = cls.block(inputs, downsample=downsample, **kwargs)
        second = cls.block(first, downsample=False, **kwargs)
        return nn.Sequential(first, second)

    @classmethod
    def block(cls, inputs, **kwargs):
        """ A network building block

        Parameters
        ----------
        inputs
            input tensor
        layout : str
            a sequence of layers in the block
        filters : int or list/tuple of ints
            number of output filters
        zero_pad : bool
            whether to pad shortcut when a number of filters increases
        width_factor : int
            a factor to increase number of filters to build a wide network
        downsample : bool
            whether to decrease spatial dimensions with strides=2 in the first convolution
        resnext : bool
            whether to use an aggregated ResNeXt block
        resnext_factor : int
            cardinality for ResNeXt block
        bottleneck : bool
            whether to use a simple (`False`) or bottleneck (`True`) block
        bottleneck_factor : int
            the filters nultiplier in the bottleneck block
        se_block : bool
            whether to include squeeze and excitation block
        se_factor : int
            se block ratio
        kwargs : dict
            ConvBlock parameters for all sub blocks

        Returns
        -------
        nn.Module
        """
        kwargs = cls.get_defaults('body/block', kwargs)
        layout, filters, downsample, zero_pad = cls.pop(['layout', 'filters', 'downsample', 'zero_pad'], kwargs)
        width_factor = cls.pop('width_factor', kwargs)
        bottleneck, bottleneck_factor = cls.pop(['bottleneck', 'bottleneck_factor'], kwargs)
        resnext, resnext_factor = cls.pop(['resnext', 'resnext_factor'], kwargs)
        #se_block, se_factor = cls.pop(['se_block', 'se_factor'], kwargs)
        post_activation = cls.pop('post_activation', kwargs)
        if isinstance(post_activation, bool) and post_activation:
            post_activation = 'an'

        filters = filters * width_factor
        if resnext:
            x = cls.next_conv_block(inputs, layout, filters, bottleneck, resnext_factor,
                                    downsample=downsample, bottleneck_factor=bottleneck_factor, **kwargs)
        else:
            x = cls.conv_block(inputs, layout, filters, bottleneck, bottleneck_factor,
                               downsample=downsample, **kwargs)

        inputs_channels = get_num_channels(inputs)
        x_channels = get_num_channels(x)

        strides = 2 if downsample else 1
        padding = None
        shortcut = None
        if zero_pad and inputs_channels < x_channels:
            if downsample:
                shortcut = ConvBlock(inputs,
                                     **{**kwargs, **dict(layout='c', filters=inputs_channels,
                                                         kernel_size=1, strides=strides)})
            if inputs_channels < x_channels:
                padding = [(0, 0) for _ in range(get_num_dims(inputs))]
                padding[-1] = (0, x_channels - inputs_channels)
                padding = sum(tuple(padding), ())
        elif inputs_channels != x_channels or downsample:
            shortcut = ConvBlock(inputs,
                                 **{**kwargs, **dict(layout='c', filters=x_channels,
                                                     kernel_size=1, strides=strides)})

        x = ResBlock(x, shortcut, padding)

        if post_activation:
            p = ConvBlock(x, layout=post_activation, **kwargs)
            x = nn.Sequential(x, p)

        return x

    @classmethod
    def conv_block(cls, inputs, layout, filters, bottleneck, bottleneck_factor=None, **kwargs):
        """ ResNet convolution block

        Parameters
        ----------
        inputs
            input tensor
        layout : str
            a sequence of layers in the block
        filters : int
            number of output filters
        downsample : bool
            whether to decrease spatial dimensions with strides=2
        bottleneck : bool
            whether to use a simple or a bottleneck block
        bottleneck_factor : int
            filter count scaling factor
        kwargs : dict
            conv_block parameters

        Returns
        -------
        nn.Module
        """
        if layout is None:
            layout = cls.default_layout(bottleneck=bottleneck, filters=filters, **kwargs)
        if bottleneck:
            x = cls.bottleneck_block(inputs, layout, filters, bottleneck_factor=bottleneck_factor, **kwargs)
        else:
            x = cls.simple_block(inputs, layout, filters, **kwargs)
        return x

    @classmethod
    def simple_block(cls, inputs, layout=None, filters=None, kernel_size=3, downsample=False, **kwargs):
        """ A simple residual block (by default with two 3x3 convolutions)

        Parameters
        ----------
        inputs
            input tensor
        layout : str
            a sequence of layers in the block
        filters : int
            number of filters
        kernel_size : int or tuple
            convolution kernel size
        downsample : bool
            whether to decrease spatial dimensions with strides=2
        kwargs : dict
            ConvBlock parameters

        Returns
        -------
        nn.Module
        """
        n = layout.count('c') + layout.count('C') - 1
        strides = ([2] + [1] * n) if downsample else 1
        return ConvBlock(inputs, layout, filters=filters, kernel_size=kernel_size, strides=strides, **kwargs)

    @classmethod
    def bottleneck_block(cls, inputs, layout=None, filters=None, kernel_size=None, bottleneck_factor=4,
                         downsample=False, **kwargs):
        """ A stack of 1x1, 3x3, 1x1 convolutions

        Parameters
        ----------
        inputs
            input tensor
        layout : str
            a sequence of layers in the block
        filters : int or list of int
            if int, number of filters in the first convolution.
            if list, number of filters in each layer
        kernel_size : int or tuple
            convolution kernel size
        bottleneck_factor : int
            scale factor for the number of filters in the last convolution
            (if filters is int, otherwise is not used)
        downsample : bool
            whether to decrease spatial dimensions with strides=2
        kwargs : dict
            ConvBlock parameters

        Returns
        -------
        nn.Module
        """
        kernel_size = [1, 3, 1] if kernel_size is None else kernel_size
        n = layout.count('c') + layout.count('C') - 2
        if kwargs.get('strides') is None:
            strides = ([1, 2] + [1] * n) if downsample else 1
        else:
            strides = kwargs.pop('strides')
        if isinstance(filters, int):
            filters = [filters, filters, filters * bottleneck_factor]
        x = ConvBlock(inputs, layout, filters=filters, kernel_size=kernel_size, strides=strides, **kwargs)
        return x

    @classmethod
    def next_conv_block(cls, inputs, layout, filters, bottleneck, resnext_factor, **kwargs):
        """ ResNeXt convolution block

        Parameters
        ----------
        inputs
            input tensor
        filters : int
            number of output filters
        bottleneck : bool
            whether to use a simple or a bottleneck block
        resnext_factor : int
            cardinality for ResNeXt model
        kwargs : dict
            `ConvBlock` parameters

        Returns
        -------
        nn.Module
        """
        if bottleneck:
            filters = filters * kwargs['bottleneck_factor']
            filters = [filters // 2, filters // 2, filters]
            kwargs['groups'] = [1, resnext_factor, 1]
        else:
            filters = [resnext_factor * 4, filters]
        x = cls.conv_block(inputs, layout, filters=filters, bottleneck=bottleneck, **kwargs)
        return x

    @classmethod
    def make_encoder(cls, inputs, **kwargs):
        """ Build the body and return encoder tensors

        Parameters
        ----------
        inputs
            input tensor
        kwargs : dict
            body params
        """
        pass

class ResBlock(nn.Module):
    """ Add modules """
    def __init__(self, conv, shortcut, padding=None):
        super().__init__()
        self.conv = conv
        self.shortcut = shortcut
        self.padding = padding
        self.output_shape = get_shape(conv)

    def forward(self, x):
        """ Make a forward pass """
        shortcut = self.shortcut(x) if self.shortcut else x
        if self.padding:
            shortcut = F.pad(shortcut, self.padding)
        return self.conv(x) + shortcut


class ResNet18(ResNet):
    """ The original ResNet-18 architecture """
    @classmethod
    def default_config(cls):
        config = ResNet.default_config()
        config['body/num_blocks'] = [2, 2, 2, 2]
        config['body/block/bottleneck'] = False
        return config


class ResNet34(ResNet):
    """ The original ResNet-34 architecture """
    @classmethod
    def default_config(cls):
        config = ResNet.default_config()
        config['body/num_blocks'] = [3, 4, 6, 3]
        config['body/block/bottleneck'] = False
        return config


class ResNet50(ResNet):
    """ The original ResNet-50 architecture """
    @classmethod
    def default_config(cls):
        config = ResNet34.default_config()
        config['body/block/bottleneck'] = True
        return config


class ResNet101(ResNet):
    """ The original ResNet-101 architecture """
    @classmethod
    def default_config(cls):
        config = ResNet.default_config()
        config['body/num_blocks'] = [3, 4, 23, 3]
        config['body/block/bottleneck'] = True
        return config


class ResNet152(ResNet):
    """ The original ResNet-152 architecture """
    @classmethod
    def default_config(cls):
        config = ResNet.default_config()
        config['body/num_blocks'] = [3, 8, 36, 3]
        config['body/block/bottleneck'] = True
        return config

class ResNeXt18(ResNet):
    """ The ResNeXt-18 architecture """
    @classmethod
    def default_config(cls):
        config = ResNet18.default_config()
        config['body/block/resnext'] = True
        return config


class ResNeXt34(ResNet):
    """ The ResNeXt-34 architecture """
    @classmethod
    def default_config(cls):
        config = ResNet34.default_config()
        config['body/block/resnext'] = True
        return config


class ResNeXt50(ResNet):
    """ The ResNeXt-50 architecture """
    @classmethod
    def default_config(cls):
        config = ResNet50.default_config()
        config['body/block/resnext'] = True
        return config


class ResNeXt101(ResNet):
    """ The ResNeXt-101 architecture """
    @classmethod
    def default_config(cls):
        config = ResNet101.default_config()
        config['body/block/resnext'] = True
        return config


class ResNeXt152(ResNet):
    """ The ResNeXt-152 architecture """
    @classmethod
    def default_config(cls):
        config = ResNet152.default_config()
        config['body/block/resnext'] = True
        return config
