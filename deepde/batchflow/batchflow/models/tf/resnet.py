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
import tensorflow as tf

from . import TFModel
from .layers import conv_block


class ResNet(TFModel):
    """ The base ResNet model

    Notes
    -----
    This class is intended to define custom ResNets.
    For more convenience use predefined :class:`~.tf.ResNet18`, :class:`~tf..ResNet34`,
    and others described down below.

    **Configuration**

    inputs : dict
        dict with 'images' and 'labels' (see :meth:`~.TFModel._make_inputs`)

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
        config = TFModel.default_config()
        config['common/conv/use_bias'] = False
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
    def body(cls, inputs, name='body', **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : list of int or list of list of int
            number of filters in each block group
        num_blocks : list of int
            number of blocks in each group
        bottleneck : bool
            whether to use a simple or bottleneck block
        bottleneck_factor : int
            filter number multiplier for a bottleneck block
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body', **kwargs)
        filters, block_args = cls.pop(['filters', 'block'], kwargs)
        block_args = {**kwargs, **block_args}

        with tf.variable_scope(name):
            x, inputs = inputs, None
            for i, n_blocks in enumerate(kwargs['num_blocks']):
                with tf.variable_scope('group-%d' % i):
                    for block in range(n_blocks):
                        downsample = i > 0 and block == 0
                        block_args['downsample'] = downsample
                        bfilters = filters[i] if isinstance(filters[i], int) else filters[i][block]
                        x = cls.block(x, filters=bfilters, name='block-%d' % block, **block_args)
                    x = tf.identity(x, name='output')
        return x

    @classmethod
    def double_block(cls, inputs, name='double_block', downsample=True, **kwargs):
        """ Two ResNet blocks one after another

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        downsample : bool
            whether to decrease spatial dimensions
        name : str
            scope name
        kwargs : dict
            block parameters

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            x = cls.block(inputs, name='block-1', downsample=downsample, **kwargs)
            x = cls.block(x, name='block-2', downsample=False, **kwargs)
        return x

    @classmethod
    def block(cls, inputs, name='block', **kwargs):
        """ A network building block

        Parameters
        ----------
        inputs : tf.Tensor
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
        name : str
            scope name
        kwargs : dict
            conv_block parameters for all sub blocks

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body/block', **kwargs)
        layout, filters, downsample, zero_pad = cls.pop(['layout', 'filters', 'downsample', 'zero_pad'], kwargs)
        width_factor = cls.pop('width_factor', kwargs)
        bottleneck, bottleneck_factor = cls.pop(['bottleneck', 'bottleneck_factor'], kwargs)
        resnext, resnext_factor = cls.pop(['resnext', 'resnext_factor'], kwargs)
        se_block, se_factor = cls.pop(['se_block', 'se_factor'], kwargs)
        post_activation = cls.pop('post_activation', kwargs)
        if isinstance(post_activation, bool) and post_activation:
            post_activation = 'an'

        with tf.variable_scope(name):
            filters = filters * width_factor
            if resnext:
                x = cls.next_conv_block(inputs, layout, filters, bottleneck, resnext_factor, name='conv',
                                        downsample=downsample, bottleneck_factor=bottleneck_factor, **kwargs)
            else:
                x = cls.conv_block(inputs, layout, filters, bottleneck, bottleneck_factor, name='conv',
                                   downsample=downsample, **kwargs)

            data_format = kwargs.get('data_format')
            inputs_channels = cls.num_channels(inputs, data_format)
            x_channels = cls.num_channels(x, data_format)

            if se_block:
                x = cls.se_block(x, se_factor, **kwargs)

            with tf.variable_scope('shortcut'):
                strides = 2 if downsample else 1
                shortcut = inputs
                if zero_pad and inputs_channels < x_channels:
                    if downsample:
                        shortcut = conv_block(shortcut, name='shortcut',
                                              **{**kwargs, **dict(layout='c', filters=inputs_channels,
                                                                  kernel_size=1, strides=strides)})
                    if inputs_channels < x_channels:
                        axis = cls.channels_axis(kwargs.get('data_format'))
                        padding = [[0, 0] for _ in range(shortcut.shape.ndims)]
                        padding[axis] = [0, x_channels - inputs_channels]
                        shortcut = tf.pad(shortcut, padding)
                elif inputs_channels != x_channels or downsample:
                    shortcut = conv_block(shortcut, name='shortcut',
                                          **{**kwargs, **dict(layout='c', filters=x_channels,
                                                              kernel_size=1, strides=strides)})

            x = x + shortcut

            if post_activation:
                x = conv_block(x, layout=post_activation, name='post_activation', **kwargs)

            x = tf.identity(x, name='output')

        return x

    @classmethod
    def conv_block(cls, inputs, layout, filters, bottleneck, bottleneck_factor=None, **kwargs):
        """ ResNet convolution block

        Parameters
        ----------
        inputs : tf.Tensor
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
        tf.Tensor
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
        inputs : tf.Tensor
            input tensor
        layout : str
            a sequence of layers in the block
        filters : int
            number of filters
        kernel_size : int or tuple
            convolution kernel size
        downsample : bool
            whether to decrease spatial dimensions with strides=2
        name : str
            scope name
        kwargs : dict
            conv_block parameters

        Returns
        -------
        tf.Tensor
        """
        n = layout.count('c') + layout.count('C') - 1
        strides = ([2] + [1] * n) if downsample else 1
        return conv_block(inputs, layout, filters=filters, kernel_size=kernel_size, strides=strides, **kwargs)

    @classmethod
    def bottleneck_block(cls, inputs, layout=None, filters=None, kernel_size=None, bottleneck_factor=4,
                         downsample=False, **kwargs):
        """ A stack of 1x1, 3x3, 1x1 convolutions

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        layout : str
            a sequence of layers in the block
        filters : int or list of int
            if int, number of filters in the first convolution.
            if list, number of filters in each layer.
        kernel_size : int or tuple
            convolution kernel size
        bottleneck_factor : int
            scale factor for the number of filters in the last convolution
            (if filters is int, otherwise is not used)
        downsample : bool
            whether to decrease spatial dimensions with strides=2
        kwargs : dict
            conv_block parameters

        Returns
        -------
        tf.Tensor
        """
        kernel_size = [1, 3, 1] if kernel_size is None else kernel_size
        n = layout.count('c') + layout.count('C') - 2
        if kwargs.get('strides') is None:
            strides = ([1, 2] + [1] * n) if downsample else 1
        else:
            strides = kwargs.pop('strides')
        if isinstance(filters, int):
            filters = [filters, filters, filters * bottleneck_factor]
        x = conv_block(inputs, layout, filters=filters, kernel_size=kernel_size, strides=strides, **kwargs)
        return x

    @classmethod
    def next_conv_block(cls, inputs, layout, filters, bottleneck, resnext_factor, name, **kwargs):
        """ ResNeXt convolution block

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        layout : str
            a sequence of layers in the block
        filters : int
            number of output filters
        bottleneck : bool
            whether to use a simple or a bottleneck block
        resnext_factor : int
            cardinality for ResNeXt model
        name : str
            scope name
        kwargs : dict
            `conv_block` parameters

        Returns
        -------
        tf.Tensor
        """
        if bottleneck:
            sub_blocks = []
            with tf.variable_scope(name):
                out_filters = filters * kwargs['bottleneck_factor']
                in_filters = out_filters // 2 // resnext_factor
                filters = [in_filters, in_filters, out_filters]
                for i in range(resnext_factor):
                    x = cls.conv_block(inputs, layout, filters=filters, bottleneck=True,
                                       name='next_conv_block-%d' % i, **kwargs)
                    sub_blocks.append(x)
                x = tf.add_n(sub_blocks)
        else:
            _filters = resnext_factor * 4
            x = cls.conv_block(inputs, layout, filters=[_filters, filters], bottleneck=False, name=name, **kwargs)

        return x

    @classmethod
    def make_encoder(cls, inputs, name='encoder', **kwargs):
        """ Build the body and return encoder tensors

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name
        kwargs : dict
            body params

        Returns
        -------
        tf.Tensor

        Raises
        ------
        ValueError
            If `filters` is not specified or number of `num_blocks` not equal to number of `filters` provided.
        """
        num_blocks = cls.get('num_blocks', config=cls.fill_params('body', **kwargs))

        if kwargs.get('filters') is None:
            raise ValueError('Specify number of filters')

        if len(num_blocks) != len(kwargs.get('filters')):
            msg = '{} encoder requires {} filters instead of {}'
            raise ValueError(msg.format(cls.__name__, len(num_blocks), len(kwargs.get('filters'))))

        with tf.variable_scope(name):
            x = cls.body(inputs, name='body', **kwargs)

            scope = tf.get_default_graph().get_name_scope()
            encoder_tensors = []
            for i, _ in enumerate(num_blocks):
                tensor_name = scope + '/body/group-%d'%i + '/output:0'
                x = tf.get_default_graph().get_tensor_by_name(tensor_name)
                encoder_tensors.append(x)
        return encoder_tensors


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
