""" Jegou S. et al "`The One Hundred Layers Tiramisu:
Fully Convolutional DenseNets for Semantic Segmentation
<https://arxiv.org/abs/1611.09326>`_"
"""
import tensorflow as tf

from . import TFModel
from .densenet import DenseNet


class DenseNetFC(TFModel):
    """ DenseNet for semantic segmentation

    **Configuration**

    inputs : dict
        dict with 'images' and 'masks' (see :meth:`~.TFModel._make_inputs`)

    body : dict
        num_layers : list of int
            number of layers in downsampling/upsampling blocks

        block : dict
            dense block parameters

        transition_down : dict
            downsampling transition layer parameters

        transition_up : dict
            upsampling transition layer parameters
    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()

        config['common/conv/use_bias'] = False
        config['initial_block'] = dict(layout='c', filters=48, kernel_size=3, strides=1)

        config['body']['block'] = dict(layout='nacd', dropout_rate=.2, growth_rate=12, bottleneck=False)
        config['body']['transition_up'] = dict(layout='t', factor=2, kernel_size=3)
        config['body']['transition_down'] = dict(layout='nacdp', kernel_size=1, strides=1,
                                                 pool_size=2, pool_strides=2, dropout_rate=.2,
                                                 reduction_factor=1)

        config['head'].update(dict(layout='c', kernel_size=1))

        config['loss'] = 'ce'

        return config

    def build_config(self, names=None):
        config = super().build_config(names)
        if config.get('head/filters') is None:
            config['head/filters'] = self.num_classes('targets')
        return config

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ FC DenseNet body

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
        num_layers, block = cls.pop(['num_layers', 'block'], kwargs)
        trans_up, trans_down = cls.pop(['transition_up', 'transition_down'], kwargs)
        block = {**kwargs, **block}
        trans_up = {**kwargs, **trans_up}
        trans_down = {**kwargs, **trans_down}

        with tf.variable_scope(name):
            x, inputs = inputs, None
            encoder_outputs = []
            for i, n_layers in enumerate(num_layers[:-1]):
                x = DenseNet.block(x, num_layers=n_layers, name='encoder-%d' % i, **block)
                encoder_outputs.append(x)
                x = cls.transition_down(x, name='transition_down-%d' % i, **trans_down)
            x = DenseNet.block(x, num_layers=num_layers[-1], name='encoder-%d' % len(num_layers), **block)

            axis = cls.channels_axis(kwargs.get('data_format'))
            for i, n_layers in enumerate(num_layers[-2::-1]):
                x = cls.transition_up(x, filters=num_layers[-i-1] * block['growth_rate'],
                                      name='transition_up-%d' % i, **trans_up)
                x = DenseNet.block(x, num_layers=n_layers, name='decoder-%d' % i, **block)
                x = cls.crop(x, encoder_outputs[-i-1], data_format=kwargs.get('data_format'))
                x = tf.concat((x, encoder_outputs[-i-1]), axis=axis)
        return x

    @classmethod
    def transition_down(cls, inputs, name='transition_down', **kwargs):
        """ A downsampling interconnect layer between two dense blocks

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
        kwargs = cls.fill_params('body/transition_down', **kwargs)
        return DenseNet.transition_layer(inputs, name=name, **kwargs)

    @classmethod
    def transition_up(cls, inputs, name='transition_up', **kwargs):
        """ An upsampling interconnect layer between two dense blocks

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
        kwargs = cls.fill_params('body/transition_up', **kwargs)
        filters = kwargs.pop('filters', cls.num_channels(inputs, kwargs.get('data_format')))
        return cls.upsample(inputs, filters=filters, name=name, **kwargs)


class DenseNetFC56(DenseNetFC):
    """ FC DenseNet-56 architecture """
    @classmethod
    def default_config(cls):
        config = DenseNetFC.default_config()
        config['body']['num_layers'] = [4] * 6
        config['body']['block']['growth_rate'] = 12
        return config

class DenseNetFC67(DenseNetFC):
    """ FC DenseNet-67 architecture """
    @classmethod
    def default_config(cls):
        config = DenseNetFC.default_config()
        config['body']['num_layers'] = [5] * 6
        config['body']['block']['growth_rate'] = 16
        return config

class DenseNetFC103(DenseNetFC):
    """ FC DenseNet-103 architecture """
    @classmethod
    def default_config(cls):
        config = DenseNetFC.default_config()
        config['body']['num_layers'] = [4, 5, 7, 10, 12, 15]
        config['body']['block']['growth_rate'] = 16
        return config
