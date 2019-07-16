"""
Huang G. et al. "`Densely Connected Convolutional Networks
<https://arxiv.org/abs/1608.06993>`_"
"""
import tensorflow as tf

from . import TFModel
from .layers import conv_block


class DenseNet(TFModel):
    """ DenseNet

    **Configuration**

    inputs : dict
        dict with 'images' and 'labels'. See :meth:`~.TFModel._make_inputs`.

    initial_block : dict

    body : dict
        num_layers : list of int
            number of layers in dense blocks

        block : dict
            parameters for dense block, including :func:`~.layers.conv_block` parameters, as well as

            growth_rate : int
                number of output filters in each layer (default=32)

            bottleneck : bool
                whether to use 1x1 convolutions in each layer (default=True)

            skip : bool
                whether to concatenate inputs to the output tensor

    transition_layer : dict
        parameters for transition layers, including :func:`~.layers.conv_block` parameters, as well as

        reduction_factor : float
            a multiplier for number of output filters (default=1)

    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()
        config['common/conv/use_bias'] = False
        config['initial_block'] = dict(layout='cnap', filters=16, kernel_size=7, strides=2,
                                       pool_size=3, pool_strides=2)
        config['body/block'] = dict(layout='nacd', dropout_rate=.2, growth_rate=32, bottleneck=True, skip=True)
        config['body/transition_layer'] = dict(layout='nacv', kernel_size=1, strides=1,
                                               pool_size=2, pool_strides=2, reduction_factor=1)
        config['head'] = dict(layout='Vf')

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
        """ Base layers

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
        num_layers, block, transition = cls.pop(['num_layers', 'block', 'transition_layer'], kwargs)
        block = {**kwargs, **block}
        transition = {**kwargs, **transition}

        with tf.variable_scope(name):
            x, inputs = inputs, None
            for i, n_layers in enumerate(num_layers):
                with tf.variable_scope('group-%d' % i):
                    x = cls.block(x, num_layers=n_layers, name='block-%d' % i, **block)
                    if 0 < i < len(num_layers):
                        x = cls.transition_layer(x, name='transition-%d' % i, **transition)
                    x = tf.identity(x, name='output')
        return x

    @classmethod
    def block(cls, inputs, num_layers=3, name=None, **kwargs):
        """ A network building block consisting of a stack of 1x1 and 3x3 convolutions.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        num_layers : int
            number of conv layers
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body/block', **kwargs)
        layout, growth_rate, bottleneck, skip = \
            cls.pop(['layout', 'growth_rate', 'bottleneck', 'skip'], kwargs)

        with tf.variable_scope(name):
            axis = cls.channels_axis(kwargs['data_format'])
            x = inputs
            all_layers = []
            for i in range(num_layers):
                if len(all_layers) > 0:
                    x = tf.concat([inputs] + all_layers, axis=axis, name='concat-%d' % i)
                if bottleneck:
                    x = conv_block(x, filters=growth_rate * 4, kernel_size=1, layout=layout,
                                   name='bottleneck-%d' % i, **kwargs)
                x = conv_block(x, filters=growth_rate, kernel_size=3, layout=layout,
                               name='conv-%d' % i, **kwargs)
                all_layers.append(x)

            if skip:
                all_layers = [inputs] + all_layers
            x = tf.concat(all_layers, axis=axis, name='concat-%d' % num_layers)
        return x

    @classmethod
    def transition_layer(cls, inputs, name='transition_layer', **kwargs):
        """ An intermediary interconnect layer between two dense blocks

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
        kwargs = cls.fill_params('body/transition_layer', **kwargs)
        reduction_factor = cls.get('reduction_factor', config=kwargs)
        num_filters = cls.num_channels(inputs, kwargs.get('data_format'))
        return conv_block(inputs, filters=num_filters * reduction_factor, name=name, **kwargs)


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
        """
        num_layers = cls.get('num_layers', config=cls.fill_params('body', **kwargs))

        with tf.variable_scope(name):
            x = cls.body(inputs, name='body', **kwargs)

            scope = tf.get_default_graph().get_name_scope()
            encoder_tensors = []
            for i, _ in enumerate(num_layers):
                tensor_name = scope + '/body/group-%d'%i + '/output:0'
                x = tf.get_default_graph().get_tensor_by_name(tensor_name)
                encoder_tensors.append(x)
        return encoder_tensors



class DenseNet121(DenseNet):
    """ The original DenseNet-121 architecture """
    @classmethod
    def default_config(cls):
        config = DenseNet.default_config()
        config['body']['num_layers'] = [6, 12, 24, 32]
        return config

class DenseNet169(DenseNet):
    """ The original DenseNet-169 architecture """
    @classmethod
    def default_config(cls):
        config = DenseNet.default_config()
        config['body']['num_layers'] = [6, 12, 32, 16]
        return config

class DenseNet201(DenseNet):
    """ The original DenseNet-201 architecture """
    @classmethod
    def default_config(cls):
        config = DenseNet.default_config()
        config['body']['num_layers'] = [6, 12, 48, 32]
        return config

class DenseNet264(DenseNet):
    """ The original DenseNet-264 architecture """
    @classmethod
    def default_config(cls):
        config = DenseNet.default_config()
        config['body']['num_layers'] = [6, 12, 64, 48]
        return config
