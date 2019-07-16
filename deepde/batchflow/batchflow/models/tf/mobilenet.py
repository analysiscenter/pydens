""" Howard A. et al. "`MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
<https://arxiv.org/abs/1704.04861>`_"

Sandler M. et al. "`Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation
<https://arxiv.org/abs/1801.04381>`_"
"""
from copy import deepcopy
import tensorflow as tf

from . import TFModel
from .layers import conv_block


_V1_DEFAULT_BODY = {
    'strides': [1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2],
    'double_filters': [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
    'width_factor': 1
}


class MobileNet(TFModel):
    """ MobileNet

    **Configuration**

    inputs : dict
        dict with 'images' and 'labels' (see :meth:`~.TFModel._make_inputs`)

    initial_block : dict
        parameters for the initial block (default is 'cna', 32, 3, strides=2)

    body : dict
        strides : list of int
            strides in separable convolutions

        double_filters : list of bool
            if True, number of filters in 1x1 covolution will be doubled

        width_factor : float
            multiplier for the number of channels (default=1)
    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()
        config['initial_block'] = dict(layout='cna', filters=32, kernel_size=3, strides=2)
        config['body'].update(_V1_DEFAULT_BODY)
        config['head'].update(dict(layout='Vf'))

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
        sep_strides, double_filters, width_factor = \
            cls.pop(['strides', 'double_filters', 'width_factor'], kwargs)

        with tf.variable_scope(name):
            x = inputs
            for i, strides in enumerate(sep_strides):
                x = cls.block(x, strides=strides, double_filters=double_filters[i], width_factor=width_factor,
                              name='block-%d' % i, **kwargs)
        return x

    @classmethod
    def block(cls, inputs, strides=1, double_filters=False, width_factor=1, name=None, **kwargs):
        """ A network building block consisting of a separable depthwise convolution and 1x1 pointwise covolution.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        strides : int
            strides in separable convolution
        double_filters : bool
            if True number of filters in 1x1 covolution will be doubled
        width_factor : float
            multiplier for the number of filters
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        num_filters = int(cls.num_channels(inputs, kwargs.get('data_format')) * width_factor)
        filters = [num_filters, num_filters*2] if double_filters else num_filters
        return conv_block(inputs, 'Cna cna', filters, [3, 1], name=name, strides=[strides, 1], **kwargs)


_V2_DEFAULT_BODY = [
    dict(repeats=1, filters=16, expansion_factor=1, strides=1),
    dict(repeats=2, filters=24, expansion_factor=6, strides=2),
    dict(repeats=3, filters=32, expansion_factor=6, strides=2),
    dict(repeats=4, filters=64, expansion_factor=6, strides=2),
    dict(repeats=3, filters=96, expansion_factor=6, strides=1),
    dict(repeats=3, filters=160, expansion_factor=6, strides=2),
    dict(repeats=1, filters=320, expansion_factor=6, strides=1),
]

class MobileNet_v2(TFModel):
    """ MobileNet version 2

    **Configuration**

    inputs : dict
        dict with 'images' and 'labels' (see :meth:`.TFModel._make_inputs`)

    initial_block : dict
        parameters for the initial block (default is 'cna', 32, 3, strides=2)

    body : dict
        layout : list of dict
            a sequence of block parameters:
            repeats : int
            filters : int
            expansion_factor : int
            strides : int

        width_factor : float
            multiplier for the number of channels (default=1)
    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()
        config['common'].update(dict(activation=tf.nn.relu6))
        config['initial_block'].update(dict(layout='cna', filters=32, kernel_size=3, strides=2))
        config['body'].update(dict(width_factor=1, layout=deepcopy(_V2_DEFAULT_BODY)))
        config['head'].update(dict(layout='cnacnV', filters=[1280, 2], kernel_size=1))

        config['loss'] = 'ce'

        return config

    def build_config(self, names=None):
        config = super().build_config(names)
        config['head']['filters'][-1] = self.num_classes('targets')
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
        width_factor, layout = cls.pop(['width_factor', 'layout'], kwargs)

        with tf.variable_scope(name):
            x = inputs
            i = 0
            for block in layout:
                repeats = block.pop('repeats')
                block['width_factor'] = width_factor
                for k in range(repeats):
                    if k > 0:
                        block['strides'] = 1
                    x = cls.block(x, **block, residual=k > 0, name='block-%d' % i, **kwargs)
                    i += 1
        return x

    @classmethod
    def block(cls, inputs, filters, residual=False, strides=1, expansion_factor=6, width_factor=1, name=None, **kwargs):
        """ A network building block consisting of a separable depthwise convolution and 1x1 pointwise covolution.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : int
            number of output filters
        residual : boold
            whether to make a residual connection
        strides : int
            stride for 3x3 convolution
        expansion_factor : int
            multiplier for the number of filters in internal convolutions
        width_factor : float
            multiplier for the number of filters
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        num_filters = int(cls.num_channels(inputs, kwargs.get('data_format')) * expansion_factor * width_factor)
        conv_filters = [num_filters, num_filters, filters]
        x = conv_block(inputs, 'cna Cna cn', conv_filters, [1, 3, 1], name=name, strides=[1, strides, 1], **kwargs)
        if residual:
            x = inputs + x
        return x
