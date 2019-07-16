"""
Shelhamer E. et al "`Fully Convolutional Networks for Semantic Segmentation
<https://arxiv.org/abs/1605.06211>`_"
"""
import tensorflow as tf

from . import TFModel, VGG16
from .layers import conv_block


class FCN(TFModel):
    """ Base Fully convolutional network (FCN) """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()
        config['common/dropout_rate'] = .5
        config['initial_block/base_network'] = VGG16
        config['body/filters'] = 100
        config['body/upsample'] = dict(layout='t', kernel_size=4)
        config['head/upsample'] = dict(layout='t')

        config['loss'] = 'ce'

        return config

    def build_config(self, names=None):
        config = super().build_config(names)

        config['body/num_classes'] = self.num_classes('targets')
        config['head/num_classes'] = self.num_classes('targets')
        config['head/targets'] = self.get_from_attr('targets')

        return config

    @classmethod
    def initial_block(cls, inputs, base_network, name='initial_block', **kwargs):
        """ Base network

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        base_network : class
            base network class
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            x = base_network.initial_block(inputs, name='initial_block', **kwargs)
            x = base_network.body(x, name='body', **kwargs)
        return x

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor

        Returns
        -------
        tf.Tensor
        """
        raise NotImplementedError()

    @classmethod
    def head(cls, inputs, targets, num_classes, name='head', **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        targets : tf.Tensor
            the tensor with source images (provide the shape to upsample to)
        num_classes : int
            number of classes

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('head', **kwargs)
        upsample_args = cls.pop('upsample', kwargs)

        x = cls.upsample(inputs, filters=num_classes, name=name, **upsample_args, **kwargs)
        x = cls.crop(x, targets, kwargs.get('data_format'))
        return x


class FCN32(FCN):
    """  Fully convolutional network (FCN32)

    **Configuration**

    inputs : dict
        dict with 'images' and 'masks' (see :meth:`~.TFModel._make_inputs`)

    initial_block : dict
        base_network : class
            base network (VGG16 by default)

    body : dict
        filters : int
            number of filters in convolutions after base network (default=100)
        upsample : dict
            upsampling parameters (default={factor:2, layout:t, kernel_size:4)

    head : dict
        upsample : dict
            upsampling parameters (default={factor:32, layout:t, kernel_size:64)
    """
    @classmethod
    def default_config(cls):
        config = FCN.default_config()
        config['head']['upsample'].update(dict(factor=32, kernel_size=64))
        config['body'].update(dict(layout='cnad cnad', dropout_rate=.5, kernel_size=[7, 1]))
        return config

    @classmethod
    def body(cls, inputs, num_classes, name='body', **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        num_classes : int
            number of classes

        Returns
        -------
        tf.Tensor
        """
        _ = num_classes
        kwargs = cls.fill_params('body', **kwargs)
        return conv_block(inputs, name=name, **kwargs)


class FCN16(FCN):
    """  Fully convolutional network (FCN16)

    **Configuration**

    inputs : dict
        dict with 'images' and 'masks' (see :meth:`~.TFModel._make_inputs`)

    initial_block : dict
        base_network : class
            base network (VGG16 by default)
        skip_name : str
            tensor name for the skip connection.
            Default='block-3/output:0' for VGG16.

    body : dict
        filters : int
            number of filters in convolutions after base network (default=100)
        upsample : dict
            upsampling parameters (default={factor:2, layout:t, kernel_size:4)

    head : dict
        upsample : dict
            upsampling parameters (default={factor:16, layout:t, kernel_size:32)
    """
    @classmethod
    def default_config(cls):
        config = FCN.default_config()
        config['head']['upsample'].update(dict(factor=16, kernel_size=32))
        config['initial_block']['skip_name'] = '/initial_block/body/block-3/output:0'
        return config

    @classmethod
    def initial_block(cls, inputs, name='initial_block', **kwargs):
        kwargs = cls.fill_params('initial_block', **kwargs)

        x = FCN.initial_block(inputs, name=name, **kwargs)
        skip_name = tf.get_default_graph().get_name_scope() + kwargs['skip_name']
        skip = tf.get_default_graph().get_tensor_by_name(skip_name)
        return x, skip

    @classmethod
    def body(cls, inputs, num_classes, name='body', **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs : tf.Tensor
            two input tensors
        num_classes : int
            number of classes

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body', **kwargs)
        filters = cls.pop('filters', kwargs)
        upsample_args = kwargs['upsample']

        with tf.variable_scope(name):
            x, skip = inputs
            inputs = None
            x = FCN32.body(x, filters=filters, num_classes=num_classes, name='fcn32', **kwargs)

            x = cls.upsample(x, factor=2, filters=num_classes, name='fcn32_upsample', **upsample_args, **kwargs)

            skip = conv_block(skip, 'c', filters=num_classes, kernel_size=1, name='pool4', **kwargs)
            x = cls.crop(x, skip, kwargs.get('data_format'))
            output = tf.add(x, skip, name='output')
        return output


class FCN8(FCN):
    """  Fully convolutional network (FCN8)

    **Configuration**

    inputs : dict
        dict with 'images' and 'masks' (see :meth:`~.TFModel._make_inputs`)

    initial_block : dict
        base_network : class
            base network (VGG16 by default)
        skip1_name : str
            tensor name for the first skip connection.
            Default='block-3/output:0' for VGG16.
        skip2_name : str
            tensor name for the second skip connection.
            Default='block-2/output:0' for VGG16.

    body : dict
        filters : int
            number of filters in convolutions after base network (default=100)
        upsample : dict
            upsampling parameters (default={factor:2, layout:t, kernel_size:4)

    head : dict
        upsample : dict
            upsampling parameters (default={factor:8, layout:t, kernel_size:16)
    """
    @classmethod
    def default_config(cls):
        config = FCN.default_config()
        config['head']['upsample'].update(dict(factor=8, kernel_size=16))
        config['initial_block']['skip1_name'] = '/initial_block/body/block-3/output:0'
        config['initial_block']['skip2_name'] = '/initial_block/body/block-2/output:0'
        return config

    @classmethod
    def initial_block(cls, inputs, name='initial_block', **kwargs):
        kwargs = cls.fill_params('initial_block', **kwargs)
        x = FCN.initial_block(inputs, name=name, **kwargs)
        skip1_name = tf.get_default_graph().get_name_scope() + kwargs['skip1_name']
        skip1 = tf.get_default_graph().get_tensor_by_name(skip1_name)
        skip2_name = tf.get_default_graph().get_name_scope() + kwargs['skip2_name']
        skip2 = tf.get_default_graph().get_tensor_by_name(skip2_name)
        return x, skip1, skip2

    @classmethod
    def body(cls, inputs, num_classes, name='body', **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        num_classes : int
            number of classes

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body', **kwargs)
        filters = cls.pop('filters', kwargs)
        upsample_args = kwargs['upsample']

        with tf.variable_scope(name):
            x, skip1, skip2 = inputs
            inputs = None

            x = FCN16.body((x, skip1), filters=filters, num_classes=num_classes, name='fcn16', **kwargs)
            x = cls.upsample(x, factor=2, filters=num_classes, name='fcn16_upsample', **upsample_args, **kwargs)

            skip2 = conv_block(skip2, 'c', num_classes, 1, name='pool3', **kwargs)

            x = cls.crop(x, skip2, kwargs.get('data_format'))
            output = tf.add(x, skip2, name='output')
        return output
