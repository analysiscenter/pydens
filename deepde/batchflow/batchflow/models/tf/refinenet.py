"""  Lin G. et al "`RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation
<https://arxiv.org/abs/1611.06612>`_"
"""
import tensorflow as tf

from .layers import conv_block
from . import TFModel
from .resnet import ResNet, ResNet101


class RefineNet(TFModel):
    """ RefineNet

    **Configuration**

    inputs : dict
        dict with 'images' and 'masks' (see :meth:`~.TFModel._make_inputs`)

    body : dict
        encoder : dict
            base_class : TFModel
                a model implementing ``make_encoder`` method which returns tensors
                with encoded representation of the inputs
            other args
                parameters for base class ``make_encoder`` method

        filters : list of int
            number of filters in each decoder block (default=[512, 256, 256, 256])

        upsample : dict
            :meth:`~.TFModel.upsample` parameters to use in each decoder block

    head : dict
        num_classes : int
            number of semantic classes
    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()

        filters = 64   # number of filters in the first block
        config['initial_block'] = dict(layout='cna cna', filters=filters, kernel_size=3,
                                       strides=1, pool_strides=1)
        config['body']['encoder'] = dict(base_class=ResNet101)
        config['body']['filters'] = [512, 256, 256, 256]
        config['body']['upsample'] = dict(layout='b', factor=2)
        config['loss'] = 'ce'
        return config

    def build_config(self, names=None):
        config = super().build_config(names)
        if config.get('head/num_classes') is None:
            config['head/num_classes'] = self.num_classes('targets')
        config['head/targets'] = self.get_from_attr('targets')
        return config

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : tuple of int
            number of filters in decoder blocks
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body', **kwargs)
        encoder = kwargs.pop('encoder')
        filters = kwargs.pop('filters')

        with tf.variable_scope(name):
            encoder_outputs = cls.encoder(inputs, **encoder, **kwargs)

            x = None
            for i, tensor in enumerate(encoder_outputs[::-1]):
                decoder_inputs = tensor if x is None else (tensor, x)
                x = cls.decoder_block(decoder_inputs, filters=filters[i], name='decoder-'+str(i), **kwargs)
        return x

    @classmethod
    def head(cls, inputs, targets, num_classes, layout='c', kernel_size=1, name='head', **kwargs):
        with tf.variable_scope(name):
            x, inputs = inputs, None
            x = cls.crop(x, targets, kwargs['data_format'])
            x = conv_block(x, layout, filters=num_classes, kernel_size=kernel_size, **kwargs)
        return x

    @classmethod
    def encoder(cls, inputs, base_class, name='encoder', **kwargs):
        """ Create encoder from a base_class model

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        base_class : TFModel
            a model class (default=ResNet101).
            Should implement ``make_encoder`` method.
        name : str
            scope name
        kwargs : dict
            parameters for ``make_encoder`` method

        Returns
        -------
        tf.Tensor
        """
        x = base_class.make_encoder(inputs, name=name, **kwargs)
        return x

    @classmethod
    def block(cls, inputs, filters=None, name='block', **kwargs):
        """ RefineNet block with Residual Conv Unit, Multi-resolution fusion and Chained-residual pooling.

        Parameters
        ----------
        inputs : tuple of tf.Tensor
            input tensors (the first should have the largest spatial dimension)
        filters : int
            the number of output filters for all convolutions within the block
        name : str
            scope name
        kwargs : dict
            upsample : dict
                upsample params

        Returns
        -------
        tf.Tensor
        """
        upsample_args = cls.pop('upsample', kwargs)
        upsample_args = {**kwargs, **upsample_args}

        with tf.variable_scope(name):
            #filters = min([cls.num_channels(t, data_format=kwargs['data_format']) for t in inputs])
            # Residual Conv Unit
            after_rcu = []
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            for i, tensor in enumerate(inputs):
                x = ResNet.double_block(tensor, filters=filters, layout='acac',
                                        bottleneck=False, downsample=False,
                                        name='rcu-%d' % i, **kwargs)
                after_rcu.append(x)
            inputs = None

            # Multi-resolution fusion
            with tf.variable_scope('mrf'):
                after_mrf = 0
                for i, tensor in enumerate(after_rcu):
                    x = conv_block(tensor, layout='ac', filters=filters, kernel_size=3,
                                   name='conv-%d' % i, **kwargs)
                    if i != 0:
                        x = cls.upsample((x, after_rcu[0]), name='upsample-%d' % i, **upsample_args)
                    after_mrf += x
            # free memory
            x, after_mrf = after_mrf, None
            after_rcu = None

            # Chained-residual pooling
            x = tf.nn.relu(x)
            after_crp = x
            num_pools = 4
            for i in range(num_pools):
                x = conv_block(x, layout='pc', filters=filters, kernel_size=3, strides=1,
                               pool_size=5, pool_strides=1, name='rcp-%d' % i, **kwargs)
                after_crp += x

            x, after_crp = after_crp, None
            x = ResNet.double_block(x, layout='ac ac', filters=filters, bottleneck=False, downsample=False,
                                    name='rcu-last', **kwargs)
            x = tf.identity(x, name='output')
        return x

    @classmethod
    def decoder_block(cls, inputs, filters, name, **kwargs):
        """ Call RefineNet block

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : int
            number of output filters
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        return cls.block(inputs, filters=filters, name=name, **kwargs)
