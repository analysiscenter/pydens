""" Peng C. et al. "`Large Kernel Matters -
Improve Semantic Segmentation by Global Convolutional Network
<https://arxiv.org/abs/1703.02719>`_"
"""
import tensorflow as tf

from .layers import conv_block
from . import TFModel
from .resnet import ResNet, ResNet101


class GlobalConvolutionNetwork(TFModel):
    """ Global Convolution Network

    **Configuration**

    inputs : dict
        dict with 'images' and 'masks' (see :meth:`~.TFModel._make_inputs`)

    body : dict
        encoder : dict
            base_class : TFModel
                encoder model class (should implement ``make_encoder``)
            filters : list of int
                number of filters in encoder convolutions
        block : dict
            parameters for Global Convolution Network conv block
        br : dict
            parameters for boundary refinement conv block
        upsample : dict
            parameters for upsampling in decoders

    head : dict
        num_classes : int
            number of semantic classes
        upsample : dict
            parameters for upsampling
    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()

        config['initial_block'] = dict(layout='cna', filters=64, kernel_size=7, strides=2)
        config['body/encoder'] = dict(base_class=ResNet101, filters=[256, 512, 1024, 2048])
        config['body/block'] = dict(layout='cn cn', filters=21, kernel_size=11)
        config['body/res_block'] = False
        config['body/br'] = dict(layout='ca c', kernel_size=3, bottleneck=False, downsample=False)
        config['body/upsample'] = dict(layout='tna', factor=2, kernel_size=4)

        config['head/upsample'] = dict(layout='tna', factor=2, kernel_size=4)

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
        """ GCN body

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
        block, br_block, encoder, upsample = cls.pop(['block', 'br', 'encoder', 'upsample'], kwargs)

        with tf.variable_scope(name):
            encoder_outputs = cls.encoder(inputs, block=block, br=br_block, encoder=encoder, **kwargs)

            encoder_outputs = [inputs] + encoder_outputs
            x = encoder_outputs[-1]
            for i, tensor in enumerate(encoder_outputs[-2::-1]):
                with tf.variable_scope('decoder-%d' % i):
                    x = cls.decoder_block(x, tensor, **upsample, **kwargs)
                    if i < len(encoder_outputs) - 2:
                        x = tf.add(x, tensor)
                    x = cls.boundary_refinement(x, name='BR', **br_block, **kwargs)
        return x

    @classmethod
    def encoder(cls, inputs, name='encoder', **kwargs):
        """ Create encoder from a base_class model

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name
        kwargs : dict
            parameters

        Returns
        -------
        tf.Tensor
        """
        encoder = cls.fill_params('body/encoder', **kwargs.pop('encoder', {}))
        base_class = cls.pop('base_class', encoder)
        block, br_block, res_block = cls.pop(['block', 'br', 'res_block'], kwargs)

        with tf.variable_scope(name):
            base_tensors = base_class.make_encoder(inputs, name='base', **{**kwargs, **encoder})
            encoder_tensors = []
            for i, tensor in enumerate(base_tensors):
                with tf.variable_scope('encoder-%d' % i):
                    if res_block:
                        kwargs['layout'] = 'cna'
                        x = cls.res_block(tensor, name='resGCN', **block, **kwargs)
                    else:
                        x = cls.block(tensor, name='GCN', **block, **kwargs)
                    x = cls.boundary_refinement(x, name='BR', **br_block, **kwargs)
                encoder_tensors.append(x)
        return encoder_tensors

    @classmethod
    def block(cls, inputs, name, **kwargs):
        """ Two branches with two convolutions with large kernels (k,1)(1,k) and (1,k)(k,1)

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : int
            number of output filters
        kernel_size : int
            convolution kernel size
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body/block', **kwargs)
        kernel_size = cls.pop('kernel_size', kwargs)
        i = inputs
        with tf.variable_scope(name):
            kernel_size = [(1, kernel_size), (kernel_size, 1)]
            x = conv_block(i, kernel_size=kernel_size, name='left', **kwargs)
            y = conv_block(i, kernel_size=kernel_size[::-1], name='right', **kwargs)
            x = x + y
        return x

    @classmethod
    def res_block(cls, inputs, name, **kwargs):
        """ The ResNet GCN block, shown in Figure 5, with a 1x1 conv layer in skip connection
        to correct shapes of the tensors involved.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : int
            number of output filters
        kernel_size : int
            convolution kernel size
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            x = cls.block(inputs, name, **kwargs)
            x = conv_block(x, name='norm', **{**kwargs, 'layout':'cn', 'kernel_size':1})
            y = conv_block(inputs, name='skip', **{**kwargs, 'layout':'cn', 'kernel_size':1})
            x = x + y
        return x

    @classmethod
    def boundary_refinement(cls, inputs, name, **kwargs):
        """ An ordinary ResNet block

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
        kwargs = cls.fill_params('body/br', **kwargs)
        kwargs['filters'] = cls.num_channels(inputs, data_format=kwargs['data_format'])
        return ResNet.block(inputs, name=name, **kwargs)

    @classmethod
    def decoder_block(cls, inputs, targets, filters=None, name='decoder', **kwargs):
        """ Upsample and crop if needed

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        targets : tf.Tensor
            tensor which size to upsample to
        filters : int or None
            number of filters in upsampling block
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        filters = filters or cls.num_channels(inputs, data_format=kwargs.get('data_format'))
        with tf.variable_scope(name):
            x = cls.upsample(inputs, filters=filters, name='upsample', **kwargs)
            x = cls.crop(x, targets, data_format=kwargs.get('data_format'))
        return x

    @classmethod
    def head(cls, inputs, targets, num_classes, name='head', **kwargs):
        """ Upsample and boundary refinement blocks

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        targets : tf.Tensor
            target tensor
        num_classes : int
            number of classes (and number of filters in the last convolution)
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('head', **kwargs)
        upsample = cls.pop('upsample', kwargs)

        with tf.variable_scope(name):
            x = cls.decoder_block(inputs, targets, filters=num_classes, name='upsample', **{**kwargs, **upsample})
            x = cls.boundary_refinement(x, name='BR', **kwargs)
        return x
