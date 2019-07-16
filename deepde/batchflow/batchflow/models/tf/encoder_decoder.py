"""  Encoder-decoder """

import tensorflow as tf

from .layers import conv_block
from . import TFModel


class EncoderDecoder(TFModel):
    """ Encoder-decoder architecture. Allows to combine features of different models,
    e.g. ResNet and DenseNet, in order to create new ones with just a few lines of code.

    Parameters
    ----------
    inputs : dict
        Dictionary with 'images' (see :meth:`~.TFModel._make_inputs`)

    body : dict
        encoder : dict
            base : TFModel
                Model implementing ``make_encoder`` method which returns tensors
                with encoded representation of the inputs.

            num_stages : int
                Number of downsampling stages.

            downsample : dict
                Parameters for downsampling (see :func:`~.layers.conv_block`)

            blocks : dict
                Parameters for pre-processing blocks:

                base : callable
                    Tensor processing function. Default is :func:`~.layers.conv_block`.
                other args : dict
                    Parameters for the base block.

            other args : dict
                Parameters for ``make_encoder`` method.

        embedding : dict
            base : callable
                Tensor processing function. Default is :func:`~.layers.conv_block`.
            other args
                Parameters for the base block.

        decoder : dict
            num_stages : int
                Number of upsampling blocks.

            factor : int or list of int
                If int, the total upsampling factor for all stages combined.
                If list, upsampling factors for each stage.

            skip : bool
                Whether to concatenate upsampled tensor with stored pre-downsample encoding.
            upsample : dict
                Parameters for upsampling (see :func:`~.layers.upsample`).

            blocks : dict
                Parameters for post-processing blocks:

                base : callable
                    Tensor processing function. Default is :func:`~.layers.conv_block`.
                other args : dict
                    Parameters for the base block.

    Examples
    --------
    Use ResNet as an encoder with desired number of blocks and filters in them (total downsampling factor is 4),
    create an embedding that contains 256 channels, then upsample it to get 8 times the size of initial image.

    >>> config = {
            'inputs': dict(images={'shape': B('image_shape')},
                           masks={'name': 'targets', 'shape': B('mask_shape')}),
            'initial_block/inputs': 'images',
            'body/encoder': {'base': ResNet,
                             'num_blocks': [2, 3, 4]
                             'filters': [16, 32, 128]},
            'body/embedding': {'layout': 'cna', 'filters': 256},
            'body/decoder': {'num_stages': 5, 'factor': 32},
        }

    Preprocess input image with 7x7 convolutions, downsample it 5 times with DenseNet blocks in between,
    use MobileNet block in the bottom, then restore original image size with subpixel convolutions and
    ResNeXt blocks in between:

    >>> config = {
            'inputs': dict(images={'shape': B('image_shape')},
                           masks={'name': 'targets', 'shape': B('mask_shape')}),
            'initial_block': {'inputs': 'images',
                              'layout': 'cna', 'filters': 4, 'kernel_size': 7},
            'body/encoder': {'num_stages': 5,
                             'blocks': {'base': DenseNet.block,
                                        'num_layers': [2, 2, 3, 4, 5],
                                        'growth_rate': 6, 'skip': True}},
            'body/embedding': {'base': MobileNet.block,
                               'width_factor': 2},
            'body/decoder': {'upsample': {'layout': 'X'},
                             'blocks': {'base': ResNet.block,
                                        'filters': [256, 128, 64, 32, 16],
                                        'resnext': True}},
        }

    Notes
    -----
    When `base` is used for decoder creation, downsampling is done one less time than
    the length of `filters` (or other size-defining parameter) list in the `encoder` configuration.
    That is due to the fact that the first block is used as preprocessing of input tensors.
    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()
        config['body/encoder'] = dict(base=None, num_stages=None)
        config['body/encoder/downsample'] = dict(layout='p', pool_size=2, pool_strides=2)
        config['body/encoder/blocks'] = dict(base=cls.block)
        config['body/embedding'] = dict(base=cls.block)
        config['body/decoder'] = dict(skip=True, num_stages=None, factor=None)
        config['body/decoder/upsample'] = dict(layout='tna')
        config['body/decoder/blocks'] = dict(base=cls.block)
        config['head'] = dict(layout='c', kernel_size=1)
        return config


    def build_config(self, names=None):
        config = super().build_config(names)
        config['head/targets'] = self.get_from_attr('targets')
        return config


    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Create encoder, embedding and decoder. """
        kwargs = cls.fill_params('body', **kwargs)
        encoder = kwargs.pop('encoder')
        embedding = kwargs.pop('embedding')
        decoder = kwargs.pop('decoder')

        with tf.variable_scope(name):
            # Encoder: transition down
            encoder_outputs = cls.encoder(inputs, **encoder, **kwargs)

            # Bottleneck: working with compressed representation
            x = cls.embedding(encoder_outputs[-1], **embedding, **kwargs)
            encoder_outputs.append(x)

            # Decoder: transition up
            x = cls.decoder(encoder_outputs, **decoder, **kwargs)
        return x


    @classmethod
    def head(cls, inputs, targets, name='head', **kwargs):
        """ Linear convolutions. """
        kwargs = cls.fill_params('head', **kwargs)
        with tf.variable_scope(name):
            x = cls.crop(inputs, targets, kwargs['data_format'])
            channels = cls.num_channels(targets)
            x = conv_block(x, filters=channels, **kwargs)
        return x


    @classmethod
    def block(cls, inputs, name='block', **kwargs):
        """ Default conv block for processing tensors in encoder and decoder.
        By default makes 3x3 convolutions followed by batch-norm and activation.
        Does not change tensor shapes.
        """
        layout = kwargs.pop('layout', None) or 'cna'
        filters = kwargs.pop('filters', None) or cls.num_channels(inputs)
        return conv_block(inputs, layout=layout, filters=filters, name=name, **kwargs)


    @classmethod
    def encoder(cls, inputs, name='encoder', **kwargs):
        """ Create encoder either by using ``make_encoder`` of passed `base` model,
        or by combining building blocks, specified in `blocks/base`.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.

        base : TFModel
            Model class. Should implement ``make_encoder`` method.

        name : str
            Scope name.

        num_stages : int
            Number of downsampling stages.

        blocks : dict
            Parameters for tensor processing before downsampling.

        downsample : dict
            Parameters for downsampling.

        kwargs : dict
            Parameters for ``make_encoder`` method.

        Returns
        -------
        list of tf.Tensors
        """
        base_class = kwargs.pop('base')
        steps, downsample, block_args = cls.pop(['num_stages', 'downsample', 'blocks'], kwargs)

        if base_class is not None:
            encoder_outputs = base_class.make_encoder(inputs, name=name, **kwargs)

        else:
            base_block = block_args.get('base')
            with tf.variable_scope(name):
                x = inputs
                encoder_outputs = [x]

                for i in range(steps):
                    with tf.variable_scope('encoder-'+str(i)):
                        # Preprocess tensor with given block
                        args = {key: value[i] for key, value in block_args.items()
                                if isinstance(value, list)}
                        args = {**kwargs, **block_args, **args} # enforce priority of keys
                        x = base_block(x, name='pre', **args)

                        # Downsampling
                        x = conv_block(x, **{**kwargs, **downsample})
                        encoder_outputs.append(x)
        return encoder_outputs


    @classmethod
    def embedding(cls, inputs, name='embedding', **kwargs):
        """ Create embedding from inputs tensor.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.

        name : str
            Scope name.

        base : callable
            Tensor processing function. Default is :func:`~.layers.conv_block`.

        kwargs : dict
            Parameters for `base` block.

        Returns
        -------
        tf.Tensor
        """
        base_block = kwargs.get('base')
        return base_block(inputs, name=name, **kwargs)


    @classmethod
    def decoder(cls, inputs, name='decoder', **kwargs):
        """ Create decoder with a given number of upsampling stages.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.

        name : str
            Scope name.

        steps : int
            Number of upsampling stages. Defaults to the number of downsamplings.

        factor : int or list of ints
            If int, the total upsampling factor for all stages combined.
            If list, upsampling factors for each stage.s, then each entry is increase of size on i-th upsampling stage.

        skip : bool
            Whether to concatenate upsampled tensor with stored pre-downsample encoding.

        upsample : dict
            Parameters for upsampling.

        blocks : dict
            Parameters for post-processing blocks.

        kwargs : dict
            Parameters for ``upsample`` method.

        Returns
        -------
        tf.Tensor

        Raises
        ------
        TypeError
            If passed `factor` is not integer or list.
        """
        steps = kwargs.pop('num_stages') or len(inputs)-2
        factor = kwargs.pop('factor') or [2]*steps
        skip, upsample, block_args = cls.pop(['skip', 'upsample', 'blocks'], kwargs)
        base_block = block_args.get('base')

        if isinstance(factor, int):
            factor = int(factor ** (1/steps))
            factor = [factor] * steps
        elif not isinstance(factor, list):
            raise TypeError('factor should be int or list of int, but %s was given' % type(factor))

        with tf.variable_scope(name):
            x = inputs[-1]

            axis = cls.channels_axis(kwargs.get('data_format'))
            for i in range(steps):
                with tf.variable_scope('decoder-'+str(i)):
                    # Upsample by a desired factor
                    x = cls.upsample(x, factor=factor[i], name='upsample', **{**kwargs, **upsample})

                    # Post-process resulting tensor
                    args = {key: value[i] for key, value in block_args.items()
                            if isinstance(value, list)}
                    args = {**kwargs, **block_args, **args} # enforce priority of subkeys
                    x = base_block(x, name='post', **args)

                    # Concatenate it with stored encoding of the ~same shape
                    if skip and (i < len(inputs)-2):
                        x = cls.crop(x, inputs[-i-3], data_format=kwargs.get('data_format'))
                        x = tf.concat((x, inputs[-i-3]), axis=axis, name='skip-concat')
        return x


class AutoEncoder(EncoderDecoder):
    """ Model without skip-connections between corresponding stages of encoder and decoder. """
    @classmethod
    def default_config(cls):
        config = EncoderDecoder.default_config()
        config['body/decoder'] = dict(skip=False)
        return config


class VariationalAutoEncoder(AutoEncoder):
    """ Autoencoder that maps input into distribution. Based on
    Kingma, Diederik P; Welling, Max "`Auto-Encoding Variational Bayes
    <https://arxiv.org/abs/1312.6114>`_"

    Notes
    -----
    Distribution that is learned is always normal.
    """
    @classmethod
    def embedding(cls, inputs, name='embedding', **kwargs):
        """ Create embedding from inputs tensor.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.

        name : str
            Scope name.

        base : callable
            Tensor processing function. Default is :func:`~.layers.conv_block`.

        kwargs : dict
            Parameters for `base` block.

        Returns
        -------
        tf.Tensor
        """
        base_block = kwargs.get('base')

        mean = base_block(inputs, name='mean', **kwargs)
        std = base_block(inputs, name='std', **kwargs)
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + eps*std
