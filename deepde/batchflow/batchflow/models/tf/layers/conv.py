""" Contains convolutional layers """
import tensorflow as tf

CONV_LAYERS = {
    1: tf.layers.conv1d,
    2: tf.layers.conv2d,
    3: tf.layers.conv3d
}

def conv(inputs, *args, **kwargs):
    """ Nd convolution layer. Just a wrapper around ``tf.layers.conv1d``, ``conv2d``, ``conv3d``.

    Parameters
    ----------
    inputs : tf.Tensor
        input tensor

    See also
    --------
    `tf.layers.conv1d <https://www.tensorflow.org/api_docs/python/tf/layers/conv1d>`_,
    `tf.layers.conv2d <https://www.tensorflow.org/api_docs/python/tf/layers/conv2d>`_,
    `tf.layers.conv3d <https://www.tensorflow.org/api_docs/python/tf/layers/conv3d>`_
    """
    dim = inputs.shape.ndims - 2
    layer_fn = CONV_LAYERS[dim]
    return layer_fn(inputs, *args, **kwargs)

def conv1d_transpose(inputs, filters, kernel_size, strides=1, padding='valid', data_format='channels_last',
                     **kwargs):
    """ Transposed 1D convolution layer

    Parameters
    ----------
    inputs : tf.Tensor
        input tensor
    filters : int
        number of filters in the ouput tensor
    kernel_size : int
        kernel size
    strides : int
        convolution stride. Default is 1.

    Returns
    -------
    tf.Tensor

    See also
    --------
    `tf.layers.conv2d_transpose <https://www.tensorflow.org/api_docs/python/tf/layers/conv2d_transpose>`_,
    `tf.layers.conv3d_transpose <https://www.tensorflow.org/api_docs/python/tf/layers/conv3d_transpose>`_
    """
    axis = 1 if data_format == 'channels_last' else 2
    x = tf.expand_dims(inputs, axis=axis)
    x = tf.layers.conv2d_transpose(x, filters=filters, kernel_size=(1, kernel_size),
                                   strides=(1, strides), padding=padding, **kwargs)
    x = tf.squeeze(x, [axis])
    return x


def conv1d_transpose_nn(value, filters, output_shape, strides,
                        padding='SAME', data_format='NWC', name=None):
    """ Transposed 1D convolution layer. Analogue of the tf.nn.conv2d_transpose.

    Parameters
    ----------
    value : tf.Tensor
        input tensor
    filters : tf.Tensor
        convolutional filter
    output_shape : tf.Tensor
        the output shape of the deconvolution op
    strides : list
        the stride of the sliding window for each dimension of the input tensor
    padding : str
        'VALID' or 'SAME'. Default - 'SAME'.
    data_format : str
        'NWC' or 'NCW'. Default - 'NWC'.
    name : str
        scope name

    Returns
    -------
    tf.Tensor

    See also
    --------
    `tf.nn.conv2d_transpose <https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose>`_,
    `tf.nn.conv3d_transpose <https://www.tensorflow.org/api_docs/python/tf/nn/conv3d_transpose>`_
    """
    axis = 1 if data_format == 'NWC' else 2
    value = tf.expand_dims(value, axis=axis)
    filters = tf.expand_dims(filters, axis=0)
    output_shape = tf.concat([output_shape[:axis], (1, ), output_shape[axis:]], axis=-1)
    strides = strides[:axis] + [1] + strides[axis:]
    x = tf.nn.conv2d_transpose(value, filters, output_shape, strides,
                               padding, data_format, name)
    x = tf.squeeze(x, [axis])
    return x

def conv_transpose(inputs, filters, kernel_size, strides, *args, **kwargs):
    """ Transposed Nd convolution layer

    Parameters
    ----------
    inputs : tf.Tensor
        input tensor
    filters : int
        number of filters in the ouput tensor
    kernel_size : int
        kernel size
    strides : int
        convolution stride. Default is 1.

    Returns
    -------
    tf.Tensor

    See also
    --------
    :func:`.conv1d_transpose`,
    `tf.layers.conv2d_transpose <https://www.tensorflow.org/api_docs/python/tf/layers/conv2d_transpose>`_,
    `tf.layers.conv3d_transpose <https://www.tensorflow.org/api_docs/python/tf/layers/conv3d_transpose>`_
    """
    dim = inputs.shape.ndims - 2
    if dim == 1:
        output = conv1d_transpose(inputs, filters, kernel_size, strides, *args, **kwargs)
    elif dim == 2:
        output = tf.layers.conv2d_transpose(inputs, filters, kernel_size, strides, *args, **kwargs)
    elif dim == 3:
        output = tf.layers.conv3d_transpose(inputs, filters, kernel_size, strides, *args, use_bias=False, **kwargs)
    return output


def _separable_conv(transpose, inputs, filters, kernel_size, strides=1, padding='same', data_format='channels_last',
                    dilation_rate=1, depth_multiplier=1, activation=None, name=None, **kwargs):
    dim = inputs.shape.ndims - 2
    context = None
    if name is not None:
        context = tf.variable_scope(name)
        context.__enter__()
    if transpose:
        conv_layer = conv_transpose
    else:
        conv_layer = conv

    kwargs = {'kernel_size': kernel_size,
              'strides': strides,
              'padding': padding,
              'data_format': data_format,
              'activation': activation,
              **kwargs}

    if not transpose:
        kwargs['dilation_rate'] = dilation_rate

    inputs_shape = inputs.get_shape().as_list()
    axis = -1 if data_format == 'channels_last' else 1
    size = [-1] * (dim + 2)
    size[axis] = 1
    channels_in = inputs_shape[axis]

    depthwise_layers = []
    for channel in range(channels_in):
        start = [0] * (dim + 2)
        start[axis] = channel

        input_slice = tf.slice(inputs, start, size)

        _kwargs = {**kwargs, 'inputs': input_slice, 'filters': depth_multiplier, 'name': 'slice-%d' % channel}

        slice_conv = conv_layer(**_kwargs)
        depthwise_layers.append(slice_conv)

    # Concatenate the per-channel convolutions along the channel dimension.
    depthwise_conv = tf.concat(depthwise_layers, axis=axis)

    if channels_in * depth_multiplier != filters:
        _kwargs = {**kwargs,
                   'inputs': depthwise_conv,
                   'filters': filters,
                   'kernel_size': 1,
                   'strides': 1,
                   'dilation_rate': 1,
                   'name': 'pointwise'}
        output = conv(**_kwargs)
    else:
        output = depthwise_conv

    if context is not None:
        context.__exit__(None, None, None)

    return output

def separable_conv(inputs, *args, **kwargs):
    """ Make Nd depthwise convolutions that acts separately on channels,
    followed by a pointwise convolution that mixes channels.

    Parameters
    ----------
    inputs : tf.Tensor
        input tensor
    filters : int
        number of filters in the ouput tensor
    kernel_size : int
        kernel size
    strides : int
        convolution stride. Default is 1.
    padding : str
        padding mode, can be 'same' or 'valid'. Default - 'same',
    data_format : str
        'channels_last' or 'channels_first'. Default - 'channels_last'.
    dilation_rate : int
        Default is 1.
    depth_multiplier : int
        The number of depthwise convolution output channels for each input channel.
        The total number of depthwise convolution output channels will be equal to
        ``num_filters_in`` * ``depth_multiplier``. Default - 1.
    activation : callable
        Default is `tf.nn.relu`.
    name : str
        The name of the layer. Default - None.

    Returns
    -------
    tf.Tensor

    """
    dim = inputs.shape.ndims - 2

    if dim == 2:
        return tf.layers.separable_conv2d(inputs, *args, **kwargs)
    return _separable_conv(False, inputs, *args, **kwargs)

def separable_conv_transpose(inputs, *args, **kwargs):
    """ Make Nd depthwise transpose convolutions that acts separately on channels,
    followed by a pointwise convolution that mixes channels.

    Parameters
    ----------
    inputs : tf.Tensor
        input tensor
    filters : int
        number of filters in the ouput tensor
    kernel_size : int
        kernel size
    strides : int
        convolution stride. Default is 1.
    padding : str
        padding mode, can be 'same' or 'valid'. Default - 'same'.
    data_format : str
        'channels_last' or 'channels_first'. Default - 'channels_last'.
    depth_multiplier : int
        The number of depthwise convolution output channels for each input channel.
        The total number of depthwise convolution output channels will be equal to
        ``num_filters_in`` * ``depth_multiplier``. Deafault - 1.
    activation : callable
        Default is `tf.nn.relu`.
    name : str
        The name of the layer. Default - None.

    Returns
    -------
    tf.Tensor

    """
    output = _separable_conv(True, inputs, *args, **kwargs)
    return output
