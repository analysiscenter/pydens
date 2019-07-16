""" Contains upsampling and resize layers """
import numpy as np
import tensorflow as tf

from .block import _conv_block as conv_block
from .conv import conv1d_transpose_nn
from .core import xip


def _calc_size(inputs, factor, data_format):
    shape = inputs.get_shape().as_list()
    channels = shape[-1] if data_format == 'channels_last' else shape[1]
    if None in shape[1:]:
        shape = _dynamic_calc_shape(inputs, factor, data_format)
    else:
        shape = _static_calc_shape(inputs, factor, data_format)
    return shape, channels

def _static_calc_shape(inputs, factor, data_format):
    shape = inputs.get_shape().as_list()
    shape = shape[1:-1] if data_format == 'channels_last' else shape[2:]
    shape = np.asarray(shape) * np.asarray(factor)
    shape = list(np.ceil(shape).astype(np.int32))
    return shape

def _dynamic_calc_shape(inputs, factor, data_format):
    shape = tf.cast(tf.shape(inputs), dtype=tf.float32)
    shape = shape[1:-1] if data_format == 'channels_last' else shape[2:]
    shape = shape * np.asarray(factor)
    shape = tf.cast(tf.ceil(shape), dtype=tf.int32)
    return shape


def depth_to_space(inputs, block_size, name='d2s', data_format='channels_last'):
    """ 1d, 2d and 3d depth_to_space transformation.

    Parameters
    ----------
    inputs : tf.Tensor
        a tensor to resize
    block_size : int
        An int that is >= 2. The size of the spatial block
    name : str
        scope name
    data_format : {'channels_last', 'channels_first'}
        position of the channels dimension

    Returns
    -------
    tf.Tensor

    See also
    --------
    `tf.depth_to_space <https://www.tensorflow.org/api_docs/python/tf/depth_to_space>`_
    """
    dim = inputs.shape.ndims - 2
    if dim == 2:
        dafo = 'NHWC' if data_format == 'channels_last' else 'NCHW'
        return tf.depth_to_space(inputs, block_size, name, data_format=dafo)

    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0] + list(range(2, dim+2)) + [1])
    x = _depth_to_space(inputs, block_size, name)
    if data_format == 'channels_first':
        x = tf.transpose(x, [0, dim+1] + list(range(1, dim+1)))
    return x


def _depth_to_space(inputs, block_size, name='d2s'):
    dim = inputs.shape.ndims - 2
    if dim == 1:
        conv_layer = conv1d_transpose_nn
    elif dim == 2:
        conv_layer = tf.nn.conv2d_transpose
    elif dim == 3:
        conv_layer = tf.nn.conv3d_transpose

    with tf.variable_scope(name):
        shape = inputs.get_shape().as_list()[1:]
        channels = shape[-1]
        if channels % (block_size ** dim) != 0:
            raise ValueError('channels of the inputs must be divisible by block_size ** {}'.format(dim))
        output_shape = tf.concat([(tf.shape(inputs)[0],), tf.shape(inputs)[1:-1]*block_size,
                                  (tf.shape(inputs)[-1], )], axis=-1)
        slices = [np.arange(0, channels // (block_size ** dim)) + i
                  for i in range(0, channels, channels // (block_size ** dim))]
        tensors = []
        for i in range(block_size ** dim):
            zero_filter = np.zeros(block_size ** dim)
            selective_filter = np.zeros(block_size ** dim)
            selective_filter[i] = 1
            zero_filter = zero_filter.reshape([block_size] * dim)
            selective_filter = selective_filter.reshape([block_size] * dim)
            fltr = []
            for j in range(channels):
                _filter = [zero_filter] * channels
                _filter[j] = selective_filter
                _filter = np.stack(_filter, axis=-1)
                fltr.append(_filter)
            fltr = np.stack(fltr, axis=-1)
            fltr = np.transpose(fltr, axes=list(range(dim))+[dim, dim+1])
            fltr = tf.constant(fltr, tf.float32)
            x = conv_layer(inputs, fltr, output_shape, [1] + [block_size] * dim + [1])
            if None in shape[:-1]:
                resized_shape = shape[:-1]
            else:
                resized_shape = list(np.array(shape[:-1]) * block_size)
            x.set_shape([None] + resized_shape + [channels/(block_size ** dim)])
            x = tf.gather(x, slices[i], axis=-1)
            tensors.append(x)
        x = tf.add_n(tensors)
    return x


def subpixel_conv(inputs, factor=2, name='subpixel', data_format='channels_last', **kwargs):
    """ Resize input tensor with subpixel convolution (depth to space operation)

    Parameters
    ----------
    inputs : tf.Tensor
        a tensor to resize
    factor : int
        upsampling factor
    layout : str
        layers applied before depth-to-space transform
    name : str
        scope name
    data_format : {'channels_last', 'channels_first'}
        position of the channels dimension

    Returns
    -------
    tf.Tensor
    """
    dim = inputs.shape.ndims - 2

    _, channels = _calc_size(inputs, factor, data_format)
    layout = kwargs.pop('layout', 'cna')
    kwargs['filters'] = channels*factor**dim

    x = inputs
    with tf.variable_scope(name):
        if layout:
            x = conv_block(inputs, layout, kernel_size=1, name='conv', data_format=data_format, **kwargs)
        x = depth_to_space(x, block_size=factor, name='d2s', data_format=data_format)
    return x


def resize_bilinear_additive(inputs, factor=2, name='bilinear_additive', data_format='channels_last', **kwargs):
    """ Resize input tensor with bilinear additive technique

    Parameters
    ----------
    inputs : tf.Tensor
        a tensor to resize
    factor : int
        upsampling factor
    layout : str
        layers applied between bilinear resize and xip
    name : str
        scope name
    data_format : {'channels_last', 'channels_first'}
        position of the channels dimension

    Returns
    -------
    tf.Tensor
    """
    dim = inputs.shape.ndims - 2
    _, channels = _calc_size(inputs, factor, data_format)
    layout = kwargs.pop('layout', 'cna')
    with tf.variable_scope(name):
        x = resize_bilinear(inputs, factor, name=name, data_format=data_format, **kwargs)
        x = conv_block(x, layout, filters=channels*factor**dim, kernel_size=1, name='conv', **kwargs)
        x = xip(x, depth=factor**dim, reduction='sum', name='addition')
    return x

def resize_bilinear_1d(inputs, size, name='resize', **kwargs):
    """ Resize 1D input tensor with bilinear method.

    Parameters
    ----------
    inputs : tf.Tensor
        a tensor to resize
    size : tf.Tensor or list
        size of the output image
    name : str
        scope name

    Returns
    -------
    tf.Tensor
    """
    x = tf.expand_dims(inputs, axis=1)
    size = tf.concat([[1], size], axis=-1)
    x = tf.image.resize_bilinear(x, size=size, name=name, **kwargs)
    x = tf.squeeze(x, [1])
    return x

def resize_bilinear_3d(tensor, size, name='resize', **kwargs):
    """ Resize 3D input tensor with bilinear method.

    Parameters
    ----------
    inputs : tf.Tensor
        a tensor to resize
    size : tf.Tensor or list
        size of the output image
    name : str
        scope name

    Returns
    -------
    tf.Tensor
    """
    with tf.variable_scope(name):
        tensor = _resize_along_axis(tensor, size, 2, **kwargs)
        tensor = _resize_except_axis(tensor, size, 2, **kwargs)
    return tensor

def _resize_along_axis(inputs, size, axis, **kwargs):
    """ Resize 3D input tensor to size along just one axis. """
    except_axis = (axis + 1) % 3
    size, _ = _calc_size_after_resize(inputs, size, axis)
    output = _resize_except_axis(inputs, size, except_axis, **kwargs)
    return output

def _resize_except_axis(inputs, size, axis, **kwargs):
    """ Resize 3D input tensor to size except just one axis. """
    perm = np.arange(5)
    reverse_perm = np.arange(5)

    if axis == 0:
        spatial_perm = [2, 3, 1]
        reverse_spatial_perm = [3, 1, 2]
    elif axis == 1:
        spatial_perm = [1, 3, 2]
        reverse_spatial_perm = [1, 3, 2]
    else:
        spatial_perm = [1, 2, 3]
        reverse_spatial_perm = [1, 2, 3]

    perm[1:4] = spatial_perm
    reverse_perm[1:4] = reverse_spatial_perm
    x = tf.transpose(inputs, perm)

    if isinstance(size, tf.Tensor):
        size = tf.unstack(size)
        size = [size[i-1] for i in spatial_perm]
        size = tf.stack(size)
    else:
        size = [size[i-1] for i in spatial_perm]

    real_size, static_shape = _calc_size_after_resize(x, size, [0, 1])
    real_size = size[:-1]
    array = tf.TensorArray(tf.float32, size=tf.shape(x)[-2])
    partial_sl = [slice(None)] * 5

    def _loop(idx, array):
        partial_sl[-2] = idx
        tensor = x[partial_sl]
        tensor = tf.image.resize_bilinear(tensor, size=real_size, name='resize_2d', **kwargs)
        array = array.write(idx, tensor)
        return [idx+1, array]
    i = 0
    _, array = tf.while_loop(lambda i, array: i < tf.shape(x)[-2], _loop, [i, array])
    array = array.stack()
    array = tf.transpose(array, [1, 2, 3, 0, 4])
    array.set_shape(static_shape)
    array = tf.transpose(array, reverse_perm)
    return array

def _calc_size_after_resize(inputs, size, axis):
    if not isinstance(axis, list):
        axis = [axis]
    except_axis = list(set(range(3)) - set(axis))
    if isinstance(size, tf.Tensor):
        size = tf.unstack(size)
        for i in except_axis:
            size[i] = tf.shape(inputs)[i+1]
        size = tf.stack(size)
        static_size = [None] * 4 + [inputs.get_shape().as_list()[-1]]
    else:
        size = size[:]
        static_size = inputs.get_shape().as_list()
        if None in static_size[1:]:
            size[except_axis] = tf.shape(inputs)[except_axis+1]
            size = tf.stack(size)
        else:
            for i in except_axis:
                size[i] = static_size[i+1]
            static_size[1:4] = size
    return size, static_size


def resize_bilinear(inputs, factor=2, shape=None, name='resize', data_format='channels_last', **kwargs):
    """ Resize input tensor with bilinear method

    Parameters
    ----------
    inputs : tf.Tensor
        a tensor to resize
    factor : float
        upsampling factor (not used if shape is specified)
    shape : tuple of int
        a shape to upsample to
    name : str
        scope name
    data_format : {'channels_last', 'channels_first'}
        position of the channels dimension

    Returns
    -------
    tf.Tensor
    """
    if shape is None:
        shape, _ = _calc_size(inputs, factor, data_format)

    with tf.variable_scope(name):
        x = inputs
        if data_format == 'channels_first':
            perm = [0] + list(range(2, inputs.shape.ndims)) + [1]
            perm_reverse = [0, inputs.shape.ndims-1] + list(range(1, inputs.shape.ndims-1))
            x = tf.transpose(x, perm)
        dim = inputs.shape.ndims - 2
        if dim == 1:
            x = resize_bilinear_1d(x, size=shape, name='resize_1d', **kwargs)
        elif dim == 2:
            x = tf.image.resize_bilinear(x, size=shape, name='resize_2d', **kwargs)
        elif dim == 3:
            x = resize_bilinear_3d(x, size=shape, name='resize_3d', **kwargs)
        if data_format == 'channels_first':
            x = tf.transpose(x, perm_reverse)
    return x


def resize_nn(inputs, factor=2, shape=None, name=None, data_format='channels_last', **kwargs):
    """ Resize input tensor with nearest neighbors method

    Parameters
    ----------
    inputs : tf.Tensor
        a tensor to resize
    factor : int
        upsampling factor (not used if shape is specified)
    shape : tuple of int
        a shape to upsample to
    name : str
        scope name
    data_format : {'channels_last', 'channels_first'}
        position of the channels dimension

    Returns
    -------
    tf.Tensor
    """
    dim = inputs.shape.ndims
    if dim != 4:
        raise ValueError("inputs must be Tensor of rank 4 but {} was given".format(dim))
    if shape is None:
        shape, _ = _calc_size(inputs, factor, data_format)
    return tf.image.resize_nearest_neighbor(inputs, size=shape, name=name, **kwargs)
