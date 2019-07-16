""" Contains convolution layers """
import numpy as np
import tensorflow as tf

from .resize import resize_bilinear_additive, resize_bilinear, resize_nn, subpixel_conv
from .block import _conv_block, _update_layers


_NEW_LAYERS = {
    'A': 'residual_bilinear_additive',
    'b': 'resize_bilinear',
    'B': 'resize_bilinear_additive',
    'N': 'resize_nn',
    'X': 'subpixel_conv'
}

_NEW_FUNCS = {
    'residual_bilinear_additive': None,
    'resize_bilinear': resize_bilinear,
    'resize_bilinear_additive': resize_bilinear_additive,
    'resize_nn': resize_nn,
    'subpixel_conv': subpixel_conv
}

_NEW_GROUPS = {'A': 'b', 'B': 'b', 'N': 'b', 'X': 'b'}

_update_layers(_NEW_LAYERS, _NEW_FUNCS, _NEW_GROUPS)


def conv_block(inputs, layout='', filters=0, kernel_size=3, name=None,
               strides=1, padding='same', data_format='channels_last', dilation_rate=1, depth_multiplier=1,
               activation=tf.nn.relu, pool_size=2, pool_strides=2, dropout_rate=0., is_training=True, **kwargs):
    """ Complex multi-dimensional block with a sequence of convolutions, batch normalization, activation, pooling,
    dropout and even dense layers.

    Parameters
    ----------
    inputs : tf.Tensor
        input tensor
    layout : str
        a sequence of operations:

        - c - convolution
        - t - transposed convolution
        - C - separable convolution
        - T - separable transposed convolution
        - f - dense (fully connected)
        - n - batch normalization
        - a - activation
        - p - pooling (default is max-pooling)
        - v - average pooling
        - P - global pooling (default is max-pooling)
        - V - global average pooling
        - d - dropout
        - D - dropblock
        - m - maximum intensity projection (:func:`~.layers.mip`)
        - b - upsample with bilinear resize
        - B - upsample with bilinear additive resize
        - N - upsample with nearest neighbors resize
        - X - upsample with subpixel convolution (:func:`~.layers.subpixel_conv`)
        - R - start residual connection
        - A - start residual connection with bilinear additive upsampling
        - `+` - end residual connection with summation
        - `.` - end residual connection with concatenation

        Default is ''.
    filters : int
        the number of filters in the output tensor
    kernel_size : int
        kernel size
    name : str
        name of the layer that will be used as a scope.
    units : int
        the number of units in the dense layer
    strides : int
        Default is 1.
    padding : str
        padding mode, can be 'same' or 'valid'. Default - 'same',
    data_format : str
        'channels_last' or 'channels_first'. Default - 'channels_last'.
    dilation_rate: int
        Default is 1.
    activation : callable
        Default is `tf.nn.relu`.
    pool_size : int
        Default is 2.
    pool_strides : int
        Default is 2.
    pool_op : str
        pooling operation ('max', 'mean', 'frac')
    dropout_rate : float
        Default is 0.
    factor : int or tuple of int
        upsampling factor
    upsampling_layout : str
        layout for upsampling layers
    is_training : bool or tf.Tensor
        Default is True.
    reuse : bool
        whether to user layer variables if exist

    dense : dict
        parameters for dense layers, like initializers, regularalizers, etc
    conv : dict
        parameters for convolution layers, like initializers, regularalizers, etc
    transposed_conv : dict
        parameters for transposed conv layers, like initializers, regularalizers, etc
    batch_norm : dict or None
        parameters for batch normalization layers, like momentum, intiializers, etc
        If None or inculdes parameters 'off' or 'disable' set to True or 1,
        the layer will be excluded whatsoever.
    pooling : dict
        parameters for pooling layers, like initializers, regularalizers, etc
    dropout : dict or None
        parameters for dropout layers, like noise_shape, etc
        If None or inculdes parameters 'off' or 'disable' set to True or 1,
        the layer will be excluded whatsoever.
    dropblock : dict or None
        parameters for dropblock layers, like dropout_rate, block_size, etc
    subpixel_conv : dict or None
        parameters for subpixel convolution (layout, activation, etc)
    resize_bilinear : dict or None
        parameters for bilinear resize
    resize_bilinear_additive : dict or None
        parameters for bilinear additive resize (layout, activation, etc)

    Returns
    -------
    output tensor : tf.Tensor

    Notes
    -----
    When ``layout`` includes several layers of the same type, each one can have its own parameters,
    if corresponding args are passed as lists (not tuples).

    Spaces may be used to improve readability.


    Examples
    --------
    A simple block: 3x3 conv, batch norm, relu, 2x2 max-pooling with stride 2::

        x = conv_block(x, 'cnap', filters=32, kernel_size=3)

    A canonical bottleneck block (1x1, 3x3, 1x1 conv with relu in-between)::

        x = conv_block(x, 'nac nac nac', [64, 64, 256], [1, 3, 1])

    A complex Nd block:

    - 5x5 conv with 32 filters
    - relu
    - 3x3 conv with 32 filters
    - relu
    - 3x3 conv with 64 filters and a spatial stride 2
    - relu
    - batch norm
    - dropout with rate 0.15

    ::

        x = conv_block(x, 'ca ca ca nd', [32, 32, 64], [5, 3, 3], strides=[1, 1, 2], dropout_rate=.15)

    A residual block::

        x = conv_block(x, 'R nac nac +', [16, 16, 64], [1, 3, 1])

    """
    tensor = _conv_block(inputs, layout, filters, kernel_size, name,
                         strides, padding, data_format, dilation_rate, depth_multiplier,
                         activation, pool_size, pool_strides, dropout_rate, is_training,
                         **kwargs)
    return tensor


def upsample(inputs, factor=None, shape=None, layout='b', name='upsample', **kwargs):
    """ Upsample inputs with a given factor

    Parameters
    ----------
    inputs : tf.Tensor
        a tensor to resize
    factor : int
        an upsamping scale
    shape : tuple of int
        a shape to upsample to (used by bilinear and NN resize)
    layout : str
        resizing technique, a sequence of:

        - A - use residual connection with bilinear additive upsampling
        - b - bilinear resize
        - B - bilinear additive upsampling
        - N - nearest neighbor resize
        - t - transposed convolution
        - T - separable transposed convolution
        - X - subpixel convolution

        all other :func:`.conv_block` layers are also allowed.

    Returns
    -------
    tf.Tensor

    Examples
    --------
    A simple bilinear upsampling::

        x = upsample(inputs, shape=(256, 256), layout='b')

    Upsampling with non-linear normalized transposed convolution::

        x = upsample(inputs, factor=2, layout='nat', kernel_size=3)

    Subpixel convolution with a residual bilinear additive connection::

        x = upsample(inputs, factor=2, layout='AX+')
    """
    if np.all(factor == 1):
        return inputs

    if 't' in layout or 'T' in layout:
        if 'kernel_size' not in kwargs:
            kwargs['kernel_size'] = factor
        if 'strides' not in kwargs:
            kwargs['strides'] = factor

    x = conv_block(inputs, layout, name=name, factor=factor, shape=shape, **kwargs)

    return x
