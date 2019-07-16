""" Contains pyramid layers """
import numpy as np
import tensorflow as tf

from . import conv_block, upsample


def pyramid_pooling(inputs, layout='cna', filters=None, kernel_size=1, pool_op='mean', pyramid=(0, 1, 2, 3, 6),
                    name='psp', **kwargs):
    """ Pyramid Pooling module

    Zhao H. et al. "`Pyramid Scene Parsing Network <https://arxiv.org/abs/1612.01105>`_"

    Parameters
    ----------
    inputs : tf.Tensor
        input tensor
    layout : str
        layout for convolution layers
    filters : int
        the number of filters in each pyramid branch
    kernel_size : int
        kernel size
    pool_op : str
        a pooling operation ('mean' or 'max')
    pyramid : tuple of int
        the number of feature regions in each dimension, default is (0, 1, 2, 3, 6).

        `0` is used to include `inputs` into the output tensor.
    name : str
        a layer name that will be used as a scope.

    Returns
    -------
    tf.Tensor
    """
    shape = inputs.get_shape().as_list()
    data_format = kwargs.get('data_format', 'channels_last')
    axis = -1 if data_format == 'channels_last' else 1
    if filters is None:
        filters = shape[axis] // len(pyramid)

    with tf.variable_scope(name):
        if None in shape[1:]:
            # if some dimension is undefined
            raise ValueError("Pyramid pooling can only be applied to a tensor with a fully defined shape.")

        item_shape = np.array(shape[1: -1] if data_format == 'channels_last' else shape[2:])

        layers = []
        for level in pyramid:
            if level == 0:
                x = inputs
            else:
                pool_size = tuple(np.ceil(item_shape / level).astype(np.int32).tolist())
                pool_strides = tuple(np.floor((item_shape - 1) / level + 1).astype(np.int32).tolist())

                x = conv_block(inputs, 'p', pool_op=pool_op, pool_size=pool_size, pool_strides=pool_strides,
                               name='pool-%d' % level, **kwargs)
                x = conv_block(x, layout, filters=filters, kernel_size=kernel_size,
                               name='conv-%d' % level, **kwargs)
                x = upsample(x, layout='b', shape=item_shape, name='upsample-%d' % level, **kwargs)
            layers.append(x)
        x = tf.concat(layers, axis=axis, name='concat')
    return x


def aspp(inputs, layout='cna', filters=None, kernel_size=3, rates=(6, 12, 18), image_level_features=2,
         name='aspp', **kwargs):
    """ Atrous Spatial Pyramid Pooling module

    Chen L. et al. "`Rethinking Atrous Convolution for Semantic Image Segmentation
    <https://arxiv.org/abs/1706.05587>`_"

    Parameters
    ----------
    inputs : tf.Tensor
        input tensor
    layout : str
        layout for convolution layers
    filters : int
        the number of filters in the output tensor
    kernel_size : int
        kernel size for dilated branches (default=3)
    rates : tuple of int
        dilation rates for branches, default=(6, 12, 18)
    image_level_features : int or tuple of int
        the number of image level features in each dimension.

        Default is 2, i.e. 2x2=4 pooling features will be calculated for 2d images,
        and 2x2x2=8 features per 3d item.

        Tuple allows to define several image level features, e.g (2, 3, 4).
    name : str
        a layer name that will be used as a scope.

    See also
    --------
    pyramid_pooling

    Returns
    -------
    tf.Tensor
    """
    data_format = kwargs.get('data_format', 'channels_last')
    axis = -1 if data_format == 'channels_last' else 1
    if filters is None:
        filters = inputs.get_shape().as_list()[axis]
    if isinstance(image_level_features, int):
        image_level_features = (image_level_features,)

    with tf.variable_scope(name):
        x = conv_block(inputs, layout, filters=filters, kernel_size=1, name='conv-1x1', **kwargs)
        layers = [x]

        for level in rates:
            x = conv_block(inputs, layout, filters=filters, kernel_size=kernel_size, dilation_rate=level,
                           name='conv-%d' % level, **kwargs)
            layers.append(x)

        x = pyramid_pooling(inputs, filters=filters, pyramid=image_level_features,
                            name='image_level_features', **kwargs)
        layers.append(x)

        x = tf.concat(layers, axis=axis, name='concat')
        x = conv_block(x, layout, filters=filters, kernel_size=1, name='last_conv', **kwargs)
    return x
