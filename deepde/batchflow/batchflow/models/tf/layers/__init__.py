""" Custom tf layers and operations """
from .core import flatten, flatten2d, maxout, mip, xip, alpha_dropout
from .conv_block import conv_block, upsample
from .conv import conv1d_transpose, conv1d_transpose_nn, conv_transpose, separable_conv, separable_conv_transpose
from .pooling import max_pooling, average_pooling, pooling, \
                     global_pooling, global_average_pooling, global_max_pooling, \
                     fractional_pooling
from .roi import roi_pooling_layer, non_max_suppression
from .resize import subpixel_conv, resize_bilinear_additive, resize_nn, resize_bilinear, depth_to_space
from .pyramid import pyramid_pooling, aspp
from .drop_block import dropblock
