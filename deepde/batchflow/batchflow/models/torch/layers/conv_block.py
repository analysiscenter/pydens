""" Contains convolution block """
import logging
import torch.nn as nn

from .core import Dense, Activation, \
                  Conv, ConvTranspose, SeparableConv, SeparableConvTranspose, \
                  Dropout, BatchNorm, Pool, GlobalPool, Interpolate, SubPixelConv
from ..utils import get_shape
from ...utils import unpack_args


logger = logging.getLogger(__name__)

FUNC_LAYERS = {
    'activation': Activation,
    'residual_start': None,
    'residual_end': None,
    'dense': Dense,
    'conv': Conv,
    'transposed_conv': ConvTranspose,
    'separable_conv': SeparableConv,
    'separable_conv_transpose': SeparableConvTranspose,
    'pooling': Pool,
    'global_pooling': GlobalPool,
    'batch_norm': BatchNorm,
    'dropout': Dropout,
    'alpha_dropout': nn.AlphaDropout,
    'resize_bilinear': Interpolate,
    'resize_nn': Interpolate,
    'subpixel_conv': SubPixelConv,
}


C_LAYERS = {
    'a': 'activation',
    'R': 'residual_start',
    '+': 'residual_end',
    '.': 'residual_end',
    'f': 'dense',
    'c': 'conv',
    't': 'transposed_conv',
    'C': 'separable_conv',
    'T': 'separable_conv_transpose',
    'p': 'pooling',
    'v': 'pooling',
    'P': 'global_pooling',
    'V': 'global_pooling',
    'n': 'batch_norm',
    'd': 'dropout',
    'D': 'alpha_dropout',
    'b': 'resize_bilinear',
    'N': 'resize_nn',
    'X': 'subpixel_conv',
}


LAYER_KEYS = ''.join(list(C_LAYERS.keys()))
GROUP_KEYS = (
    LAYER_KEYS
    .replace('t', 'c')
    .replace('C', 'c')
    .replace('T', 'c')
    .replace('v', 'p')
    .replace('V', 'P')
    .replace('D', 'd')
)

C_GROUPS = dict(zip(LAYER_KEYS, GROUP_KEYS))


class ConvBlock(nn.Module):
    """ Complex multi-dimensional block with a sequence of convolutions, batch normalization, activation, pooling,
    dropout, dense and other layers.

    Parameters
    ----------
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
        - D - alpha dropout
        - X - upsample with subpixel convolution (:func:`~.layers.SubPixelConv`)

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
    padding : str or int
        padding mode, can be 'same' or 'valid'. Default - 'same',
    dilation_rate: int
        Default is 1.
    depth_multipler : int
        Filters factor for separable convolutions
    activation : str or callable
        Default is 'relu'.
    pool_size : int
        Default is 2.
    pool_strides : int
        Default is 2.
    pool_op : str
        pooling operation ('max', 'mean', 'frac')
    dropout_rate : float
        Default is 0.
    units : int
        number of neurons in dense layers
    factor : int or tuple of int
        upsampling factor
    shape : tuple of int
        a shape to upsample to
    inputs : torch.Tensor, torch.nn.Module, numpy.ndarray or tuple
        shape or an example of input tensor to infer shape
    dense : dict
        common parameters for dense layers, like initializers, regularalizers, etc
    conv : dict
        common parameters for convolution layers, like initializers, regularalizers, etc
    transposed_conv : dict
        common parameters for transposed conv layers, like initializers, regularalizers, etc
    batch_norm : dict or None
        common parameters for batch normalization layers, like momentum, intiializers, etc
    pooling : dict
        common parameters for pooling layers, like initializers, regularalizers, etc
    dropout : dict or None
        common parameters for dropout layers, like noise_shape, etc

    Returns
    -------
    torch.nn.Module

    Notes
    -----
    When ``layout`` includes several layers of the same type, each one can have its own parameters,
    if corresponding args are passed as lists (not tuples).

    Spaces may be used to improve readability.

    If common layer parameters (dense, conv, etc) is set to False or includes a key 'disable' set to True,
    all the layers of that type will be excluded whatsoever.

    Such a feature comes in handy for analyzing various model architectures and configurations
    (see :class:`.Research`).

    Examples
    --------
    A simple block: 3x3 conv, batch norm, relu, 2x2 max-pooling with stride 2::

        x = ConvBlock('cnap', filters=32, kernel_size=3, inputs=(None, 1, 28, 28))

    A canonical bottleneck block (1x1, 3x3, 1x1 conv with relu in-between)::

        x = conv_block('nac nac nac', [64, 64, 256], [1, 3, 1], inputs=images_tensor)

    A complex Nd block:

    - 5x5 conv with 32 filters
    - relu
    - 3x3 conv with 32 filters
    - relu
    - 3x3 conv with 64 filters and a spatial stride 2
    - relu
    - batch norm
    - dropout with rate 0.15

    A simple block with disabled batch-norms to test self-normalization::

        x = conv_block('cna cna cna', filters=[64, 128, 256], kernel_size=3, activation='selu', batch_norm=False,
                       inputs=previous_layer)
    ::

        x = conv_block('ca ca ca nd', [32, 32, 64], [5, 3, 3], strides=[1, 1, 2], dropout_rate=.15, inputs=prev_layer)

    """
    def __init__(self, inputs=None, layout='', filters=None, kernel_size=3, strides=1, padding='same', dilation_rate=1,
                 depth_multiplier=1, activation='relu', pool_size=2, pool_strides=2, dropout_rate=0, units=None,
                 **kwargs):
        super().__init__()

        layout = layout or ''
        self.layout = layout.replace(' ', '')

        if len(self.layout) == 0:
            logger.warning('ConvBlock: layout is empty, so there is nothing to do')
            return

        layout_dict = {}
        for layer in self.layout:
            if C_GROUPS[layer] not in layout_dict:
                layout_dict[C_GROUPS[layer]] = [-1, 0]
            layout_dict[C_GROUPS[layer]][1] += 1

        new_layer = inputs
        modules = []
        for _, layer in enumerate(self.layout):

            layout_dict[C_GROUPS[layer]][0] += 1
            layer_name = C_LAYERS[layer]
            layer_fn = FUNC_LAYERS[layer_name]

            layer_args = kwargs.get(layer_name, {})
            skip_layer = layer_args is False or isinstance(layer_args, dict) and layer_args.get('disable', False)

            if skip_layer:
                pass

            elif layer == 'a':
                args = dict(activation=activation)

            elif layer == 'f':
                if units is None:
                    raise ValueError('units cannot be None if layout includes dense layers')
                args = dict(units=units)

            elif layer in ['c', 't']:
                args = dict(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                            dilation_rate=dilation_rate)
                if 'groups' in kwargs:
                    args['groups'] = kwargs['groups']

            elif layer in ['C', 'T']:
                args = dict(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                            dilation_rate=dilation_rate, depth_multiplier=depth_multiplier)

            elif layer == 'n':
                args = dict()

            elif C_GROUPS[layer] == 'p':
                pool_op = 'mean' if layer == 'v' else kwargs.pop('pool_op', 'max')
                args = dict(op=pool_op, kernel_size=pool_size, stride=pool_strides, padding=padding)

            elif C_GROUPS[layer] == 'P':
                pool_op = 'mean' if layer == 'V' else kwargs.pop('pool_op', 'max')
                args = dict(op=pool_op)

            elif layer in ['d', 'D']:
                if dropout_rate:
                    args = dict(p=dropout_rate)
                else:
                    logger.warning('ConvBlock: dropout_rate is zero or undefined, so dropout layer is skipped')
                    skip_layer = True

            elif layer == 'b':
                args = dict(scale_factor=kwargs.get('factor'), mode=kwargs.get('mode', 'bilinear'),
                            size=kwargs.get('shape'))

            elif layer == 'N':
                args = dict(scale_factor=kwargs.get('factor'), mode=kwargs.get('mode', 'nearest'),
                            size=kwargs.get('shape'))

            elif layer == 'X':
                args = dict(upscale_factor=kwargs.get('factor'))

            else:
                raise ValueError('Unknown layer symbol', layer)

            if not skip_layer:
                layer_args = layer_args.copy() if isinstance(layer_args, dict) else {}
                layer_args.pop('disable', None)
                args = {**args, **layer_args}
                args = unpack_args(args, *layout_dict[C_GROUPS[layer]])

                if 'batchflow.' in layer_fn.__module__:
                    args['inputs'] = new_layer

                new_layer = layer_fn(**args)
                modules.append(new_layer)

        self.block = nn.Sequential(*modules)
        self.output_shape = get_shape(self.block)


    def forward(self, x):
        """ Make forward pass """
        if self.layout:
            x = self.block(x)
        return x
