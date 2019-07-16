""" Contains common layers """
import inspect
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_num_dims, get_num_channels, get_shape


class Identity(nn.Module):
    """ Module which just returns its inputs

    Notes
    -----
    It slows training and inference so you should have a very good reason to use it.
    For instance, this could be a good option to replace some other module when debugging.
    """
    def __init__(self, inputs=None):
        super().__init__()
        self.output_shape = get_shape(inputs)

    def forward(self, x):
        return x


class Flatten(nn.Module):
    """ A module which reshapes inputs into 2-dimension (batch_items, features) """
    def __init__(self, inputs=None):
        super().__init__()
        shape = get_shape(inputs)
        if any(s is None for s in shape[1:]):
            self.output_shape = (shape[0], None)
        else:
            self.output_shape = (shape[0], np.prod(shape[1:]).tolist())

    def forward(self, x):
        return x.view(x.size(0), -1)


class Dense(nn.Module):
    """ A dense layer """
    def __init__(self, units=None, out_features=None, bias=True, inputs=None):
        super().__init__()

        units = units or out_features

        shape = get_shape(inputs)
        self.output_shape = (shape[0], units)
        self.linear = nn.Linear(np.prod(shape[1:]), units, bias)

    def forward(self, x):
        """ Make forward pass """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.linear(x)


ACTIVATIONS = {f.lower(): f for f in dir(nn)}

class Activation(nn.Module):
    """ A proxy activation module

    Parameters
    ----------
    activation : str, nn.Module, callable or None
        an activation function, can be

        - None - for identity function `f(x) = x`
        - str - a name from `torch.nn`
        - an instance of activation module (e.g. `torch.nn.ReLU()` or `torch.nn.ELU(alpha=2.0)`)
        - a class of activation module (e.g. `torch.nn.ReLU` or `torch.nn.ELU`)
        - a callable (e.g. `F.relu` or your custom function)

    args
        custom positional arguments passed to

        - a module class when creating a function
        - a callable during forward pass

    kwargs
        custom named arguments
    """
    def __init__(self, activation, *args, inputs=None, **kwargs):
        super().__init__()

        if 'inplace' not in kwargs:
            kwargs['inplace'] = True

        self.args = tuple()
        self.kwargs = {}
        self.output_shape = get_shape(inputs)

        if activation is None:
            self.activation = None
        if isinstance(activation, str):
            a = activation.lower()
            if a in ACTIVATIONS:
                _activation = getattr(nn, ACTIVATIONS[a])
                # check does activation has `in_place` parameter
                has_inplace = 'inplace' in inspect.getfullargspec(_activation).args
                if not has_inplace:
                    kwargs.pop('inplace', None)
                self.activation = _activation(*args, **kwargs)
            else:
                raise ValueError('Unknown activation', activation)
        elif isinstance(activation, nn.Module):
            self.activation = activation
        elif issubclass(activation, nn.Module):
            self.activation = activation(*args, **kwargs)
        elif callable(activation):
            self.activation = activation
            self.args = args
            self.kwargs = kwargs
        else:
            raise ValueError("Activation can be str, nn.Module or a callable, but given", activation)

    def forward(self, x):
        """ Make forward pass """
        if self.activation:
            return self.activation(x, *self.args, **self.kwargs)
        return x

def _get_padding(kernel_size=None, width=None, dilation=1, stride=1):
    kernel_size = dilation * (kernel_size - 1) + 1
    if stride >= width:
        p = max(0, kernel_size - width)
    else:
        if width % stride == 0:
            p = kernel_size - stride
        else:
            p = kernel_size - width % stride
    p = (p // 2, p - p // 2)
    return p

def _calc_padding(inputs, padding=0, kernel_size=None, dilation=1, transposed=False, stride=1, **kwargs):
    _ = kwargs

    dims = get_num_dims(inputs)
    shape = get_shape(inputs)

    if isinstance(padding, str):
        if padding == 'valid':
            padding = 0
        elif padding == 'same':
            if transposed:
                padding = 0
            else:
                if isinstance(kernel_size, int):
                    kernel_size = (kernel_size,) * dims
                if isinstance(dilation, int):
                    dilation = (dilation,) * dims
                if isinstance(stride, int):
                    stride = (stride,) * dims
                padding = tuple(_get_padding(kernel_size[i], shape[i+2], dilation[i], stride[i]) for i in range(dims))
        else:
            raise ValueError("padding can be 'same' or 'valid'")
    elif isinstance(padding, int):
        pass
    elif isinstance(padding, tuple):
        pass
    else:
        raise ValueError("padding can be 'same' or 'valid' or int or tuple of int")

    return padding

def _calc_output_shape(inputs, kernel_size=None, stride=None, dilation=1, padding=0, transposed=False, **kwargs):
    shape = get_shape(inputs)
    output_shape = list(shape)
    for i in range(2, len(shape)):
        if shape[i]:
            k = kernel_size[i - 2] if isinstance(kernel_size, tuple) else kernel_size
            p = padding[i - 2] if isinstance(padding, tuple) else padding
            p = sum(p) if isinstance(p, tuple) else p * 2
            s = stride[i - 2] if isinstance(stride, tuple) else stride
            d = dilation[i - 2] if isinstance(dilation, tuple) else dilation
            if transposed:
                output_shape[i] = (shape[i] - 1) * s + k - p
            else:
                output_shape[i] = (shape[i] + p - d * (k - 1) - 1) // s + 1
        else:
            output_shape[i] = None

    output_shape[1] = kwargs.get('out_channels') or output_shape[1]
    return tuple(output_shape)


class _Conv(nn.Module):
    """ An universal module for plain and transposed convolutions """
    def __init__(self, filters, kernel_size, stride=None, strides=None, padding='same',
                 dilation=None, dilation_rate=None, groups=None, bias=True, inputs=None,
                 _fn=None, transposed=False):
        super().__init__()

        shape = get_shape(inputs)

        args = {}

        args['in_channels'] = get_num_channels(shape)

        args['out_channels'] = filters

        args['groups'] = groups or 1

        args['kernel_size'] = kernel_size

        args['dilation'] = dilation or dilation_rate or 1

        args['stride'] = stride or strides or 1

        args['bias'] = bias

        _padding = _calc_padding(shape, padding=padding, transposed=transposed, **args)
        if isinstance(_padding, tuple) and isinstance(_padding[0], tuple):
            args['padding'] = 0
            self.padding = sum(_padding, ())
        else:
            args['padding'] = _padding
            self.padding = 0

        self.conv = _fn[get_num_dims(shape)](**args)

        args.pop('padding')
        self.output_shape = _calc_output_shape(shape, padding=_padding, transposed=transposed, **args)

    def forward(self, x):
        if self.padding:
            x = F.pad(x, self.padding[::-1])
        return self.conv(x)


CONV = {
    1: nn.Conv1d,
    2: nn.Conv2d,
    3: nn.Conv3d,
}

class Conv(_Conv):
    """ Multi-dimensional convolutional layer """
    def __init__(self, filters, kernel_size, stride=None, strides=None, padding='same',
                 dilation=None, dilation_rate=None, groups=None, bias=True, inputs=None):
        super().__init__(filters, kernel_size, stride, strides, padding,
                         dilation, dilation_rate, groups, bias, inputs, CONV)

class _SeparableConv(nn.Module):
    """ A universal multi-dimensional separable convolutional layer """
    def __init__(self, filters, kernel_size, stride=None, strides=None, padding='same',
                 dilation=None, dilation_rate=None, bias=True, depth_multiplier=1, inputs=None, _fn=None):
        super().__init__()

        in_channels = get_num_channels(inputs)
        out_channels = in_channels * depth_multiplier
        self.conv = _fn(out_channels, kernel_size, stride, strides, padding,
                        dilation, dilation_rate, in_channels, bias, inputs)

        if filters != out_channels:
            self.conv = nn.Sequential(
                self.conv,
                Conv(filters, 1, 1, 1, padding, 1, 1, 1, bias, inputs=self.conv.output_shape)
            )

    def forward(self, x):
        return self.conv(x)

class SeparableConv(_SeparableConv):
    """ Multi-dimensional separable convolutional layer """
    def __init__(self, filters, kernel_size, stride=None, strides=None, padding='same',
                 dilation=None, dilation_rate=None, bias=True, depth_multiplier=1, inputs=None):
        super().__init__(filters, kernel_size, stride, strides, padding,
                         dilation, dilation_rate, bias, depth_multiplier, inputs, Conv)

CONV_TR = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d,
}

class ConvTranspose(_Conv):
    """ Multi-dimensional transposed convolutional layer """
    def __init__(self, filters, kernel_size, stride=None, strides=None, padding='same',
                 dilation=None, dilation_rate=None, groups=None, bias=True, inputs=None):
        super().__init__(filters, kernel_size, stride, strides, padding,
                         dilation, dilation_rate, groups, bias, inputs, CONV_TR, True)


class SeparableConvTranspose(_SeparableConv):
    """ Multi-dimensional transposed separable convolutional layer """
    def __init__(self, filters, kernel_size, stride=None, strides=None, padding='same',
                 dilation=None, dilation_rate=None, bias=True, depth_multiplier=1, inputs=None):
        super().__init__(filters, kernel_size, stride, strides, padding,
                         dilation, dilation_rate, bias, depth_multiplier, inputs, ConvTranspose, True)

BATCH_NORM = {
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d,
}

class BatchNorm(nn.Module):
    """ Multi-dimensional batch normalization layer """
    def __init__(self, inputs=None, **kwargs):
        super().__init__()
        num_features = get_num_channels(inputs)
        self.norm = BATCH_NORM[get_num_dims(inputs)](num_features=num_features, **kwargs)
        self.output_shape = get_shape(inputs)

    def forward(self, x):
        return self.norm(x)


DROPOUT = {
    1: nn.Dropout,
    2: nn.Dropout2d,
    3: nn.Dropout3d,
}

class Dropout(nn.Module):
    """ Multi-dimensional dropout layer """
    def __init__(self, inputs=None, **kwargs):
        super().__init__()
        self.dropout = DROPOUT[get_num_dims(inputs)](**kwargs)
        self.output_shape = get_shape(inputs)

    def forward(self, x):
        return self.dropout(x)


class _Pool(nn.Module):
    """ A universal pooling layer """
    def __init__(self, inputs=None, padding='same', _fn=None, **kwargs):
        super().__init__()

        self.padding = None
        if isinstance(_fn, dict):
            if padding is not None:
                _padding = _calc_padding(inputs, padding=padding, **kwargs)
                self.output_shape = _calc_output_shape(inputs, padding=_padding, **kwargs)
                if isinstance(_padding, tuple) and isinstance(_padding[0], tuple):
                    self.padding = sum(_padding, ())
                else:
                    kwargs['padding'] = _padding

            self.pool = _fn[get_num_dims(inputs)](**kwargs)
        else:
            self.pool = _fn(inputs=inputs, **kwargs)

    def forward(self, x):
        if self.padding:
            x = F.pad(x, self.padding[::-1])
        return self.pool(x)


MAXPOOL = {
    1: nn.MaxPool1d,
    2: nn.MaxPool2d,
    3: nn.MaxPool3d,
}

class MaxPool(_Pool):
    """ Multi-dimensional max pooling layer """
    def __init__(self, padding='same', **kwargs):
        super().__init__(_fn=MAXPOOL, **kwargs)

AVGPOOL = {
    1: nn.AvgPool1d,
    2: nn.AvgPool2d,
    3: nn.AvgPool3d,
}

class AvgPool(_Pool):
    """ Multi-dimensional average pooling layer """
    def __init__(self, padding='same', **kwargs):
        super().__init__(_fn=AVGPOOL, **kwargs)


class Pool(_Pool):
    """ Multi-dimensional pooling layer """
    def __init__(self, inputs=None, op='max', **kwargs):
        if op == 'max':
            _fn = MaxPool
        elif op in ['avg', 'mean']:
            _fn = AvgPool
        super().__init__(_fn=_fn, inputs=inputs, **kwargs)
        self.output_shape = self.pool.output_shape


ADAPTIVE_MAXPOOL = {
    1: nn.AdaptiveMaxPool1d,
    2: nn.AdaptiveMaxPool2d,
    3: nn.AdaptiveMaxPool3d,
}

class AdaptiveMaxPool(_Pool):
    """ Multi-dimensional adaptive max pooling layer """
    def __init__(self, inputs=None, output_size=None, **kwargs):
        shape = get_shape(inputs)
        super().__init__(_fn=ADAPTIVE_MAXPOOL, inputs=inputs, output_size=output_size, padding=None, **kwargs)
        self.output_shape = tuple(shape[:2]) + tuple(output_size)


ADAPTIVE_AVGPOOL = {
    1: nn.AdaptiveAvgPool1d,
    2: nn.AdaptiveAvgPool2d,
    3: nn.AdaptiveAvgPool3d,
}

class AdaptiveAvgPool(_Pool):
    """ Multi-dimensional adaptive average pooling layer """
    def __init__(self, inputs=None, output_size=None, **kwargs):
        shape = get_shape(inputs)
        super().__init__(_fn=ADAPTIVE_AVGPOOL, inputs=inputs, output_size=output_size, padding=None, **kwargs)
        self.output_shape = tuple(shape[:2]) + tuple(output_size)


class AdaptivePool(_Pool):
    """ Multi-dimensional adaptive pooling layer """
    def __init__(self, op='max', inputs=None, **kwargs):
        if op == 'max':
            _fn = AdaptiveMaxPool
        elif op in ['avg', 'mean']:
            _fn = AdaptiveAvgPool
        super().__init__(_fn=_fn, inputs=inputs, padding=None, **kwargs)
        self.output_shape = self.pool.output_shape


class GlobalPool(nn.Module):
    """ Multi-dimensional global pooling layer """
    def __init__(self, inputs=None, op='max', **kwargs):
        super().__init__()
        shape = get_shape(inputs)
        pool_shape = [1] * len(shape[2:])
        self.output_shape = tuple(shape[:2])
        self.pool = AdaptivePool(op=op, output_size=pool_shape, inputs=inputs, **kwargs)

    def forward(self, x):
        x = self.pool(x)
        return x.view(x.size(0), -1)


_INTERPOLATE_MODES = {
    'n': 'nearest',
    'l': 'linear',
    'b': 'bilinear',
    't': 'trilinear',
}

class Interpolate(nn.Module):
    """ Upsample inputs with a given factor

    Notes
    -----
    This is just a wrapper around ``F.interpolate``.

    For brevity ``mode`` can be specified with the first letter only: 'n', 'l', 'b', 't'.

    All the parameters should the specified as keyword arguments (i.e. with names and values).
    """
    def __init__(self, *args, inputs=None, **kwargs):
        super().__init__()
        _ = args
        self.kwargs = kwargs

        mode = self.kwargs.get('mode')
        if mode in _INTERPOLATE_MODES:
            self.kwargs['mode'] = _INTERPOLATE_MODES[mode]

        shape = get_shape(inputs)
        self.output_shape = [*shape]
        if kwargs.get('size'):
            self.output_shape[2:] = kwargs['size']
        else:
            for i, s in enumerate(self.output_shape[2:]):
                self.output_shape[i+2] = None if s is None else s * kwargs['scale_factor']
        self.output_shape = tuple(self.output_shape)

    def forward(self, x):
        return F.interpolate(x, **self.kwargs)


class PixelShuffle(nn.PixelShuffle):
    """ Resize input tensor with depth to space operation """
    def __init__(self, upscale_factor=None, inputs=None):
        super().__init__(upscale_factor)
        shape = get_shape(inputs)
        self.output_shape = [*shape]
        self.output_shape[1] = self.output_shape[1] / upscale_factor ** 2
        for i, s in enumerate(self.output_shape[2:]):
            self.output_shape[i+2] = None if s is None else s * upscale_factor
        self.output_shape = tuple(self.output_shape)


class SubPixelConv(PixelShuffle):
    """ An alias for PixelShuffle """
    pass
