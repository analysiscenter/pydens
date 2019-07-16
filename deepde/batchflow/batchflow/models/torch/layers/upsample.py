""" Contains upsampling layer """
import torch.nn as nn
from .conv_block import ConvBlock


class Upsample(nn.Module):
    """ Upsample inputs with a given factor

    Parameters
    ----------
    factor : int
        an upsamping scale
    shape : tuple of int
        a shape to upsample to (used by bilinear and NN resize)
    layout : str
        resizing technique, a sequence of:

        - b - bilinear resize
        - N - nearest neighbor resize
        - t - transposed convolution
        - T - separable transposed convolution
        - X - subpixel convolution

        all other :class:`~.torch.ConvBlock` layers are also allowed.

    inputs
        an input tensor

    Examples
    --------
    A simple bilinear upsampling::

        x = Upsample(layout='b', shape=(256, 256), inputs=inputs)

    Upsampling with non-linear normalized transposed convolution::

        x = Upsample(layout='nat', factor=2, kernel_size=3, inputs=inputs)

    Subpixel convolution::

        x = Upsample(layout='X', factor=2, inputs=inputs)
    """
    def __init__(self, factor=2, shape=None, layout='b', *args, inputs=None, **kwargs):
        super().__init__()

        _ = args

        if 't' in layout or 'T' in layout:
            if 'kernel_size' not in kwargs:
                kwargs['kernel_size'] = factor
            if 'strides' not in kwargs:
                kwargs['strides'] = factor

        self.upsample = ConvBlock(layout=layout, factor=factor, shape=shape, inputs=inputs, **kwargs)
        self.output_shape = self.upsample.output_shape

    def forward(self, x):
        return self.upsample(x)
