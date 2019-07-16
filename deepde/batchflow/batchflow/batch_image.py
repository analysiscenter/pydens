""" Contains Batch classes for images """
import os
from numbers import Number
from functools import wraps

import numpy as np
from skimage.transform import resize
import scipy.ndimage

import PIL
import PIL.ImageOps
import PIL.ImageChops
import PIL.ImageFilter
import PIL.ImageEnhance

from .batch import Batch
from .decorators import action, inbatch_parallel
from .dsindex import FilesIndex


def get_scipy_transforms():
    """ Returns ``dict`` {'function_name' : function} of functions from scipy.ndimage.

    Function is included if it has 'input : ndarray' or 'input : array_like' in its docstring.
    """

    scipy_transformations = {}
    hooks = ['input : ndarray', 'input : array_like']
    for function_name in scipy.ndimage.__dict__['__all__']:
        function = getattr(scipy.ndimage, function_name)
        doc = getattr(function, '__doc__')
        if doc is not None and (hooks[0] in doc or hooks[1] in doc):
            scipy_transformations[function_name] = function
    return scipy_transformations


def transform_actions(prefix='', suffix='', wrapper=None):
    """ Transforms classmethods that have names like <prefix><name><suffix> to pipeline's actions executed in parallel.

    First, it finds all *class methods* which names have the form <prefix><method_name><suffix>
    (ignores those that start and end with '__').

    Then, all found classmethods are decorated through ``wrapper`` and resulting
    methods are added to the class with the names of the form <method_name>.

    Parameters
    ----------
    prefix : str
    suffix : str
    wrapper : str
        name of the wrapper inside ``Batch`` class

    Examples
    --------
    >>> from dataset import ImagesBatch
    >>> @transform_actions(prefix='_', suffix='_')
    ... class MyImagesBatch(ImagesBatch):
    ...     @classmethod
    ...     def _flip_(cls, image):
    ...             return image[:,::-1]

    Note that if you only want to redefine actions you still have to decorate your class.

    >>> from dataset.opensets import CIFAR10
    >>> dataset = CIFAR10(batch_class=MyImagesBatch, path='.')

    Now dataset.pipeline has flip action that operates as described above.
    If you want to apply an action with some probability, then specify ``p`` parameter:

    >>> from dataset import Pipeline
    >>> pipeline = (Pipeline()
    ...                 ...preprocessing...
    ...                 .flip(p=0.7)
    ...                 ...postprocessing...

    Now each image will be flipped with probability 0.7.
    """
    def _decorator(cls):
        for method_name, method in cls.__dict__.copy().items():
            if method_name.startswith(prefix) and method_name.endswith(suffix) and\
               not method_name.startswith('__') and not method_name.endswith('__'):
                def _wrapper():
                    #pylint: disable=cell-var-from-loop
                    wrapped_method = method
                    @wraps(wrapped_method)
                    def _func(self, *args, src='images', target='for', **kwargs):
                        return getattr(cls, wrapper)(self, wrapped_method, src=src,
                                                     use_self=True, target=target, *args, **kwargs)
                    return _func
                name_slice = slice(len(prefix), -len(suffix))
                wrapped_method_name = method_name[name_slice]
                setattr(cls, wrapped_method_name, action(_wrapper()))
        return cls
    return _decorator


def add_methods(transformations=None, prefix='_', suffix='_'):
    """ Bounds given functions to a decorated class

    All bounded methods' names will be extended with ``prefix`` and ``suffix``.
    For example, if ``transformations``={'method_name': method}, ``suffix``='_all' and ``prefix``='_'
    then a decorated class will have '_method_name_all' method.

    Parameters
    ----------
    transformations : dict
        dict of the form {'method_name' : function_to_bound} -- functions to bound to a class
    prefix : str
    suffix : str
    """
    def _decorator(cls):
        for func_name, func in transformations.items():
            def _method_decorator():
                #pylint: disable=cell-var-from-loop
                added_func = func
                @wraps(added_func)
                def _method(_, *args, **kwargs):
                    return added_func(*args, **kwargs)
                return _method
            method_name = ''.join((prefix, func_name, suffix))
            added_method = _method_decorator()
            setattr(cls, method_name, added_method)
        return cls
    return _decorator


class BaseImagesBatch(Batch):
    """ Batch class for 2D images """
    components = "images", "labels", "masks"
    formats_lower = ['jpg', 'png', 'jpeg']
    formats = set(formats_lower + [x.upper() for x in formats_lower])

    def _make_path(self, ix, src=None):
        """ Compose path.

        Parameters
        ----------
        ix : str
            element's index (filename)
        src : str
            Path to folder with images. Used if `self.index` is not `FilesIndex`.

        Returns
        -------
        path : str
            Full path to an element.
        """

        if isinstance(src, FilesIndex):
            path = src.get_fullpath(ix)
        elif isinstance(self.index, FilesIndex):
            path = self.index.get_fullpath(ix)
        else:
            path = os.path.join(src, str(ix))
        return path

    def _load_image(self, ix, src=None, fmt=None, dst="images"):
        """ Loads image.

        .. note:: Please note that ``dst`` must be ``str`` only, sequence is not allowed here.

        Parameters
        ----------
        src : str, dataset.FilesIndex, None
            path to the folder with an image. If src is None then it is determined from the index.
        dst : str
            Component to write images to.
        fmt : str
            Format of the an image

        Raises
        ------
        NotImplementedError
            If this method is not defined in a child class
        """
        _ = self, ix, src, dst, fmt
        raise NotImplementedError("Must be implemented in a child class")

    @action
    def load(self, *args, src=None, fmt=None, dst=None, **kwargs):
        """ Load data.

        .. note:: if `fmt='images'` than ``components`` must be a single component (str).
        .. note:: All parameters must be named only.

        Parameters
        ----------
        src : str, None
            Path to the folder with data. If src is None then path is determined from the index.
        fmt : {'image', 'blosc', 'csv', 'hdf5', 'feather'}
            Format of the file to download.
        dst : str, sequence
            components to download.
        """
        if fmt == 'image':
            return self._load_image(src, fmt=fmt, dst=dst)
        return super().load(src=src, fmt=fmt, dst=dst, *args, **kwargs)


    def _dump_image(self, ix, src='images', dst=None, fmt=None):
        """ Saves image to dst.

        .. note:: Please note that ``src`` must be ``str`` only, sequence is not allowed here.

        Parameters
        ----------
        src : str
            Component to get images from.
        dst : str
            Folder where to dump. If dst is None then it is determined from index.

        Raises
        ------
        NotImplementedError
            If this method is not defined in a child class
        """
        _ = self, ix, src, dst, fmt
        raise NotImplementedError("Must be implemented in a child class")

    @action
    def dump(self, *args, dst=None, fmt=None, components="images", **kwargs):
        """ Dump data.

        .. note:: If `fmt='images'` than ``dst`` must be a single component (str).

        .. note:: All parameters must be named only.

        Parameters
        ----------
        dst : str, None
            Path to the folder where to dump. If dst is None then path is determined from the index.
        fmt : {'image', 'blosc', 'csv', 'hdf5', 'feather'}
            Format of the file to save.
        components : str, sequence
            Components to save.
        ext: str
            Format to save images to.

        Returns
        -------
        self
        """
        if fmt == 'image':
            return self._dump_image(components, dst, fmt=kwargs.pop('ext'))
        return super().dump(dst=dst, fmt=fmt, components=components, *args, **kwargs)


@transform_actions(prefix='_', suffix='_all', wrapper='apply_transform_all')
@transform_actions(prefix='_', suffix='_', wrapper='apply_transform')
@add_methods(transformations={**get_scipy_transforms(),
                              'pad': np.pad,
                              'resize': resize}, prefix='_sp_', suffix='_')
class ImagesBatch(BaseImagesBatch):
    """ Batch class for 2D images.

    Images are stored as numpy arrays of PIL.Image.

    PIL.Image has the following system of coordinates:
                       X
      0 -------------- >
      |
      |
      |  images's pixels
      |
      |
    Y v

    Pixel's position is defined as (x, y)
    """
    @classmethod
    def _get_image_shape(cls, image):
        if isinstance(image, PIL.Image.Image):
            return image.size
        return image.shape[:2]

    @property
    def image_shape(self):
        """: tuple - shape of the image"""
        _, shapes_count = np.unique([image.size for image in self.images], return_counts=True, axis=0)
        if len(shapes_count) == 1:
            if isinstance(self.images[0], PIL.Image.Image):
                return (*self.images[0].size, len(self.images[0].getbands()))
            return self.images[0].shape
        raise RuntimeError('Images have different shapes')

    @inbatch_parallel(init='indices', post='_assemble')
    def _load_image(self, ix, src=None, fmt=None, dst="images"):
        """ Loads image

        .. note:: Please note that ``dst`` must be ``str`` only, sequence is not allowed here.

        Parameters
        ----------
        src : str, dataset.FilesIndex, None
            Path to the folder with an image. If src is None then it is determined from the index.
        dst : str
            Component to write images to.
        fmt : str
            Format of an image.
        """
        return PIL.Image.open(self._make_path(ix, src))

    @inbatch_parallel(init='indices')
    def _dump_image(self, ix, src='images', dst=None, fmt=None):
        """ Saves image to dst.

        .. note:: Please note that ``src`` must be ``str`` only, sequence is not allowed here.

        Parameters
        ----------
        src : str
            Component to get images from.
        dst : str
            Folder where to dump.
        fmt : str
            Format of saved image.
        """
        if dst is None:
            raise RuntimeError('You must specify `dst`')
        image = self.get(ix, src)
        ix = str(ix) + '.' + fmt if fmt is not None else str(ix)
        image.save(os.path.join(dst, ix))

    def _assemble_component(self, result, *args, component='images', **kwargs):
        """ Assemble one component after parallel execution.

        Parameters
        ----------
        result : sequence, array_like
            Results after inbatch_parallel.
        component : str
            component to assemble
        preserve_shape : bool
            If True then all images are cropped from the top left corner to have similar shapes.
            Shape is chosen to be minimal among given images.
        """
        if isinstance(result[0], PIL.Image.Image):
            setattr(self, component, np.asarray(result, dtype=object))
        else:
            try:
                setattr(self, component, np.stack(result))
            except ValueError:
                array_result = np.empty(len(result), dtype=object)
                array_result[:] = result
                setattr(self, component, array_result)

    def _to_array_(self, image, dtype=None, channels='last'):
        """converts images in Batch to np.ndarray format

        Parameters
        ----------
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        """
        image = np.array(image)
        if len(image.shape) == 2:
            if channels == 'last':
                image = image[..., None]
            else:
                image = image[None, ...]

        if dtype is not None:
            image = image.astype(dtype)

        return image

    def _to_pil_(self, image, mode=None):
        """converts images in Batch to PIL format

        Parameters
        ----------
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        """
        if isinstance(image, PIL.Image.Image):
            return image

        if mode is None:
            if len(image.shape) == 2:
                mode = 'L'
            elif len(image.shape) == 3:
                if image.shape[-1] == 3:
                    mode = 'RGB'
                elif image.shape[-1] == 1:
                    mode = 'L'
                    image = image[:, :, 0]
                elif image.shape[-1] == 2:
                    mode = 'LA'
                elif image.shape[-1] == 4:
                    mode = 'RGBA'
            else:
                raise ValueError('Unknown image type as image has', image.shape[-1], 'channels')
        elif mode == 'L' and len(image.shape) == 3:
            image = image[..., 0]
        return PIL.Image.fromarray(image, mode)

    def _calc_origin(self, image_shape, origin, background_shape):
        """ Calculate coordinate of the input image with respect to the background.

        Parameters
        ----------
        image_shape : sequence
            shape of the input image.
        origin : array_like, sequence, {'center', 'top_left', 'random'}
            Position of the input image with respect to the background.

            - 'center' - place the center of the input image on the center of the background and crop
                         the input image accordingly.
            - 'top_left' - place the upper-left corner of the input image on the upper-left of the background
                           and crop the input image accordingly.
            - 'random' - place the upper-left corner of the input image on the randomly sampled position
                         in the background. Position is sampled uniformly such that there is no need for cropping.
            - other - place the upper-left corner of the input image on the given position in the background.
        background_shape : sequence
            shape of the background image.

        Returns
        -------
        sequence : calculated origin in the form (column, row)
        """
        if isinstance(origin, str):
            if origin == 'top_left':
                origin = 0, 0
            elif origin == 'center':
                origin = np.maximum(0, np.asarray(background_shape) - image_shape) // 2
            elif origin == 'random':
                origin = (np.random.randint(background_shape[0]-image_shape[0]+1),
                          np.random.randint(background_shape[1]-image_shape[1]+1))
        return np.asarray(origin, dtype=np.int)

    def _scale_(self, image, factor, preserve_shape=False, origin='center', resample=0):
        """ Scale the content of each image in the batch.

        Resulting shape is obtained as original_shape * factor.

        Parameters
        -----------
        factor : float, sequence
            resulting shape is obtained as original_shape * factor

            - float - scale all axes with the given factor
            - sequence (factor_1, factort_2, ...) - scale each axis with the given factor separately

        preserve_shape : bool
            whether to preserve the shape of the image after scaling

        origin : {'center', 'top_left', 'random'}, sequence
            Relevant only if `preserve_shape` is True.
            Position of the scaled image with respect to the original one's shape.

            - 'center' - place the center of the rescaled image on the center of the original one and crop
                         the rescaled image accordingly
            - 'top_left' - place the upper-left corner of the rescaled image on the upper-left of the original one
                           and crop the rescaled image accordingly
            - 'random' - place the upper-left corner of the rescaled image on the randomly sampled position
                         in the original one. Position is sampled uniformly such that there is no need for cropping.
            - sequence - place the upper-left corner of the rescaled image on the given position in the original one.
        resample: int
            Parameter passed to PIL.Image.resize. Interpolation order
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        Returns
        -------
        self
        """
        original_shape = self._get_image_shape(image)
        rescaled_shape = list(np.int32(np.ceil(np.asarray(original_shape)*factor)))
        rescaled_image = image.resize(rescaled_shape, resample=resample)
        if preserve_shape:
            rescaled_image = self._preserve_shape(original_shape, rescaled_image, origin)
        return rescaled_image

    def _crop_(self, image, origin, shape, crop_boundaries=False):
        """ Crop an image.

        Extract image data from the window of the size given by `shape` and placed at `origin`.

        Parameters
        ----------
        origin : sequence, str
            Upper-left corner of the cropping box. Can be one of:

            - sequence - corner's coordinates in the form of (row, column)
            - 'top_left' - crop an image such that upper-left corners of
                           an image and the cropping box coincide
            - 'center' - crop an image such that centers of
                         the image and the cropping box coincide
            - 'random' - place the upper-left corner of the cropping box at a random position
        shape : sequence

            - sequence - crop size in the form of (rows, columns)
        crop_boundaries : bool
            If `True` then crop is got only from image's area. Shape of the crop might diverge with the passed one
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        origin = self._calc_origin(shape, origin, image.size)
        right_bottom = origin + shape

        if crop_boundaries:
            out_of_boundaries = origin < 0
            origin[out_of_boundaries] = 0

            image_shape = np.asarray(image.size)
            out_of_boundaries = right_bottom > image_shape
            right_bottom[out_of_boundaries] = image_shape[out_of_boundaries]

        return image.crop((*origin, *right_bottom))

    def _put_on_background_(self, image, background, origin, mask=None):
        """ Put an image on a background at given origin

        Parameters
        ----------
        background : PIL.Image, np.ndarray of np.uint8
        origin : sequence, str
            Upper-left corner of the cropping box. Can be one of:

            - sequence - corner's coordinates in the form of (row, column).
            - 'top_left' - crop an image such that upper-left corners of an image and the cropping box coincide.
            - 'center' - crop an image such that centers of an image and the cropping box coincide.
            - 'random' - place the upper-left corner of the cropping box at a random position.

        mask : None, PIL.Image, np.ndarray of np.uint8
            mask passed to PIL.Image.paste
        """
        if not isinstance(background, PIL.Image.Image):
            background = PIL.Image.fromarray(background)
        else:
            background = background.copy()

        if not isinstance(mask, PIL.Image.Image):
            mask = PIL.Image.fromarray(mask) if mask is not None else None

        origin = list(self._calc_origin(self._get_image_shape(image), origin,
                                        self._get_image_shape(background)))

        background.paste(image, origin, mask)

        return background

    def _preserve_shape(self, original_shape, transformed_image, origin='center'):
        """ Change the transformed image's shape by cropping and adding empty pixels to fit the shape of original image.

        Parameters
        ----------
        original_shape : sequence
        transformed_image : np.ndarray
        origin : {'center', 'top_left', 'random'}, sequence
            Position of the transformed image with respect to the original one's shape.

            - 'center' - place the center of the transformed image on the center of the original one and crop
                         the transformed image accordingly.
            - 'top_left' - place the upper-left corner of the transformed image on the upper-left of the original one
                           and crop the transformed image accordingly.
            - 'random' - place the upper-left corner of the transformed image on the randomly sampled position
                         in the original one. Position is sampled uniformly such that there is no need for cropping.
            - sequence - place the upper-left corner of the transformed image on the given position in the original one.

        Returns
        -------
        np.ndarray : image after described actions
        """
        n_channels = len(transformed_image.getbands())
        if n_channels == 1:
            background = np.zeros(original_shape, dtype=np.uint8)
        else:
            background = np.zeros((*original_shape, n_channels), dtype=np.uint8)

        crop_origin = 'top_left' if origin != 'center' else 'center'

        return self._put_on_background_(self._crop_(transformed_image, crop_origin, original_shape, True),
                                        background,
                                        origin)

    def _filter_(self, image, mode, *args, **kwargs):
        """ Filters an image. Calls image.filter(getattr(PIL.ImageFilter, mode)(*args, **kwargs))

        For more details see http://pillow.readthedocs.io/en/stable/reference/ImageFilter.html

        Parameters
        ----------
        mode : str
            Name of the filter.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        return image.filter(getattr(PIL.ImageFilter, mode)(*args, **kwargs))

    def _transform_(self, image, *args, **kwargs):
        """ Calls image.transform(*args, **kwargs)

        For more information see http://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.transform

        Parameters
        ----------
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        size = kwargs.pop('size', self._get_image_shape(image))
        return image.transform(*args, size=size, **kwargs)

    def _resize_(self, image, *args, **kwargs):
        """ Calls image.resize(*args, **kwargs)

        For more details see https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize

        Parameters
        ----------
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        return image.resize(*args, **kwargs)

    def _shift_(self, image, offset, mode='const'):
        """ Shifts an image.

        Parameters
        ----------
        offset : (Number, Number)
        mode : {'const', 'wrap'}
            How to fill borders
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        if mode == 'const':
            image = image.transform(size=image.size,
                                    method=PIL.Image.AFFINE,
                                    data=(1, 0, -offset[0], 0, 1, -offset[1]))
        elif mode == 'wrap':
            image = PIL.ImageChops.offset(image, *offset)
        else:
            raise ValueError("mode must be one of ['const', 'wrap']")
        return image

    def _pad_(self, image, *args, **kwargs):
        """ Calls PIL.ImageOps.expand.

        For more details see http://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.expand

        Parameters
        ----------
        offset : sequence
            Size of the borders in pixels. The order is (left, top, right, bottom).
        mode : {'const', 'wrap'}
            Filling mode
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        return PIL.ImageOps.expand(image, *args, **kwargs)

    def _rotate_(self, image, *args, **kwargs):
        """ Rotates an image.

            kwargs are passed to PIL.Image.rotate

        Parameters
        ----------
        angle: Number
            In degrees counter clockwise.
        resample: int
            Interpolation order
        expand: bool
            Whether to expand the output to hold the whole image. Default is False.
        center: (Number, Number)
            Center of rotation. Default is the center of the image.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        return image.rotate(*args, **kwargs)

    def _flip_(self, image, mode='lr'):
        """ Flips image.

        Parameters
        ----------
        mode : {'lr', 'ud'}

            - 'lr' - apply the left/right flip
            - 'ud' - apply the upside/down flip
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        if mode == 'lr':
            return PIL.ImageOps.mirror(image)
        return PIL.ImageOps.flip(image)

    def _invert_(self, image, channels='all'):
        """ Invert givn channels.

        Parameters
        ----------
        channels : int, sequence
            Indices of the channels to invert.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        if channels == 'all':
            image = PIL.ImageChops.invert(image)
        else:
            bands = list(image.split())
            channels = (channels,) if isinstance(channels, Number) else channels
            for channel in channels:
                bands[channel] = PIL.ImageChops.invert(bands[channel])
            image = PIL.Image.merge('RGB', bands)
        return image

    def _salt_(self, image, p_noise=.015, color=255, size=(1, 1)):
        """ Set random pixel on image to givan value.

        Every pixel will be set to ``color`` value with probability ``p_noise``.

        Parameters
        ----------
        p_noise : float
            Probability of salting a pixel.
        color : float, int, sequence, callable
            Color's value.

            - int, float, sequence -- value of color
            - callable -- color is sampled for every chosen pixel (rules are the same as for int, float and sequence)
        size : int, sequence of int, callable
            Size of salt

            - int -- square salt with side ``size``
            - sequence -- recangular salt in the form (row, columns)
            - callable -- size is sampled for every chosen pixel (rules are the same as for int and sequence)
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        mask_size = np.asarray(self._get_image_shape(image))
        mask_salt = np.random.binomial(1, p_noise, size=mask_size).astype(bool)
        image = np.array(image)
        if isinstance(size, (tuple, int)) and size in [1, (1, 1)] and not callable(color):
            image[mask_salt] = color
        else:
            size_lambda = size if callable(size) else lambda: size
            color_lambda = color if callable(color) else lambda: color
            mask_salt = np.where(mask_salt)
            for i in range(len(mask_salt[0])):
                current_size = size_lambda()
                current_size = (current_size, current_size) if isinstance(current_size, Number) else current_size
                left_top = np.asarray((mask_salt[0][i], mask_salt[1][i]))
                right_bottom = np.minimum(left_top + current_size, self._get_image_shape(image))
                image[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]] = color_lambda()

        return PIL.Image.fromarray(image)

    def _clip_(self, image, low=0, high=255):
        """ Truncate image's pixels.

        Parameters
        ----------
        low : int, float, sequence
            Actual pixel's value is equal max(value, low). If sequence is given, then its length must coincide
            with the number of channels in an image and each channel is thresholded separately
        high : int, float, sequence
            Actual pixel's value is equal min(value, high). If sequence is given, then its length must coincide
            with the number of channels in an image and each channel is thresholded separately
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        if isinstance(low, Number):
            low = tuple([low]*3)
        if isinstance(high, Number):
            high = tuple([high]*3)

        high = PIL.Image.new('RGB', image.size, high)
        low = PIL.Image.new('RGB', image.size, low)
        return PIL.ImageChops.lighter(PIL.ImageChops.darker(image, high), low)

    def _enhance_(self, image, layout='hcbs', factor=(1, 1, 1, 1)):
        """ Apply enhancements from PIL.ImageEnhance to the image.

        Parameters
        ----------
        layout : str
            defines layout of operations, default is `hcbs`:
            h - color
            c - contrast
            b - brightness
            s - sharpness

        factor : float or tuple of float
            factor of enhancement for each operation listed in `layout`.
        """
        enhancements = {
            'h': 'Color',
            'c': 'Contrast',
            'b': 'Brightness',
            's': 'Sharpness'
        }

        if isinstance(factor, float):
            factor = (factor,) * len(layout)
        if len(layout) != len(factor):
            raise ValueError("'layout' and 'factor' should be of same length!")

        for alias, multiplier in zip(layout, factor):
            enhancement = enhancements.get(alias)
            if enhancement is None:
                raise ValueError('Unknown enhancement alias: ', alias)
            image = getattr(PIL.ImageEnhance, enhancement)(image).enhance(multiplier)

        return image

    def _multiply_(self, image, multiplier=1., clip=False, preserve_type=False):
        """ Multiply each pixel by the given multiplier.

        Parameters
        ----------
        multiplier : float, sequence
        clip : bool
            whether to force image's pixels to be in [0, 255] or [0, 1.]
        preserve_type : bool
            Whether to preserve ``dtype`` of transformed images.
            If ``False`` is given then the resulting type will be ``np.float``.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        multiplier = np.float32(multiplier)
        if isinstance(image, PIL.Image.Image):
            return PIL.Image.fromarray(np.clip(multiplier*np.asarray(image), 0, 255).astype(np.uint8))
        dtype = image.dtype if preserve_type else np.float
        if clip:
            image = np.clip(multiplier*image, 0, 255 if dtype == np.uint8 else 1.)
        else:
            image = multiplier * image
        return image.astype(dtype)

    def _add_(self, image, term=1., clip=False, preserve_type=False):
        """ Add term to each pixel.

        Parameters
        ----------
        term : float, sequence
        clip : bool
            whether to force image's pixels to be in [0, 255] or [0, 1.]
        preserve_type : bool
            Whether to preserve ``dtype`` of transformed images.
            If ``False`` is given then the resulting type will be ``np.float``.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        term = np.float32(term)
        if isinstance(image, PIL.Image.Image):
            return PIL.Image.fromarray(np.clip(term+np.asarray(image), 0, 255).astype(np.uint8))
        dtype = image.dtype if preserve_type else np.float
        if clip:
            image = np.clip(term+image, 0, 255 if dtype == np.uint8 else 1.)
        else:
            image = term + image
        return image.astype(dtype)

    def _pil_convert_(self, image, mode="L"):
        """ Convert image. Actually calls image.convert(mode)

        Parameters
        ----------
        mode : str
            Pass 'L' to convert to grayscale
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        return image.convert(mode)

    def _posterize_(self, image, bits=4):
        """ Posterizes image.

        More concretely, it quantizes pixels' values so that they have``2^bits`` colors

        Parameters
        ----------
        bits : int
            Number of bits used to store a color's component.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        return PIL.ImageOps.posterize(image, bits)

    def _cutout_(self, image, origin, shape, color):
        """ Fills given areas with color

        .. note:: It is assumed that ``origins``, ``shapes`` and ``colors`` have the same length.

        Parameters
        ----------
        origin : sequence, str
            Upper-left corner of a filled box. Can be one of:

            - sequence - corner's coordinates in the form of (row, column).
            - 'top_left' - crop an image such that upper-left corners of
                           an image and the filled box coincide.
            - 'center' - crop an image such that centers of
                         an image and the filled box coincide.
            - 'random' - place the upper-left corner of the filled box at a random position.
        shape : sequence, int
            Shape of a filled box. Can be one of:

            - sequence - crop size in the form of (rows, columns)
            - int - shape has squared form
        color : sequence, number
            Color of a filled box. Can be one of:

            - sequence - (r,g,b) form
            - number - grayscale
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        image = image.copy()
        shape = (shape, shape) if isinstance(shape, Number) else shape
        origin = self._calc_origin(shape, origin, self._get_image_shape(image))
        color = (color, color, color) if isinstance(color, Number) else color
        image.paste(PIL.Image.new('RGB', tuple(shape), tuple(color)), tuple(origin))
        return image

    def _assemble_patches(self, patches, *args, dst, **kwargs):
        """ Assembles patches after parallel execution.

        Parameters
        ----------
        patches : sequence
            Patches to gather. pathces.shape must be like (batch.size, patches_i, patch_height, patch_width, n_channels)
        dst : str
            Component to put patches in.
        """
        _ = args, kwargs
        new_items = np.concatenate(patches)
        setattr(self, dst, new_items)

    @action
    @inbatch_parallel(init='indices', post='_assemble_patches')
    def split_to_patches(self, ix, patch_shape, stride=1, drop_last=False, src='images', dst=None):
        """ Splits image to patches.

        Small images with the same shape (``patch_shape``) are cropped from the original one with stride ``stride``.

        Parameters
        ----------
        patch_shape : int, sequence
            Patch's shape in the from (rows, columns). If int is given then patches have square shape.
        stride : int, square
            Step of the moving window from which patches are cropped. If int is given then the window has square shape.
        drop_last : bool
            Whether to drop patches whose window covers area out of the image.
            If False is passed then these patches are cropped from the edge of an image. See more in tutorials.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        _ = dst
        image = self.get(ix, src)
        image_shape = self._get_image_shape(image)
        image = np.array(image)
        stride = (stride, stride) if isinstance(stride, Number) else stride
        patch_shape = (patch_shape, patch_shape) if isinstance(patch_shape, Number) else patch_shape
        patches = []

        def _iterate_columns(row_from, row_to):
            column = 0
            while column < image_shape[1]-patch_shape[1]+1:
                patches.append(PIL.Image.fromarray(image[row_from:row_to, column:column+patch_shape[1]]))
                column += stride[1]
            if not drop_last and column + patch_shape[1] != image_shape[1]:
                patches.append(PIL.Image.fromarray(image[row_from:row_to,
                                                         image_shape[1]-patch_shape[1]:image_shape[1]]))

        row = 0
        while row < image_shape[0]-patch_shape[0]+1:
            _iterate_columns(row, row+patch_shape[0])
            row += stride[0]
        if not drop_last and row + patch_shape[0] != image_shape[0]:
            _iterate_columns(image_shape[0]-patch_shape[0], image_shape[0])

        return np.array(patches, dtype=object)

    def _additive_noise_(self, image, noise, clip=False, preserve_type=False):
        """ Add additive noise to an image.

        Parameters
        ----------
        noise : callable
            Distribution. Must have ``size`` parameter.
        clip : bool
            whether to force image's pixels to be in [0, 255] or [0, 1.]
        preserve_type : bool
            Whether to preserve ``dtype`` of transformed images.
            If ``False`` is given then the resulting type will be ``np.float``.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        noise = noise(size=(*image.size, len(image.getbands())) if isinstance(image, PIL.Image.Image) else image.shape)
        return self._add_(image, noise, clip, preserve_type)

    def _multiplicative_noise_(self, image, noise, clip=False, preserve_type=False):
        """ Add multiplicativa noise to an image.

        Parameters
        ----------
        noise : callable
            Distribution. Must have ``size`` parameter.
        clip : bool
            whether to force image's pixels to be in [0, 255] or [0, 1.]
        preserve_type : bool
            Whether to preserve ``dtype`` of transformed images.
            If ``False`` is given then the resulting type will be ``np.float``.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        noise = noise(size=(*image.size, len(image.getbands())) if isinstance(image, PIL.Image.Image) else image.shape)
        return self._multiply_(image, noise, clip, preserve_type)

    def _elastic_transform_(self, image, alpha, sigma, **kwargs):
        """Elastic deformation of images as described in [Simard2003]_.
        [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.

        Code slightly differs with https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a

        Parameters
        ----------
        alpha : number
            maximum of vectors' norms.
        sigma : number
            Smooth factor.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        image = np.array(image)
        # full shape is needed
        shape = image.shape
        if len(shape) == 2:
            image = image[..., None]
            shape = image.shape

        kwargs.setdefault('mode', 'constant')
        kwargs.setdefault('cval', 0)

        column_shift = self._sp_gaussian_filter_(np.random.uniform(-1, 1, size=shape), sigma, **kwargs) * alpha
        row_shift = self._sp_gaussian_filter_(np.random.uniform(-1, 1, size=shape), sigma, **kwargs) * alpha

        row, column, channel = np.meshgrid(range(shape[0]), range(shape[1]), range(shape[2]))

        indices = (column + column_shift, row + row_shift, channel)

        distored_image = self._sp_map_coordinates_(image, indices, order=1, mode='reflect')

        if shape[-1] == 1:
            return PIL.Image.fromarray(np.uint8(distored_image.reshape(image.shape))[..., 0])
        return PIL.Image.fromarray(np.uint8(distored_image.reshape(image.shape)))
