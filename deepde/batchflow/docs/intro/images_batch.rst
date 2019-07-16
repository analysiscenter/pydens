===============================
Batch class for handling images
===============================

ImagesBatch class handles 2D images and their labels. Images are stored as PIL.Image (usually) or np.ndarray.

Components
----------
The class has two components: ``images`` and ``labels``.

Conversion between formats
--------------------------
Almost all actions in the batch work with `PIL` images. If dataset contains `np.ndarray` images, just call :meth:`to_pil <batchflow.ImagesBatch.to_pil>` method to convert them inside batch.
To convert images to `np.ndarray` use :meth:`to_array <batchflow.ImagesBatch.to_array>` (this might be needed, for example, before passing images to a model).

Augmentation
------------

ImagesBatch provides typical augmentation actions, for example:

* :meth:`crop <batchflow.ImagesBatch.crop>` -- crop rectangular area from an image
    ..  image:: ../_static/ImagesBatch_examples/crop.png
* :meth:`flip <batchflow.ImagesBatch.flip>` -- flip an image (left to right or upside down)
    ..  image:: ../_static/ImagesBatch_examples/flip.png
* :meth:`scale <batchflow.ImagesBatch.scale>` -- scale an image (stretch or tie)
    ..  image:: ../_static/ImagesBatch_examples/scale.png
* :meth:`put_on_background <batchflow.ImagesBatch.put_on_background>` -- put an image on a given background
    ..  image:: ../_static/ImagesBatch_examples/put_on_background.png
* :meth:`resize <batchflow.ImagesBatch.resize>` -- resize an image a to the given shape
    ..  image:: ../_static/ImagesBatch_examples/resize.png
* :meth:`pad <batchflow.ImagesBatch.pad>` -- add constant values to the border of an image (enlarging the last's shape)
    ..  image:: ../_static/ImagesBatch_examples/pad.png
* :meth:`invert <batchflow.ImagesBatch.invert>` -- invert given channels in an image
    ..  image:: ../_static/ImagesBatch_examples/invert.png
* :meth:`salt <batchflow.ImagesBatch.salt>` -- set pixels in random positions to given colour
    ..  image:: ../_static/ImagesBatch_examples/salt.png
* :meth:`clip <batchflow.ImagesBatch.clip>` -- truncate pixels' values
    ..  image:: ../_static/ImagesBatch_examples/threshold.png
* :meth:`multiply <batchflow.ImagesBatch.multiply>` -- multiply an image by the given number
    ..  image:: ../_static/ImagesBatch_examples/multiply.png
* :meth:`multiplicative_noise <batchflow.ImagesBatch.multiplicative_noise>` -- add multiplicative noise to an image
    ..  image:: ../_static/ImagesBatch_examples/multiplicative_noise.png
* :meth:`add <batchflow.ImagesBatch.add>` -- add given term to an image
    ..  image:: ../_static/ImagesBatch_examples/add.png
* :meth:`additive_noise <batchflow.ImagesBatch.additive_noise>` -- add additive noise an image
    ..  image:: ../_static/ImagesBatch_examples/additive_noise.png
* :meth:`posterize <batchflow.ImagesBatch.posterize>` -- posterize an image
    ..  image:: ../_static/ImagesBatch_examples/posterize.png
* :meth:`cutout <batchflow.ImagesBatch.cutout>` -- add colored rectangular areas to an image
    ..  image:: ../_static/ImagesBatch_examples/cutout.png
* :meth:`elastic_transform <batchflow.ImagesBatch.elastic_transform>` -- add colored rectangular areas to an image
    ..  image:: ../_static/ImagesBatch_examples/elastic.png


Perhaps, any function from scipy.ndimage is accesible as sp_<method_name>. Just use it as a usual action (without specifying input parameter). Note that they only works with scipy.ndarray and usually much slower than respective PIL methods.
.. note:: All these methods can be executed for randomly sampled images from a batch. You just need to specify ``p`` parameter when calling an action (probability of applying an action to an image).

.. note:: Use ``R()`` or ``P(R())`` :doc:`named expressions <named_expr>` to sample an argument for actions. In the first case the argument will be sampled for all images in a batch. If ``P(R())`` is passed then the argument will be sampled for each image.

Examples:

All images in a batch are rotated by 10 degrees:

.. code-block:: python

    ...
    (Pipeline().
        ...
        .rotate(angle=10)
        ...

All images in a batch are rotated by the common angle sampled from the normal distribution

.. code-block:: python

    ...
    (Pipeline().
        ...
        .rotate(angle=R('normal', loc=0, scale=1))
        ...

Each image in a batch are rotated by its own sampled angle

.. code-block:: python

    ...
    (Pipeline().
        ...
        .rotate(angle=P(R('normal', loc=0, scale=1)))
        ...


Rotate each image with probability 0.7 by its own sampled angle

.. code-block:: python

    ...
    (Pipeline().
        ...
        .rotate(angle=P(R('normal', loc=0, scale=1)), p=0.7)
        ...

See more details in `the augmentation tutorial <https://github.com/analysiscenter/batchflow/blob/master/examples/tutorials/06_image_augmentation.ipynb>`_.

Loading from files
------------------

To load images, use action :meth:`load <batchflow.ImagesBatch.load>` with ``fmt='image'``.


Saving
------

To dump images, use action :meth:`dump <batchflow.ImagesBatch.dump>`


`transform_actions` decorator
-----------------------------

This decorator finds all defined methods whose names starts with user-defined `suffix` and `prefix` and
decorates them with ``wrapper`` which is an argument too.

For example, there are two wrapper functions defined in :class:`~batchflow.Batch`:
    1. :meth:`~batchflow.Batch.apply_transform_all`
    2. :meth:`~batchflow.Batch.apply_transform`

And, by default, all methods that start with '_' and end with '_' are wrapped with the first mentioned method and those ones that start with '_' and end with '_all' are wrapped by the second one.

Defining custom actions
-----------------------

There are 3 ways to define an action:

    1. By writting a classic ``action`` like in  :class:`~batchflow.Batch`
    2. By writing a method that takes ``image`` as the first argument and returns transformed one. Method's name must be surrounded by unary '_'.
    3. By writing a method that takes nd.array of ``images`` as the first argument and ``indices`` as the second. This method transforms ``images[indices]`` and returns ``images``. Method's name must start with '_' and end with '_all'.

.. note:: In the last two approaches, actual action's name doesn't include mentioned suffices and prefixes. For example, if you define method ``_method_name_`` then in a pipeline you should call ``method_name``. For more details, see below.

.. note:: Last two methods' names must not be surrounded by double '_' (like `__init__`) otherwise they will be ignored.

Let's take a closer look on the two last approaches:

``_method_name_``
~~~~~~~~~~~~~~~~~

It must have the following signature:

   ``_method_name_(image, ...)``

This method is actually wrapped with :meth:`~batchflow.Batch.apply_transform`. And (usually) executed in parallel for each image.


.. note:: If you define these actions in a child class then you must decorate it with ``@transform_actions(prefix='_', suffix='_', wrapper='apply_transform')``

Example:

.. code-block:: python

    @transform_actions(prefix='_', suffix='_', wrapper='apply_transform')
    class MyImagesBatch(ImagesBatch):
        ...
        def _flip_(image, mode):
            """ Flips an image.
            """

            if mode == 'lr':
                return PIL.ImageOps.mirror(image)
            return PIL.ImageOps.flip(image)
        ...

To use this action in a pipeline you must write:

.. code-block:: python

    ...
    (Pipeline().
        ...
        .flip(mode='lr')
        ...

.. note:: Note that prefix '_' and suffix '_' are removed from the action's name.

.. note:: All actions written in this way can be applied with given probability to every image. To achieve this, pass parameter ``p`` to an action, like ``flip(mode='lr', p=0.5)``

.. note:: These actions are performed for every image each in its own thread. To change it (for example, execute in asynchronous mode), pass parameter `target` (``.flip(mode='lr', target='a')``). For more detail, see :doc:`parallel <parallel>`.


``_method_name_all``
~~~~~~~~~~~~~~~~~~~~


It must have the following signature:

   ``_method_name_all(images, indices, ...)``

This method is actually wrapped with :meth:`~batchflow.Batch.apply_transform_all`. And executed once with the whole batch passed. ``indices`` parameter declares images that must be transformed (it is needed, for example, if you want to perform action only to the subset of the elements. See below for more details)


.. note:: If you define these actions in a child class then you must decorate it with ``@transform_actions(prefix='_', suffix='_all', wrapper='apply_transform_all')``

Example:

.. code-block:: python

    @transform_actions(prefix='_', suffix='_', wrapper='apply_transform_all')
    class MyImagesBatch(ImagesBatch):
        ...
        def _flip_all(self, images=None, indices=[0], mode='lr'):
            """ Flips images at given indices.
            """

            for ind in indices:
              if mode == 'lr':
                  images[ind] = PIL.ImageOps.mirror(images[ind])
              images[ind] = PIL.ImageOps.flip(images[ind])

            return images
        ...

To use this action in a pipeline you must write:

.. code-block:: python

    ...
    (Pipeline().
        ...
        .flip(mode='lr')
        ...


.. note:: Note that prefix '_' and suffix '_all' are removed from the action's name.

.. note:: All actions written in this way can be applied with given probability to every image. To achieve this, pass parameter ``p`` to an action, like ``flip(mode='lr', p=0.5)``

.. note:: These actions are performed once for all batch. Please note that you can't pass ``P(R())`` named expression as an argument.


Assembling after parallel execution
-----------------------------------

Note that if images have different shapes after an action then there are two ways to tackle it:

  1. Do nothing. Then images will be stored in `np.ndarray` with `dtype=object`.
  2. Pass `preserve_shape=True` to an action which changes the shape of an image. Then image
     is cropped from the left upper corner (unless action has `origin` parameter).

Cropping to patches
-------------------------

If you have a very big image then you can compose little patches from it.
See :meth:`split_to_patches <batchflow.ImagesBatch.split_to_patches>` and tutorial for more details.
