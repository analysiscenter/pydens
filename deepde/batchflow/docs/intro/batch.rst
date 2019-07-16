===========
Batch class
===========

Batch class holds the data and contains processing functions.
Normally, you never create batch instances, as they are created in the `Dataset` or `Pipeline` batch generators.

Index
=====

`Batch` class stores the :doc:`index <index>` of all data items which belong to the batch. You can access the index through `self.index` (it is an instance of :ref:`DatasetIndex` or its child). The sequence of indices is also available as `self.indices`.

Data
====

The base :class:`~batchflow.Batch` class has a private property :attr:`~batchflow.Batch._data` which you can use to store your data in. Just call :func:`~batchflow.Batch.put_into_data`. After that, you can access data through a public property :attr:`~batchflow.Batch.data`. This approach allows to conceal an internal data structure and provides for a more convenient and (perhaps) more stable public interface to access the data.::

    class MyBatch(Batch):
        def some_method(self):
            self.put_into_data(self.indices, some_data)

If your batch has components_, you might put only a few components::

    class MyBatch(Batch):
        def some_method(self):
            self.put_into_data(self.indices, some_data, components=['comp1', 'comp2'])

Even though this is just a convention and you are not obliged to use it, many predefined methods follow it, thus making your life a bit easier.

preloaded
^^^^^^^^^

To fill in the batch with preloaded data you might initialize it with `preloaded` argument::

   batch = MyBatch(index, preloaded=data)

So :attr:`~batchflow.Batch.data` will contain data right after batch creation and you don't need to call :func:`~batchflow.Batch.load` action.

You also might initialize the whole dataset::

   dataset = Dataset(index, batch_class=Mybatch, preloaded=data)

Thus :func:`~batchflow.Dataset.gen_batch` and :func:`~batchflow.Dataset.next_batch` will create batches that contain preloaded data.

To put it simply, `preloaded=data` is roughly equivalent to `batch.load(data, fmt=None)`.

.. _components:

components
^^^^^^^^^^

Not infrequently, the batch stores a more complex data structures, e.g. features and labels or images, masks, bounding boxes and labels. To work with these you might employ data components. Just define a property as follows::

   class MyBatch(Batch):
       components = 'images', 'masks', 'labels'

And this allows you to address components to read and write data::

   image_5 = batch.images[5]
   batch.images[i] = new_image
   label_k = batch[k].labels
   batch[4].masks = new_masks

Numerous methods take ``components`` parameters which allows to specify which components will be affected by the method.
For instance, you can load components from different sources, or save components to disk, or apply some transformations
(like resizing, zooming or rotating).


Dataset
=======

Each batch also refers to a dataset which it was created from - `batch.dataset`. However, note that while a batch travels through a pipeline it might be transformed beyond recognition,
but the dataset reference does not change.

Another way to access dataset attributes is to use :class:`~.batchflow.D`-expression.

.. _actions:

Action methods
==============

`Action`-methods form a public API of the batch class which is available in :doc:`pipelines <pipeline>`. If you operate directly with the batch class instances, you don't need `action`-methods. However, pipelines provide the most convenient interface to process the whole dataset and to separate data processing steps and model training / validation cycles.

In order to convert a batch class method to an action you add `@action` decorator::

   from batchflow import Batch, action

   class MyBatch(Batch):
       ...
       @action
       def some_action(self):
           # process your data
           return self

Take into account that an `action`-method should return an instance of some `Batch`-class: the very same one or some other class.
If an `action` changes the instance's data directly, it may simply return `self`.


Models and model-based actions
==============================

To get access to a model just call :func:`~batchflow.Batch.get_model_by_name` within actions or ordinary batch class methods.::

   class MyBatch(Batch):
       ...
       @action
       def train_my_model(model_name):
           my_model = self.get_model_by_name(model_name)
           my_model.train(...)

For more details see :doc:`Working with models <models>`.


Running methods in parallel
===========================

As a batch can be quite large it might make sense to parallel the computations. And it is pretty easy to do::

   from batchflow import Batch, inbatch_parallel, action

   class MyBatch(Batch):
       ...
       @action
       @inbatch_parallel(init='_init_fn', post='_post_fn', target='threads')
       def some_action(self, item, arg1, arg2):
           # process just one item
           return some_value

For further details see :doc:`how to make parallel actions <parallel>`.


Writing your own Batch
======================

Constructor should include `*args` and `*kwargs`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   class MyBatch(Batch):
       ...
       def __init__(self, index, your_param1, your_param2, *args, **kwargs):
           super().__init__(index)
           # process your data

It is not so important if you are extremely careful when calling batch generators and parallelizing actions, so you are absolutly sure that a batch cannot get unexpected arguments.
But usually it is just easier to add `*args` and `*kwargs` and have a guarantee that your program will not break or hang up (as it most likely will do if you do batch prefetching with multiprocessing).

Don't load data in the constructor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The constructor should just intialize properties.
`Action`-method `load` is the best place for reading data from files or other sources.

So DON'T do this::

   class MyBatch(Batch):
       ...
       def __init__(self, index, your_param1, your_param2, *args, **kwargs):
           super().__init__()
           ...
           self._data = read(file)

Instead DO that::

   class MyBatch(Batch):
       ...
       def __init__(self, index, your_param1, your_param2, *args, **kwargs):
           super().__init__(index)
           ...

       @action
       def load(self, src, fmt=None):
           # load data from source
           ...
           self.put_into_data(read(file))
           return self

Store your data in `_data` property
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is just a convenient convention which makes your life more consistent.

Use components
^^^^^^^^^^^^^^

Quite often a batch contains several semantic data parts, like images and labels, or transactions and ther scores.
For a more flexible data processing and covenient actions create data components. It takes just one line of code::

    class MyBatch(Batch):
        components = 'images', 'masks', 'labels'

See above `for more details about components <#components>`_.

Make `actions` whenever possible
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you create some method transforming batch data, you might want to call it as a step in a :doc:`pipeline` processing the whole dataset.
So make it an `action`::

   class MyBatch(Batch):
       ...
       @action
       def change_data(self, arg1, arg2):
           # process your data
           return self

`Actions` should return an instance of some batch class.

Parallelize everyting you can
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want a really fast data processing you can't do without `numba` or `cython`.
And don't forget about input/output operations.
For more details see :doc:`how to make a parallel actions <parallel>`.

Define `load` and `dump` action-methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`load` and `dump` allows for a convenient and managable data flow.::

   class MyBatch(Batch):
       ...
       @action
       def load(self, src, fmt='raw'):
           if fmt == 'raw':
               self.put_into_data(...) # load from a raw file
           elif fmt == 'blosc':
               self.put_into_data(...) # load from a blosc file
           else:
               super().load(src, fmt)
           return self

       @action
       def dump(self, dst, fmt='raw'):
           if fmt == 'raw':
               # write self.data to a raw file
           elif fmt == 'blosc':
               # write self.data to a blosc file
           else:
               super().dum(dst, fmt)
           return self

This lets you create explicit pipeline workflows::

   batch
      .load('/some/path', 'raw')
      .some_action(param1)
      .other_action(param2)
      .one_more_action()
      .dump('/other/path', 'blosc')

Make all I/O in `async` methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is extremely important if you read batch data from many files.::

   class MyBatch(Batch):
       ...
       @action
       def load(self, src, fmt='raw'):
           if fmt == 'raw':
               self.put_into_data(self._load_raw(src))
           elif fmt == 'blosc':
               self.put_into_data(self._load_blosc(src))
           else:
               raise ValueError("Unknown format '%s'" % fmt)
           return self

       @inbatch_parallel(init='_init_io', post='_post_io', target='async')
       async def _load_raw(self, item, full_path):
           # load one data item from a raw format file
           return loaded_item

       def _init_io(self):
           return [[item_id, self.index.get_fullpath(item_id)] for item_id in self.indices]

       def _post_io(self, all_res):
           if any_action_failed(all_res):
               raise IOError("Could not load data.")
           else:
               self.put_into_data(np.concatenate(all_res))
           return self

Make all I/O in `async` methods even if there is nothing to parallelize
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   class MyBatch(Batch):
       ...
       @inbatch_parallel(init='run_once', target='async')
       async def read_some_data(self, src, fmt='raw'):
           ...
   ...
   some_pipeline
       .do_whatever_you_want()
       .read_some_data('/some/path')
       .do_something_else()

Init-function `run_once` runs the decorated method once (so no parallelism whatsoever).
Besides, the method does not receive any additional arguments, only those passed to it directly.
However, an `action` defined as asynchronous will be waited for.
You may define your own `post`-method in order to check the result and process the exceptions if they arise.

API
---

See :doc:`Batch API <../api/batchflow.batch>`.
