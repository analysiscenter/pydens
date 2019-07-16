=====
Index
=====

Why the index is needed?
========================

A dataset may be so large that it does not fit into memory and thus you cannot process it at once. That is why each data item in the `Dataset` should have an unique id. It does not have to be meaningful (like a card number or a transaction id), sometimes it may be just a hash or an ordered number. However, each index item should address exactly one data item (which in turn can have a complex structure, like a list, an array, a dataframe, or even a graph).

The value of ids in the index is important only in 2 situations:

* in `load` `action-method <batch#action-methods>`_ , when batch gets data from some external source like `batch_items = external_source[batch.indices]` and thus the external source should contain those indices, otherwise `load` will fail. Similarly, when data is loaded from files, indices usually point to those files and their full paths (see `FilesIndex`_ below).
* in item selection - `batch[some_item_id]` - so the index should contain the id you're referring to.

Evereywhere else the particular id value is pretty meaningless as all operations use an item position in the index, not its id.

.. _DatasetIndex:

DatasetIndex
============

`DatasetIndex` is a base index class which stores a sequence of unique ids for your data items. In the simplest case it might be just an ordered sequence of numbers (0, 1, 2, 3,..., e.g. `numpy.arange(len(dataset))`\ ).

.. code-block:: python

   dataset_index = DatasetIndex(np.arange(my_array.shape[0]))

In other cases it can be a list of domain-specific identificators (e.g. client ids, product codes, serial numbers, timestamps, etc).

.. code-block:: python

   dataset_index = DatasetIndex(dataframe['client_id'])

You will rarely need to work with an index directly, but if you want to do something specific you may use its :doc:`public API <../api/batchflow.index>`.

.. _FilesIndex:

FilesIndex
==========

When data comes from a file system, it might be convenient to use `FilesIndex`.

.. code-block:: python

   files_index = FilesIndex(path="/path/to/some/files/*.csv")

Thus `files_index` will contain the list of filenames that match a given mask.
The details of mask specification may be found in the :func:`~glob.glob` documentation.

No file extensions
^^^^^^^^^^^^^^^^^^

When filenames contain extensions which are not a part of the id, then they may be stripped with an option `no_ext`\ ::

   dataset_index = FilesIndex(path="/path/to/some/files/*.csv", no_ext=True)

Sorting
^^^^^^^

Since order may be random, you may want to sort your index items::

   dataset_index = FilesIndex(path="/path/to/some/files/*.csv", sort=True)

However, this rarely makes any sense.

Directories
^^^^^^^^^^^

Sometimes you need directories, not files. For instance, a CT images dataset includes one subdirectory per each patient, it is named with patient id and contains many images of that patient. So the index should be built from these subdirectories, and not separate images.

.. code-block:: python

   dirs_index = FilesIndex(path="/path/to/archive/2016-*/scans/*", dirs=True)

Here `dirs_index` will contain a list of all subdirectories names.

Numerous sources
^^^^^^^^^^^^^^^^

If files you are interested in are located in different places you may still build one united index::

   dataset_index = FilesIndex(["/current/year/data/*", "/path/to/archive/2016/*", "/previous/years/*"])

Creating your own index class
-----------------------------

Constructor
^^^^^^^^^^^

We highly recommend to use the following pattern::

   class MyIndex(DatasetIndex):
       def __init__(self, index, my_arg, *args, **kwargs):
           # initialize new properties
           super().__init__(index, my_arg, *args, **kwargs)
           # do whatever you need

So to summarize:


#. the parent class should be `DatasetIndex` or its child
#. include `*args` and `**kwargs` in the constructor definition
#. pass all the arguments to the parent constructor

build_index
^^^^^^^^^^^

You might want to redefine `build_index` method which actually creates the index.
It takes all the arguments from the constructor and returns a numpy array with index items.
This method is called automatically from the :class:`~batchflow.DatasetIndex` constructor.

API
---

See :doc:`Index API <../api/batchflow.index>`.
