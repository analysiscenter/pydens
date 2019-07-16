========================
Within batch parallelism
========================

Basic usage
===========

The ``inbatch_parallel`` decorator allows to run a method in parallel::

   from batchflow import Batch, inbatch_parallel, action

   class MyBatch(Batch):
       ...
       @action
       @inbatch_parallel(init='_init_fn', post='_post_fn', target='threads')
       def some_action(self, item, arg1, arg2):
           # process just one item
           return some_value

Parallelized methods
====================

You can parallelize actions as well as ordinary methods.

Decorator arguments
===================

init='some_method'
^^^^^^^^^^^^^^^^^^

Required.
The only required argument which contains a method name to be called to initialize the parallel execution.

post='other_method'
^^^^^^^^^^^^^^^^^^^

Optional.
A method name which is called after all parallelized tasks are finished.

target='threads'
^^^^^^^^^^^^^^^^

Optional.
Specifies a parallelization engine, should be one of ``threads``, ``async``, ``mpc``, ``for``.


Additional decorator arguments
==============================

You can pass any other arguments to the decorator and they will be passed further to ``init`` and ``post`` functions.

.. code-block:: python

   class MyBatch(Batch):
   ...
       @inbatch_parallel(init='_init_default', post='_post_default', target='threads', clear_data=False)
       def some_method(self, item):
           # process just one item
           return some_value

       def _init_default(self, clear_data):
           ...

       def _post_default(self, list_of_res, clear_data):
           ...

All these arguments should be named argments only. So you should not write like this::

   @inbatch_parallel('_init_default', '_post_default', 'threads', clear_data)

It might sometimes works though. But no guarantees.

The preferred way is::

   @inbatch_parallel(init='_init_default', post='_post_default', target='threads', clear_data=False)

Using this technique you can pass an action name to the ``init`` function::

   class MyBatch(Batch):
   ...
       @inbatch_parallel(init='_init_default', post='_post_default', target='threads', method='one_method')
       def one_method(self, item):
           # process just one item
           return some_value

       @inbatch_parallel(init='_init_default', post='_post_default', target='threads', method='some_other_method')
       def some_other_method(self, item):
           # process just one item
           return some_value

However, usually you might consider writing specific init / post functions for different actions.

Init function
=============

Init function defines how to parallelize the decorated method. It returns a list of arguments for each invocation of the parallelized action.
So if you want to run 10 parallel copies of the method, ``init`` should return a list of 10 items. Usually you run the method once for each item in the batch. However you might also run one method per 10 or 100 or any other number of items if it is beneficial for your specific circumstances (memory, performance, etc.)

The simplest ``init`` just returns a sequence of indices::

   class MyBatch(Batch):
   ...
       @action
       @inbatch_parallel(init='indices')
       def some_action(self, item_id)
           # process an item and return a value for that item
           return proc_value

For a batch of 10 items ``some_action`` will be called 10 times as ``some_action(index1)``, ``some_action(index2)``, ..., ``some_action(index10)``.

You may define as many arguments as you need::

   class MyBatch(Batch):
   ...
       def _init_fn(self, *args, **kwargs):
           return [[self._data, item, another_arg, one_more_arg] for item in self.indices]

Here the action will be fired as:

``some_action(self._data, index1, another_arg, one_more_arg)``

``some_action(self._data, index2, another_arg, one_more_arg)``

``...``

``item_args`` does not have to be strictly a list, but any sequence - tuple, numpy array, etc - that supports the unpacking operation (``*seq`` <https://docs.python.org/3/tutorial/controlflow.html#unpacking-argument-lists>`_\ ):

**Attention!** It cannot be a tuple of 2 arguments (see below why).

You can also pass named arguments::

   class MyBatch(Batch):
   ...
       def _init_fn(self, *args, **kwargs):
           return [dict(data=self._data, item=item, arg1=another_arg, arg2=one_more_arg) for item in self.indices]

And the action will be fired as:

``some_action(data=self._data, item=index1, arg1=another_arg, arg2=one_more_arg)``

``some_action(data=self._data, item=index2, arg1=another_arg, arg2=one_more_arg)``

``...``

And you can also combine positional and named arguments::

   class MyBatch(Batch):
   ...
       def _init_fn(self, *args, **kwargs):
           return [tuple(list(self._data, item), dict(arg1=another_arg, arg2=one_more_arg)) for item in self.indices]

So the action will be fired as:

``some_action(self._data, index1, arg1=another_arg, arg2=one_more_arg)``

``some_action(self._data, index2, arg1=another_arg, arg2=one_more_arg)``

``...``

Thus, 2-items tuple is reserved for this situation (1st item is a list of positional arguments and 2nd is a dict of named arguments).

That is why you cannot pass a tuple of 2 arguments::

       ...
       item_args = tuple(some_arg, some_other_arg)
       ...

Instead make it a list::

       ...
       item_args = list(some_arg, some_other_arg)
       ...

Init's additional arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Take into account that all arguments passed into actions are also passed into the ``init`` function. So when you call::

   some_pipeline.some_parallel_action(10, 12, my_arg=12)

The ``init`` function will be called like that::

   init_function(10, 12, my_arg=12)

This is convenient when you need to initialize some additional variables depending on the arguments. For instance, to create a numpy array of a certain shape filled with specific values or set up a random state or even pass additional arguments back to action methods.

If you have specified `additional decorator arguments <#additional-decorator-arguments>`_,
they are also passed to the ``init`` function::

   init_function(10, 12, my_arg=12, arg_from_parallel_decorator=True)


Standard init functions
^^^^^^^^^^^^^^^^^^^^^^^

Most of the times you don't need to write your own init function as you might use standard ones:

``indices``
~~~~~~~~~~~

.. code-block:: python

       @inbatch_parallel(init='indices')
       def some_method(self, ix, arg1, arg2):

The first argument (after ``self``) contains an id (from index) of each data item.

``items``
~~~~~~~~~

.. code-block:: python

       @inbatch_parallel(init='items')
       def some_method(self, item, arg1, arg2):

The first argument (after ``self``) contains an item itself (i.e. i-th element of batch - ``batch[i]``).

``run_once``
~~~~~~~~~~~~
You cannot call an ``async`` action in pipelines, because ``async``-methods should be ``awaited`` for. This is where ``@inbatch_parallel`` might be helpful without any parallelism whatsoever. All you need is ``run_once`` init-function::

   class MyBatch(Batch):
       ...
       @inbatch_parallel(init='run_once')
       async def read_some_data(self, src, fmt='raw'):
           ...
   ...
   some_pipeline
       .do_whatever_you_want()
       .read_some_data('/some/path')
       .do_something_else()

No additional arguments is passed to ``read_some_data`` and it will be executed only once.

data components
~~~~~~~~~~~~~~~

If data components are defined, they might be used as init-functions::

       @inbatch_parallel(init='images')
       def some_method(self, image, arg1, arg2):

The first argument (after ``self``) contains an i-th image (i.e. ``batch.images[i]``).

Post function
=============

When all parallelized tasks are finished, the ``post`` function is called.

The first argument it receives is the list of results from each parallel task.

.. code-block:: python

   class MyBatch(Batch):
       ...
       def _post_default(self, list_of_res, *args, **kwargs):
           ...
           return self

       @action
       @inbatch_parallel(init='indices', post='_post_default')
       def some_action(self, item_id)
           # process an item and return a value for that item
           return proc_value

Here ``_post_default`` will be called as

.. code-block:: python

   _post_default([proc_value_from_1, proc_value_from_2, ..., proc_value_from_last])

If anything went wrong, then instead of ``proc_value``, there would be an instance of some ``Exception`` caught in the parallel tasks.

This is where ``any_action_failed`` might come in handy:

.. code-block:: python

   from batchflow import Batch, action, inbatch_parallel, any_action_failed

   class MyBatch(Batch):
       ...
       def _post_fn(self, list_of_res, *args, **kwargs):
           if any_action_failed(list_of_res):
               # something went wrong
           else:
               # process the results
           return self

       @action
       @inbatch_parallel(init='indices', post='_post_fn')
       def some_action(self, item_id)
           # process an item and return a value for that item
           return proc_value

``Post``-function should return an instance of a batch class (not necessarily the same). Most of the time it would be just ``self``.

If an action-method changes data directly, you don't need a ``post``-function.

.. code-block:: python

   from batchflow import Batch, action, inbatch_parallel, any_action_failed

   class MyBatch(Batch):
       ...
       @action
       @inbatch_parallel(init='indices')
       def some_action(self, item_id)
           # process an item and return a value for that item
           self._data[item_id] = new_value

Don't forget about GIL. A python function with ``target=threads`` won't give any performance increase, though this might simplify your code.
However, ``numba`` or ``cython`` allow for a real multithreading.

.. code-block:: python

   from batchflow import Batch, action, inbatch_parallel, any_action_failed
   from numba import njit

   @njit(nogil=True)
   def change_data(data, index):
       # data is a numpy array
       data[index] = new_value


   class MyBatch(Batch):
       ...
       def _init_numba(self, *args, **kwargs):
           return [[self.data, i] for i in self.indices]

       @action
       @inbatch_parallel(init='_init_numba', target='threads')
       def some_action(self, data, item_id)
           return change_data(data, item_id)

Here all batch items will be updated simultaneously.

Targets
=======

There are 4 targets available: ``threads``, ``async``, ``mpc``, ``for``.

threads
^^^^^^^

A method will be parallelized with `concurrent.futures.ThreadPoolExecutor <https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor>`_.
Take into account that due to `GIL <https://wiki.python.org/moin/GlobalInterpreterLock>`_ only one python thread is executed in any given moment (pseudo-parallelism). `Cython <http://cython.org/>`_ and `numba <http://numba.pydata.org/>`_ might help overcome this limitation.
However, a usual python function with intensive I/O processing or waiting for some synchronization might get a considerable performance increase even with threads.

This is the default engine which is used if ``target`` is not specified in the ``inbatch_parallel`` decorator.

async
^^^^^

For I/O-intensive processing you might want to consider writing an ```async`` method <https://docs.python.org/3/library/asyncio-task.html>`_.

.. code-block:: python

   class MyBatch(Batch):
       ...
       @action
       @inbatch_parallel(init='_init_default', post='_post_default', target='async')
       async def some_action(self, item, some_arg)
           # do something
           proc_value = await other_async_function(some_arg)
           return proc_value

Specifying ``target='async'`` for methods declared as ``async`` is not necessary,
since in this case the decorator can determine that you need an ``async``-parallelism.
However, for a not ``async`` method returning awaitable objects you have to explicitly use ``target='async'``.

mpc
^^^

With ``mpc`` you might run calculations in separate processes thus removing GIL restrictions. For this `concurrent.futures.ProcessPoolExecutor <https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor>`_ is used. The decorated method should just return a function which will be executed in a separate process.

.. code-block:: python

   from batchflow import Batch, action, inbatch_parallel

   def mpc_fn(data, index, arg):
       # do something
       return new_data

   class MyBatch(Batch):
       ...
       @action
       @inbatch_parallel(init='_init_default', post='_post_default', target='mpc')
       def some_action(self, arg)
           # do not process anything, just return a function which will be run as a separate process
           return mpc_fn

Multiprocessing requires all code and data to be serialized (with `pickle <https://docs.python.org/3/library/pickle.html>`_\ ) in order to be sent to another process. And many classes and methods are not so easy (or even impossible) to pickle. That is why functions might be a better choice for parallelism. Nevertheless, with all these thoughts in mind you should carefully consider your parallelized function and the arguments it receives.

Besides, you might want to implement a thorough logging mechanism as multiprocessing configurations are susceptible to hanging up. Without logging it would be quite hard to understand what happened and debug your code.

for
^^^

When parallelism is not needed at all, you might still create actions which process single items, but they will be called one after another in a loop.
This is not only convenient but also might have a much better performance than ``mpc``-parallelism (e.g. when data is small, a lot of time is wasted to inter-process data flows).

It is also useful for debugging: you can replace ``mpc`` or ``threads`` with ``for`` in order to debug the code in a simple single-threaded fasion and then switch to parallel invocations.


Arguments with default values
=============================

If you have a function with default arguments, you may call it without passing those arguments.

.. code-block:: python

   class MyBatch(Batch):
       ...
       @action
       @inbatch_parallel(init='_init_default', post='_post_default', target='mpc')
       def some_action(self, arg1, arg2, arg3=3)
           ...

   # arg3 takes the default value = 3
   batch.some_action(1, 2)

However, when you call it this way, the default arguments are not available externally (in particular, in decorators).
This is the problem for ``mpc`` parallelism.

The best solutions would be not to use default values at all, but if you really need them, you should copy them into parallelized functions::

   def mpc_fn(item, arg1, arg2, arg3=10):
       # ...

   class MyBatch(Batch):
       ...
       @action
       @inbatch_parallel(init='_init_default', post='_post_default', target='mpc')
       def some_action(self, arg1, arg2, arg3=10)
           return mpc_fn

You might also return a `partial <https://docs.python.org/3/library/functools.html#functools.partial>`_ with these arguments::

   from functools import

   def mpc_fn(item, arg1, arg2, arg3=10):
       # ...

   class MyBatch(Batch):
       ...
       @action
       @inbatch_parallel(init='_init_default', post='_post_default', target='mpc')
       def some_action(self, arg1, arg2, arg3=10)
           return partial(mpc_fn, arg3=arg3)


Number of parallel jobs
=======================

By default each action runs as many parallel tasks as the number of cores your computer/server has. That is why sometimes you might want to run fewer or more tasks. Then you can specify this number in each action call with ``n_workers`` option::

   some_pipeline.parallel_action(some_arg, n_workers=3)

Here ``parallel_action`` will have only 3 parallel tasks being executed simultaneously. Others will wait in the queue.

However, implicitly specifying ``n_workers`` is rarely needed in practice and thus highly discouraged.

**Attention!** You cannot use ``n_workers`` with ``target=async``.


Writing numba-methods
=====================

When you need an extremely fast processing, `numba <http://numba.pydata.org/>`_ might come in handy.
However, it works only with functions, but not with methods. It is not a big issue as you can write
a method which calls a function. It is not convenient, though. And code becomes not so easy to follow.

That is why you'll love `@mjit` decorator. It looks absolutely like `@jit <https://numba.pydata.org/numba-doc/latest/user/jit.html/>`_, but it works with methods.

.. code-block:: python

   class MyBatch(Batch):
       ...
       @action
       @mjit
       def fast_loop(self, data):
           for i in range(data.shape[0]):
              data[i] = -np.exp(-np.exp(data[i]))

       @action
       @inbatch_parallel('images')
       @mjit
       def fast_parallel_loop(self, image):
           for i in range(image.shape[0]):
               for j in range(image.shape[1]):
                   image[i, j] = -np.exp(-np.exp(image[i, j]))

       @action
       @inbatch_parallel('images')
       @mjit
       def fast_parallel_action(self, image):
           image[:] = -np.exp(-np.exp(image))

`mjit` just takes a method body and compiles it with `numba.jit`. So the method should comply with all numba requirements.

.. note:: `self` cannot be used within `@mjit` methods. It will always be None.
          As a result, you have no access to any class attributes and methods.

.. note:: By default, a method is compiled with `nopython=True` and `nogil=True`.
          You can redefine these parameters when needed.

`prange <https://numba.pydata.org/numba-doc/latest/user/parallel.html>`_ is also allowed within `@mjit` methods.
`@inbatch_parallel` works as fast, though. So choose freely what is more convenient in each case.
