=======================
Inter-batch parallelism
=======================

For long-running :doc:`pipelines <pipeline>` you might employ a `prefetch` feature which allows for a parallel batch processing.

.. code-block:: python

   for batch in some_pipeline.gen_batch(BATCH_SIZE, prefetch=3):
       ...

This line states that 3 additional batches should be processed in the background.
Take into account that all the batches will be processed simultaneously without any prioritization.
However, the order of batches is preserved, i.e. batch #2 would be returned after batch #1 even if all the actions for batch #2 finished earlier. This statement is correct even for shuffled order (though it might seem illogical to some people).

Let's look at an example. Here is a simple pipeline:

.. code-block:: python

   some_pipeline = some_dataset.p.action1().action2().action3()

At first, we run it without `prefetch`:

.. code-block:: python

   for batch in some_pipeline.gen_batch(BATCH_SIZE):
       ...

The execution sequence will look as follows::

   batch #1 is created
       action 1 started
       action 1 finished
       action 2 started
       action 2 finished
       action 3 started
       action 3 finished
   batch #2 is created
       action 1 started
       action 1 finished
       action 2 started
       action 2 finished
       action 3 started
       action 3 finished
   batch #3 is created
       action 1 started
       action 1 finished
       action 2 started
       action 2 finished
       action 3 started
       action 3 finished

So, all the batches start one after another, and all the actions also start one after another.

Now use `prefetch`:

.. code-block:: python

   for batch in some_pipeline.gen_batch(BATCH_SIZE, prefetch=1):
       ...

This changes the execution sequence dramatically::

   batch #1 is created
   batch #2 is created
       action 1 started for batch #1
       action 1 started for batch #2
       action 1 finished for batch #1
       action 2 started for batch #1
       action 1 finished for batch #2
       action 2 started for batch #2
       action 2 finished for batch #1
       action 3 started for batch #1
       action 2 finished for batch #2
       action 3 started for batch #2
       action 3 finished for batch #1
   batch #3 is created
       action 1 started for batch #3
       action 3 finished for batch #2 <-- after that batch #4 could start
       action 1 finished for batch #3
       action 2 started for batch #3
       action 2 finished for batch #3
       action 3 started for batch #3
       action 3 finished for batch #3

Batch #2 is created immediately after batch #1. Then all the actions are executed for all running batches.
As actions execution time might vary between batches, the actual sequence might look different in your case.

However, the main principle remains the same - `prefetch` parameter indicates how many additional batches should be processed in advance, before you need them. As a consequence, when you need them, they will be returned much faster or even almost immediately (if all the actions have been executed already).

You can use `prefetch` in `next_batch`\ , `gen_batch` and `run`.

Blocked method
^^^^^^^^^^^^^^

Sometimes you might want to guarantee that only one call of a specific action is executed simultaneously, e.g. due to a race condition or dependence on some external resources. To make this happen provide a lock to an action:

.. code-block:: python

   class MyBatch(Batch):
       ...
       @action(use_lock=True)
       def only_one(self):
           ...
       ...

       @action(use_lock="unique_lock_name")
       def also_one(self):
           ...

Thus, whenever you make prefetching, only one batch at a time will execute `only_one` action.
