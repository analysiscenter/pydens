====================
A short introduction
====================


Index
=====

Index holds a sequence of data item ids. As a dataset is split into batches, you should have a mechanism to uniquely address each data item.
In simple cases it can be just a `numpy.arange`:

.. code-block:: python

    dataset_index = DatasetIndex(np.arange(my_array.shape[0]))

`FilesIndex` is helpful when your data comes from multiple files.

.. code-block:: python

    dataset_index = FilesIndex("/path/to/files/*.png")

Most of the times creating an index in one line of code is all you need to do about a dataset index.

For more details see :doc:`How to work with an Index <dsindex>`.


Dataset
=======

A dataset consists of an index (1-d sequence with unique keys per each data item) and a batch class which processes small subsets of data.

.. code-block:: python

    client_ds = Dataset(dataset_index, batch_class=Batch)

Now you can iterate over sequential or random batches:

.. code-block:: python

    batch = client_ds.next_batch(BATCH_SIZE, shuffle=True, n_epochs=3)

You will rarely need anything than creating a dataset in one line of code,
but you may always dig deeper into :doc:`how to work with datasets <dataset>`.


Batch
=====

Batch class holds the data and contains processing functions.
Normally, you never create batch instances, as they are created in the `Dataset` or `Pipeline` batch generators.

See more info about :doc:`useful batch methods and actions and how to create your own batch class <batch>`.


Pipeline
========

After a batch class is created, you can define a processing workflow for the whole dataset:

.. code-block:: python

    my_pipeline = my_dataset.pipeline()
                    .load('/some/path')
                    .some_processing()
                    .another_processing()
                    .save('/other/path')
                    .run(BATCH_SIZE, shuffle=False)

All the methods here (except `run`) are :doc:`actions from the batch class <batch>`.

Now you are ready for a deeper immersion into :doc:`how to create, use and run pipelines <pipeline>`.


Within-batch parallelism
========================

In order to accelerate data processing you can run batch methods in parallel:

.. code-block:: python

    from batchflow import Batch, inbatch_parallel, action

    class MyBatch(Batch):
        ...
        @action
        @inbatch_parallel(init='_init_fn', post='_post_fn', target='threads')
        def some_action(self, item):
            # process just one item from the batch
            return some_value

:doc:`How to make parallel methods <parallel>`.


Inter-batch parallelism
=======================

To further increase pipeline performance and eliminate inter batch delays you may process several batches in parallel:

.. code-block:: python

    some_pipeline.next_batch(BATCH_SIZE, prefetch=3)

The parameter `prefetch` defines how many additional batches will be processed in the background.

See more indo about :doc:`prefetching <prefetch>`.


Models
======

Mostly, pipelines are needed to train machine learning models or predict using these models.

See :doc:`Working with models <models>` to understand what a model is and how to use it within pipelines.

There is a bunch of :doc:`predefined models <model_zoo>` which you can use out of the box.


Research
========
To perform multiple experiments with different parameters you can use `Research` class:

.. code-block:: python

    from batchflow.research import Research
    ...
    research = (Research()
                .add_pipeline(train_pipeline, variables='loss', name='train')
                .add_pipeline(test_pipeline, variables='accuracy', name='test', import_model_from='train')
                .add_grid_config('model_class': [VGG7, VGG16], 'layout': ['cna', 'can'])
                .run(n_reps=10, n_iters=1000))

See more indo about :doc:`Research <research>`.
