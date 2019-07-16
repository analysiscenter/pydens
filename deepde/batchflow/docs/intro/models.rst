===================
Working with models
===================

Pipelines can include model definitions, training, evauluation and prediction actions which links models and pipelines.


Model class
===========

Models are defined in model classes. You can take ready to use architectures or write your own models.::

   from batchflow.models import BaseModel

   class MyModel(BaseModel):
       def _build(self, *args, **kwargs):
           #... do whatever you want


Model types
===========

There are two modes of model definition:

* static
* dynamic

A static model is created when the pipeline is created, i.e. before it is executed.
As a result, the model might have an access to the pipeline, its config and variables.

A dynamic model is created during the pipeline's execution, when some action uses the model the first time.
Consequently, the model has an access to the pipeline and the very first batch thus allowing to create models adapting
to batch size, content, shape and data types.


.. _init_a_model:

Adding a model to a pipeline
============================

First of all, a model should be initialized::

   full_workflow = my_dataset.p
                             .init_model('static', MyModel, 'my_model', config)
                             ...

In :meth:`~batchflow.Pipeline.init_model()` you state a mode (``static`` or ``dynamic``), a model class, an optional short model name (otherwise, a class name will be used) and an optional configuration.
A static model is initialized immediately in the ``init_model``, while a dynamic model will be initialized when the pipeline is run and the very first batch flows into the pipeline.

If a model was already created in another pipeline, it might be `imported <#importing-models>`_.


Configuring a model
===================

Most often than not models have many options and hyperparameters which define the model structure
(e.g. number and types of layers for a neural network or a number and depth of trees for forests),
as well as a training procedure (e.g. an optimization algorithm or regularization constants).

Global options:

* ``build`` : bool - whether to call ``model.build(...)`` to create a model. Default is ``True``.
* ``load`` : dict - parameters for model loading from some storage. If present, a model will be loaded by calling `self.load(**config['load'])`.

Loading usually requires some additional config parameters like paths, file names or file formats. Check the documentation for the model you use for more details.

For some models only one of ``build`` or ``load`` should be specified. While other models might need a building phase even if a model is loaded from a disk.

Read a model specification to know how to configure it.

For flexibilty ``config`` might include so called :doc:`named expressions <named_expr>` which are defined by name but substitued with their actual values:

* ``B('name')`` - a batch component or attribute
* ``V('name')`` - a pipeline variable
* ``C('name')`` - a pipeline config option
* ``F(name)`` - a function, method or any other callable

::

   pipeline
       .init_variable('images_shape', [256, 256])
       .init_model('static', MyModel, config={'input_shape': V('images_shape')})

   pipeline
       .init_variable('shape_name', 'images_shape')
       .init_model('dynamic', MyModel, config={V('shape_name)': B('images_shape')})

   pipeline
       .init_model('dynamic', MyModel, config={'input_shape': F(lambda batch: batch.images.shape[1:])})


Training a model
================

A train action should be stated below an initialization action::

   full_workflow = (my_dataset.p
       .init_model('static', MyModel, 'my_model', config)
       ...
       .train_model('my_model', x=B('images'), y=B('labels'))
   )

:meth:`~batchflow.Pipeline.train_model` arguments might be specific to a particular model you use. So read a model specfication to find out what it expects for training.

Model independent arguments are:

* ``make_data`` - a function or method which takes a current batch and a model instance and return a dict of arguments for ``model.train(...)``.
* ``save_to`` - a location or a sequence of locations where to store an output of ``model.train`` (if there is any).
  Could be :doc:`a named expression <named_expr>`: ``B("name")``, ``C("name")`` or ``V("name")``.
* ``mode`` - could be one of:

  * ``'w'`` or ``'write'`` to rewrite a location with a new value
  * ``'a'`` or ``'append'`` to append a value to a location (e.g. if a location is a list)
  * ``'e'`` or ``'extend'`` to extend a location with a new value (e.g. if a location is a list and a value is a list too)
  * ``'u'`` or ``'update'`` to update a location with a new value (e.g. if a location is a dict).

  For sets and dicts ``'u'`` and ``'a'`` do the same.

::

   full_workflow = (my_dataset.p
       .init_model('static', MyModel, 'my_model', my_config)
       .init_model('dynamic', AnotherModel, 'another_model', another_config)
       .init_variable('current_loss', 0)
       .init_variable('current_accuracy', 0)
       .init_variable('loss_history', init_on_each_run=list)
       ...
       .train_model('my_model', output=['loss', 'accuracy'], x=B('images'), y=B('labels'),
                    save_to=[V('current_loss'), V('current_accuracy')])
       .train_model('another_model', fetches='loss',
                    feed_dict={'x': B('images'), 'y': B('labels')},
                    save_to=V('loss_history'), mode='append')
   )

Here, parameters ``output``, ``x`` and ``y`` are specific to ``my_model``, while ``fetches`` and ``feed_dict`` are specific to ``another_model``.

You can also write an action which works with a model directly.::

   class MyBatch(Batch):
       ...
       @action(model='some_model')
       def train_linked_model(self, model):
           ...

       @action
       def train_in_batch(self, model_name):
           model = self.get_model_by_name(model_name)
           ...


   full_workflow = (my_dataset.p
       .init_model('static', MyModel, 'my_model', my_config)
       .init_model('dynamic', MyOtherModel, 'some_model', some_config)
       .some_preprocessing()
       .some_augmentation()
       .train_in_batch('my_model')
       .train_linked_model()
   )


Predicting with a model
=======================

:meth:`~batchflow.Pipeline.predict_model` is very similar to `train_model <#training-a-model>`_ described above::

   full_workflow = (my_dataset.p
       .init_model('static', MyModel, 'my_model', config)
       .init_variable('predicted_labels', init_on_each_run=list)
       ...
       .predict_model('my_model', x=B('images'), save_to=V('predicted_labels'))
   )

Read a model specfication to find out what it needs for predicting and what its output is.


.. _loading_a_model:

Loading a model
===============

A model can be loaded into a pipeline::

   some_pipeline.load_model('dynamic', ResNet18, 'my_model', path='/some/path')

The parameters are the same as in :ref:`the model initalization <init_a_model>`.

Note, that :meth:`~batchflow.Pipeline.load_model` just adds a loading action to the pipeline, but the actual loading
will happen only when pipeline is being executed.

Also take into account that ``load_model`` will be called at each iteration which might be desired or undesired depending
on the specific circumstances.

To load model only once before the pipeline is executed you might use :ref:`before <after_pipeline>` pipeline::

    some_pipeline.before.load_model('dynamic', ResNet18, 'my_model', path='/some/path')

There is also and imperative :meth:`~batchflow.Pipeline.load_model_now`, i.e. it loads a model immediately, and not when a pipeline is executed.
Thus, it cannot be a part of a pipeline's chain of actions. ``load_model_now`` is expected to be called in an action method or before a training
or inference pipeline is run (e.g. before `pipeline.run <pipeline#running-pipelines>`_).


.. _saving_a_model:

Saving a model
==============

You can write a model to a persistent storage at any time by calling ``save_model(...)``::

   some_pipeline.save_model('my_model', path='/some/path')

As usual, the first argument is a model name, while all other arguments are model specific, so read a model documentation
to find out what parameters are required to save a model.

Note, that :meth:`~batchflow.Pipeline.save_model` just adds a saving action to the pipeline, but the actual saving
will happen only when pipeline is being executed.

Also take into account that ``save_model`` will be called at each iteration which might be desired or undesired depending
on the specific circumstances.

To save model only once after the pipeline you might use :ref:`after <after_pipeline>` pipeline::

    some_pipeline.after.save_model('my_model', path='/some/path')

There is also and imperative :meth:`~batchflow.Pipeline.save_model_now`, i.e. it saves a model immediately, and not when a pipeline is executed.
Thus, it cannot be a part of a pipeline's chain of actions. ``save_model_now`` is expected to be called in an action method or after a training pipeline has finished
(e.g. after `pipeline.run <pipeline#running-pipelines>`_).


Models and template pipelines
=============================

A template pipeline is not linked to any dataset and thus it will never run. It might be used as a building block for more complex pipelines.::

   template_pipeline = (Pipeline()
       .init_model('static', MyModel)
       .init_model('dynamic', MyModel2)
       .prepocess()
       .normalize()
       .train_model('MyModel', ...)
       .train_model('MyModel2', ...)
   )

Linking a pipeline to a dataset creates a new pipeline that can be run.::

   mnist_pipeline = (template_pipeline << mnist_dataset).run(BATCH_SIZE, n_epochs=10)
   cifar_pipeline = (template_pipeline << cifar_dataset).run(BATCH_SIZE, n_epochs=10)

Take into account, that a static model is created only once in the template_pipeline.
But it will be used in each children pipeline with different datasets (which might be a good or bad thing).

Whilst, a separate instance of a dynamic model will be created in each children pipeline.


Importing models
================

Models exist within pipelines. This is very convenient if a single pipeline includes everything: preprocessing,
model training, model evaluation, model saving and so on. However, sometimes you might want to share a model between
pipelines. For instance, when you train a model in one pipeline and later use it in an inference pipeline.

This can be easily achieved with a model import.::

   train_pipeline = (images_dataset.p
       .init_model('dynamic', Resnet50)
       .load(...)
       .random_rotate(angle=(-30, 30))
       .train_model("Resnet50")
       .run(BATCH_SIZE, shuffle=True, n_epochs=10)
   )

   inference_pipeline_template = (Pipeline()
       .resize(shape=(256, 256))
       .normalize()
       .import_model("Resnet50", train_pipeline)
       .predict_model("Resnet50")
   )
   ...

   infer = (inference_pipeline_template << some_dataset).run(INFER_BATCH_SIZE, shuffle=False)

When ``inference_pipeline_template`` is run, the model ``Resnet50`` from ``train_pipeline`` will be imported.
If you still have questions about import_model, search the answer in :meth:`~batchflow.Pipeline.import_model`.


Parallel training
=================

If you :doc:`prefetch <prefetch>` with actions based on non-thread-safe models, you might encounter that your model
hardly learns anything. The reason is that model variables might not update concurrently. To solve this problem a lock
can be added to an action to allow for only one concurrent execution::

   class MyBatch(Batch):
       ...
       @action(use_lock="some_model_lock")
       def train_it(self, model_name):
           model = self.get_model_by_name(model_name)
           model.train(input_images=self.images, input_labels=self.labels)
           return self

However, as far as ``TensorFlow`` is concerned, its optimizers have a parameter `use_locking <https://www.tensorflow.org/api_docs/python/tf/train/Optimizer#__init__>`_
which allows for concurrent updates when set to ``True``.


Ready to use models
===================
See documentation for :doc:`Tensorflow <tf_models>` and :doc:`Torch models <torch_models>` and
the list of :doc:`implemented architectures <model_zoo>`.


Model metrics
=============
Module :doc:`models.metrics <../api/batchflow.models.metrics>` comes in handy to evaluate model performance.
It contains many useful metrics (sensitivity, specificity, accuracy, false discovery rate and many others)
for different scenarios (2-class and multiclass classification, pixel-wise and instance-wise semantic segmentation).

Models can be evaluated in a one-shot manner when you pass `targets` and `predictions`::

    metrics = ClassificationMetrics(targets, predictions, num_classes=10, fmt='labels')
    metrics.evaluate(['sensitivity', 'specificity'], multiclass='macro')

Or in a pipeline::

    pipeline = (test_dataset.p
        .init_variables(['metrics', 'inferred_masks'])
        .import_model('unet', train_pipeline)
        .predict_model('unet', fetches='predictions', feed_dict={'x': B('images')},
                       save_to=V('inferred_masks'))
        .gather_metrics(SegmentationMetricsByPixels, targets=B('masks'), predictions=V('inferred_masks'),
                        fmt='proba', save_to=V('metrics'), mode='u')
        .run(BATCH_SIZE)
    )

    metrics = pipeline.get_variable('metrics')
    print(metrics.evaluate(['sensitivity', 'specificity']))

For more information about metrics see :doc:`metrics API <../api/batchflow.models.metrics>` and :meth:`~.Pipeline.gather_metrics`.
