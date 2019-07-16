=================
Tensorflow models
=================

Getting started
===============
A model might be used for training or inference. In both cases you need to specify a model config and a pipeline.

A typical minimal config includes ``inputs`` and ``initial_block`` sections::

    model_config = {
        'inputs': dict(images={'shape': (128, 128, 3)},
                       targets={'classes': 10}),
        'initial_block/inputs': 'images'
    }

A minimal training pipeline consists of :meth:`~.Pipeline.init_model` and :meth:`~.Pipeline.train_model`::

    pipeline = my_dataset.p
        .init_model('dynamic', MyModel, 'my_model', model_config)
        .train_model('my_model', fetches='loss', images=B('images'), labels=B('labels'))
        .run(BATCH_SIZE, shuffle=True, n_epochs=5)

To create an inference pipeline replace ``train_model`` with :meth:`~.Pipeline.predict_model`.

A pipeline can also :meth:`~.TFModel.load` a pretrained model which was previously :ref:`saved <saving_a_model>` to a disk::

    model_config = {
        'build': 'first',
        'load': dict(path='/path/to/model'),
    }

    pipeline = my_dataset.p
        .init_model('dynamic', MyModel, 'my_model', model_config)
        .predict_model('my_model', fetches='predictions', images=B('images'))
        .run(BATCH_SIZE)

Note that you can indicate through 'build' option whether a model graph needs to be created or updated by calling :meth:`~TFModel.build`.
Most often than not 'build' should be `False` (which is a default value), but sometimes it might be convenient to create the graph
before loading or change the graph after loading.

Specify `build='first'` to create the graph and load a pretrained model afterwards.
While `build=True` means that the model will be loaded first and after that `model.build()` will be executed thus allowing to change the graph.


Model structure
===============
A typical model comprises of

- initial_block
- body (which, in turn, might include blocks)
- head.

This division might seem somewhat arbitrary, though, many modern networks follow it.


initial_block
-------------
This block just transforms the raw inputs into more managable and initially preprocessed tensors.

Some networks do not need this (like VGG). However, most network have 1 or 2 convolutional layers
and sometimes also a max pooling layer with stride 2. These layers can be put into body, as well.
But the initial block takes all irregular front layers, thus allowing for a regular body structure.


body
----
Body contains a repetitive structure of building blocks. Most networks (like VGG, ResNet and the likes) have a straight sequence of blocks, while others (e.g. UNet, LinkNet, RefineNet, ResNetAttention) look like graphs with many interconnections.

Initial block's output goes into body as inputs.
And body's output is a compressed representation (embedding) of the input tensors.
It can later be used for various tasks: classification, regression, detection, etc.
So ``body`` produces a task-independent embedding.


block
-----
The network building block reflects the model's unique logic and specific technology.

Not surprisingly, many networks comprise different types of blocks, for example:

- UNet and LinkNet have encoder and decoder blocks
- Inception includes inception, reduction, and expanded blocks
- DenseNet have dense and transition blocks
- SqueezeNet alternates fire blocks with max-pooling.

When creating a custom model you can have as many block types as you need, though aim to make them universal and reusable elsewhere.
For instance, :class:`~.tf.LinkNet`, :class:`~.tf.GlobalConvolutionNetwork`, and :class:`~.tf.ResNetAttention` use :class:`~.tf.ResNet` blocks.


head
----
It receives body's output and produces a task-specific result, for instance, class logits for classification.
The default head consists of one :func:`.conv_block`. So, by specifying a model's config you can
instantiate models for different tasks.

Classification with 10 classes::

    config = {
        ...
        'loss': 'ce',
        'inputs': dict(images={'shape': (128, 128, 3)},
                       labels={'classes': 10})
        'head': dict(layout='cdV', filters=10, dropout_rate=.2),
        'initial_block/inputs': 'images'
    }

Regression::

    config = {
        ...
        'loss': 'mse',
        'inputs': dict(heart_signals={'shape': (4000, 1)},
                       targets={'shape': 1})
        'head': dict(layout='df', units=1, dropout_rate=.2),
        'initial_block/inputs': 'heart_signals'
    }


How to configure a model
========================

Configuration options may vary between models. However, some parameters are available in many (if not all) models.

inputs
------
Inputs section contains a description of model input data, its shapes, transformations needed and names of the resulting tensors.

Each input might have following parameters:
    ``dtype`` : str or tf.DType (by default 'float32')
        data type

    ``shape`` : int or tuple / list
        a tensor shape which includes the number of channels/classes and doesn't include a batch size.

    ``classes`` : array-like or int
        an array of class labels if data labels are strings or just a number of classes

    ``data_format`` : str {``'channels_first'``, ``'channels_last'``} or {``'f'``, ``'l'``}
        The ordering of the dimensions in the inputs. Default is 'channels_last'.

    ``transform`` : str or callable
        Predefined transforms are

        - ``'ohe'`` - one-hot encoding
        - ``'mip @ d'`` - maximum intensity projection :func:`~.layers.mip` with depth ``d`` (should be int)

    ``name`` : str
        a name for the transformed tensor.

Even though all parameters are optional, at least some of them should be specified for each input tensor.

For instance, this config will create placeholders with the names ``images`` and ``targets``::

    model_config = {
        'inputs': dict(images={'shape': (128, 128, 3)},
                       targets={'classes': 10}),
    }

Later, names ``images`` and ``targets`` will be used to feed data into the model when training or predicting.
Take into account that one-hot encoding is not required for labels when using cross-entropy loss as it is applied
automatically. However, for custom losses one-hot encoding might be necessary.

For more information on the configuration of the inputs, see :meth:`~batchflow.models.tf.TFModel._make_inputs`.

Models based on :class:`.TFModel` expect that one of the inputs has a name ``targets`` (before or after transformations),
while model output turns into a tensor named ``predictions``. These tensors are used to define a model loss function.

Initial block
-------------
Initial block specifies which inputs flow into the model to turn into prediction::

    model_config = {
        'initial_block/inputs': 'images',
    }

As the default initial block contains a :func:`~.tf.layers.conv_block`, all its parameters might be also specfied in the config::

    model_config = {
        'initial_block': dict(layout='cnap', filters=64, kernel_size=7, strides=2),
        'initial_block/inputs': 'images',
    }

So the configured initial block gets `images` tensor and applies a convolution with 7x7 kernel and stride 2.

For :doc:`predefined models <model_zoo_tf>` an initial block has the default configuration in accordance with the original paper.
So you almost never need to redefine it.

However, ``initial_block/inputs`` should always be specified.

Initial block might be defined as a callable as well::

    model_config = {
        'initial_block': my_initial_block_fn,
        'initial_block/inputs': 'images',
    }



body
----
Body is the main part of a model. Thus its configuration highly depends on the model structure and purpose.

For instance, :class:`~.tf.ResNet` body config includes ``block`` section with specific residual block parameters. While :class:`~.tf.UNet` body contains ``upsample`` section which specifies the technique to resize tensors in a decoder part of the network.

See the model documentation to find out how to configure its body.

Body is usually defined as a dict, but might also take a callable::

    model_config = {
        'body': my_network_ops_fn,
    }


head
----
For many models head is just another :func:`~.layers.conv_block`. So you may configure layout, the number of filters, dense layer units or other parameters. As usual, it is rarely needed for predefined models.

Head can also be defined with a callable.


predictions
-----------
It defines the content of ``predictions`` tensor which is used in the configured model loss.

Available operations are:
    - None - do nothing (identity operation)
    - callable - apply a given function to an output tensor
    - 'proba' - softmax
    - 'sigmoid' - sigmoid
    - 'labels' - argmax
    - 'softplus' - softplus.

Mostly, the predictions tensor is just the head output and thus it needs no configuration.
However, different losses might require different predictions (e.g. cross entropy expects logits,
while some custom loss might expect probabilities).

output
------
Output defines the auxiliary operations which are applied to the head output.
These operations can be used during model training or evaluation.

For instance, 'proba' gets probabilities if head outputs logits::

    {'output': ['proba', 'labels']}

For advanced usage see :meth:`~.TFModel.output`.


Loss, learning rate decay, optimizer
------------------------------------

These parameters might be defined in one of three formats:

- name
- tuple (name, args)
- dict {'name': name, ...other args}

where name might be one of:
- short name (‘mse’, ‘ce’, ‘l1’, ‘cos’, ‘hinge’, ‘huber’, ‘logloss’, ‘dice’)
- function name from ``TensorFlow`` (e.g. ‘absolute_difference’ or ‘sparse_softmax_cross_entropy’)
- callable.

For example::

    {'loss': 'mse'}
    {'loss': {'name': 'sigmoid_cross_entropy', 'label_smoothing': 1e-6}}
    {'loss': (tf.losses.huber_loss, {'reduction': tf.losses.Reduction.MEAN})}
    {'loss': external_loss_fn}

Available short names for losses: mse, ce, l1, cos, hinge, huber, logloss, dice.

::

    {'decay': 'exp'}
    {'decay': ('polynomial_decay', {'decay_steps': 10000})}
    {'decay': {'name': tf.train.inverse_time_decay, 'decay_rate': .5}

Short names for decay: exp, invtime, naturalexp, const, poly.

::

    {'optimizer': 'Adam'}
    {'optimizer': ('Ftlr', {'learning_rate_power': 0})}
    {'optimizer': {'name': 'Adagrad', 'initial_accumulator_value': 0.01}}
    {'optimizer': functools.partial(tf.train.MomentumOptimizer, momentum=0.95)}
    {'optimizer': some_optimizer_fn}

Short names for optimizer: 'Adam', 'Adagrad', 'GradientDescent' and any other optimizer from
`tf.train <https://www.tensorflow.org/api_docs/python/tf/train>`_ without the word `Optimizer`.

.. note:: To allow for parallel training `use_locking <https://www.tensorflow.org/api_docs/python/tf/train/Optimizer#__init__>`_ should be set to ``True``. For example::

    {'optimizer': 'Adam', 'use_locking': True}

For more detail see :class:`.TFModel` documentation.


How to write a custom model
===========================

To begin with, take a look at :ref:`conv_block <conv_block>` to find out how to write complex networks in just one line of code.
This block is a convenient building block for concise, yet very expressive neural networks.

The simplest case you should avoid
----------------------------------
Just redefine ``body()`` method.

For example, let's create a small fully convolutional network with 3 layers of 3x3 convolutions, batch normalization, dropout
and a dense layer at the end::

    from batchflow.models.tf import TFModel
    from batchflow.models.tf.layers import conv_block

    class MyModel(TFModel):
        def body(self, inputs, **kwargs):
            x = conv_block(inputs, layout='cna cna cna df', filters=[64, 128, 256], units=10, kernel_size=3,
                           dropout_rate=.2, **kwargs)
            return x

Despite simplicity, this approach is highly discouraged as:

- the model parameters are hard coded in the body
- the model cannot be configured within a pipeline
- the model does not allow model composition, i.e. using this model components in other models.

The right way
-------------
Here we split network configuration and network definition into separate methods::

    from batchflow.models.tf import TFModel
    from batchflow.models.tf.layers import conv_block

    class MyModel(TFModel):
        @classmethod
        def default_config(cls):
            config = TFModel.default_config()
            config['body'].update(dict(filters=[64, 128, 256], kernel_size=3, layout='cna cna cna'))
            config['head'].update(dict(units=2, layout='df', dropout_rate=.2))
            return config

        def build_config(self, names=None):
            config = super().build_config(names)
            config['head/units'] = self.num_classes('targets')
            config['head/filters'] = self.num_classes('targets')
            return config

        @classmethod
        def body(cls, inputs, name='body', **kwargs):
            kwargs = cls.fill_params('body', **kwargs)
            x = conv_block(inputs, **kwargs)
            return x

Note that ``default_config`` and ``body`` are ``@classmethods`` now, which means that they might be called without
instantiating a ``MyModel`` object.
This is needed for model composition, e.g. when ``MyModel`` serves as a base network for an FCN or SSD network.

On the other hand, ``build_config`` is an ordinary method, so it is called only when an instance of ``MyModel`` is created.

Thus, ``default_config`` should contain all the constants which are totaly independent of the dataset
and a specific task at hand, while ``build_config`` is intended to extract values from the dataset through pipeline's configuration (for details see `Configuring a model <models#configuring-a-model>`_).

Now you can train the model with a simple pipeline::

    model_config = {
        'loss': 'ce',
        'decay': 'invtime',
        'optimizer': 'Adam',
        'inputs': dict(images={'shape': (128, 128, 3)},
                       labels={'classes': 10}),
        'initial_block/inputs': 'images'
    }

    pipeline = my_dataset.p
        .init_variable('loss_history', init_on_each_run=list)
        .init_model('dynamic', MyModel, 'my_model', model_config)
        .train_model('my_model', fetches='loss',
                     images=B('images'), labels=B('labels'),
                     save_to=V('loss_history'), mode='a')
        .run(BATCH_SIZE, shuffle=True, n_epochs=5)

To switch to a fully convolutional head with 3x3 convolutions and global average pooling,
just add 1 line to the config::

    model_config = {
        ...
        'head/layout': 'cV'
    }

As a result, the very same model class might be used

- in numerous scenarios
- with different configurations
- for various tasks
- with heterogenous data.


Things worth mentioning:

#. Override :meth:`~.TFModel.initial_block`, :meth:`~.TFModel.body` and :meth:`~.TFModel.head`, if needed.
   In many cases config is just enough to build a network without additional code writing.

#. Input data and its parameters should be defined in configuration under ``inputs`` key.
   See :meth:`.TFModel._make_inputs` for details.

#. You might want to use a convenient multidimensional :func:`.conv_block` and other predefined :doc:`layers <tf_layers>`.
   Of course, you can use usual `tensorflow layers <https://www.tensorflow.org/api_docs/python/tf/layers>`_.

#. If you make dropout, batch norm, etc by hand, you might use a predefined ``self.is_training`` tensor.

#. For decay and training control there is a predefined ``self.global_step`` tensor.

#. In many cases there is no need to write a loss function, learning rate decay and optimizer
   as they might be defined through config.

#. For a configured loss one of the inputs should have a name ``targets`` and
   one of the output tensors should be named ``predictions``.

#. If you have defined your own loss function, call `tf.losses.add_loss(...)
   <https://www.tensorflow.org/api_docs/python/tf/losses/add_loss>`_.

#. If you need to use your own optimizer, assign ``self.train_step`` to the train step operation.
   Don't forget about `UPDATE_OPS control dependency
   <https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization>`_.


Ready to use models
===================

.. toctree::
   :maxdepth: 2

   ../api/batchflow.models.tf.models
