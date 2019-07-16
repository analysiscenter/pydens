===========
Research
===========

Research class is intended for multiple running of the same pipelines
with different parameters in order to get some metrics value.

Basic usage
-----------
Let's compare `VGG7` and `VGG16` performance on `MNIST` dataset with
different layouts of convolutional blocks. For each combination of
layout and model class, we train model for 1000 iterations and repeat
that process 10 times and save accuracy and loss on train and accuracy on test.

Firstly, import classes from `batchflow` to create pipelines:

.. code-block:: python

    from batchflow import B, C, V, F, Config
    from batchflow.opensets import MNIST
    from batchflow.models.tf import VGG7, VGG16

Define model config. All parameters that we want to vary we define
as ``C('parameter_name')``:

.. code-block:: python

    model_config={
        'session/config': tf.ConfigProto(allow_soft_placement=True),
        'inputs': dict(images={'shape': (28, 28, 1)},
                       labels={'classes': 10, 'transform': 'ohe', 'name': 'targets'}),
        'input_block/inputs': 'images',
        'body/block/layout': C('layout'),
        'output/ops': 'accuracy',
        'device': C('tf_device') # it's technical parameter for TFModel
    }

Strictly saying, the whole ``model_config`` with different
``'model_config/body/block/layout'`` is a pipeline parameter but due
to a substitution rule of named expressions you can define
named expression inside of `dict` or `Config` that is used as action parameter
(See :doc:`Named expressions <../intro/named_expr>`).

Define a dataset and train a pipeline:

.. code-block:: python

    mnist = MNIST()

    feed_dict = {'images': B('images'),
                 'labels': B('labels')}

    train_ppl = (mnist.train.p
                 .init_variable('loss', init_on_each_run=list)
                 .init_variable('accuracy', init_on_each_run=list)
                 .init_model('dynamic', C('model'), 'conv', config=model_config)
                 .to_array()
                 .train_model('conv',
                              fetches=['loss', 'output_accuracy'],
                              feed_dict={'images': B('images'), 'labels': B('labels')},
                              save_to=[V('loss'), V('accuracy')], mode='w')
                 .run(64, shuffle=True, n_epochs=None, lazy=True)
                )

Action parameters that we want to vary we define as ``C('model_class')``. Note
that to specify parameters of batch generating ``run`` action must be defined with ``lazy=True``.

Create an instance of `Research` class and add train pipeline:

.. code-block:: python

    research = Research()
    research.add_pipeline(train_ppl, variables='loss', name='train')

Parameter ``name`` defines pipeline name inside ``research``. At each iteration
that pipeline will be executed with ``.next_batch()`` and all ``variables`` from the pipeline
will be saved so that variables must be added with ``mode='w'``.

All parameter combinations we define through the dict where a key is
a parameter name and value is a list of possible parameter values.
Create a grid of parameters and add to ``research``:

.. code-block:: python

    grid_config = {'model_class': [VGG7, VGG16], 'layout': ['cna', 'can']}
    research.add_grid(grid_config)

You can get all variants of config by ``list(grid.gen_configs())``:

::

    [ConfigAlias({'layout': 'cna', 'model': 'VGG7'}),
     ConfigAlias({'layout': 'cna', 'model': 'VGG16'}),
     ConfigAlias({'layout': 'can', 'model': 'VGG7'}),
     ConfigAlias({'layout': 'can', 'model': 'VGG16'})]

Each element is a ConfigAlias. It's a Config dict of parameter values
and dict with aliases for parameter values.

In order to control test accuracy we create test pipeline and add it
to ``research``:

.. code-block:: python

    test_ppl = (mnist.test.p
                .init_variable('accuracy', init_on_each_run=list)
                .import_model('conv', C('import_from'))
                .to_array()
                .predict_model('conv',
                               fetches=['output_accuracy'],
                               feed_dict={'images': B('images'), 'labels': B('labels')},
                               save_to=[V('accuracy')], mode='a')
                .run(64, shuffle=True, n_epochs=1, lazy=True)
                )

    research.add_pipeline(test_ppl, variables='accuracy', name='test', run=True, execute='%100', import_model='train')

That pipeline will be executed with ``.run()`` each 100 iterations because
of parameters ``run=True``  and ``execute='%100'``. Pipeline variable ``accuracy``
will be saved after each execution. In order to add a mean value of accuracy
on the whole test dataset, you can define a function

.. code-block:: python

    def get_accuracy(iteration, experiment, pipeline):
        import numpy as np
        pipeline = experiment[pipeline].pipeline
        acc = pipeline.get_variable('accuracy')
        return np.mean(acc)

and then add it into research:

.. code-block:: python

    research.add_function(get_accuracy, returns='accuracy', name='test_accuracy', execute='%100', pipeline='test')

That function will get iteration, experiment, args and kwargs
(in that case it's ``pipeline='test'"``).

Experiment is an OrderedDict for all pipelines and functions
that were added to Research and are running in current job.
Key is a name of ExecutableUnit (class for function and pipeline),
value is ExecutableUnit. Each pipeline and function added to Research
is saved as an ExecutableUnit. Each ExecutableUnit has the following
attributes:

::

    function : callable
        is None if `Executable` is a pipeline
    pipeline : Pipeline
        is None if `Executable` is a function
    root_pipeline : Pipeline
        is None if `Executable` is a function or pipeline is not divided into root and branch
    dataset : Dataset or None
        dataset for pipelines
    part : str or None
        part of dataset to use
    cv_split : int or None
        partition of dataset
    result : dict
        current results of the `Executable`. Keys are names of variables (for pipeline)
        or returns (for function) values are lists of variable values
    path : str
        path to the folder where results will be dumped
    exec : int, list of ints or None
    dump : int, list of ints or None
    to_run : bool
    variables : list
        variables (for pipeline) or returns (for function)
    on_root : bool
    args : list
    kwargs : dict()


Note that we use ``C('import_model')`` in ``import_model`` action
and add test pipeline with parameter ``import_model='train'``.
All ``kwargs`` in ``pipeline`` are used to define
parameters that depend on another pipeline in the same way.

Method ``run`` starts computations:

.. code-block:: python

    research.run(n_reps=10, n_iters=1000, name='my_research', bar=True)

All results will be saved as
``{research_name}/results/{config_alias}/{repetition_index}/{unitname}_{iteration}``
as pickled dict (by dill) where keys are variable names and values are lists
of corresponding values.

There is method ``load_results`` to create ``pandas.DataFrame`` with results
of the research.

Parallel runnings
-----------------

If you have a lot of GPU devices (say, 4) you can do research faster,
just define ``workers=4``
and ``gpu = [0, 1, 2, 3]`` as a list of available devices.
In that case you can run 4 jobs in parallel!

.. code-block:: python

    research.run(n_reps=10, n_iters=1000, workers=4, gpu=[0,1,2,3], name='my_research', bar=True)

In that case, two workers will execute tasks in different processes
on different GPU. If you use `TorchModel`, add parameter `framework='torch'` to `run`.

Another way of parallel running
--------------------------------

If you have heavy loading you can do it just one time for few pipelines
with models. In that case devide pipelines into root and branch:

.. code-block:: python

    mnist = MNIST()
    train_root = mnist.train.p.run(64, shuffle=True, n_epochs=None, lazy=True)

    train_branch = (Pipeline()
                .init_variable('loss', init_on_each_run=list)
                .init_variable('accuracy', init_on_each_run=list)
                .init_model('dynamic', C('model'), 'conv', config=model_config)
                .to_array()
                .train_model('conv',
                             fetches=['loss', 'output_accuracy'],
                             feed_dict={'images': B('images'), 'labels': B('labels')},
                             save_to=[V('loss'), V('accuracy')], mode='w')
    )


Then define research in the following way:

.. code-block:: python

    research = (Research()
        .add_pipeline(root=train_root, branch=train_branch, variables='loss', name='train')
        .add_pipeline(test_ppl, variables='accuracy', name='test', run=True, execute='%100', import_model='train')
        .add_grid(grid)
        .add_function(get_accuracy, returns='accuracy', name='test_accuracy', execute='%100', pipeline='test')
    )

And now you can define the number of branches in each worker:

.. code-block:: python

    research.run(n_reps=2, n_iters=1000, workers=2, branches=2, gpu=[0,1,2,3], name='my_research', bar=True)


Dumping of results and logging
--------------------------------

By default if unit has varaibles or returns then results
will be dumped at last iteration. But there is unit parameter dump
that allows to save result not only in the end. It defines as execute
parameter. For example, dump train results each 200 iterations.
Besides, each research has log file. In order to add information about
unit execution and dumping into log, define ``logging=True``.

.. code-block:: python

    research = (Research()
        .add_pipeline(root=train_root, branch=train_template,
                  variables='loss', name='train', dump='%200')
        .add_pipeline(test_ppl,
                  variables='accuracy', name='test', run=True, execute='%100', import_from='train', logging=True)
        .add_grid(grid)
        .add_function(get_accuracy, returns='accuracy', name='test_accuracy', execute='%100', pipeline='test')
    )

    research.run(n_reps=2, n_iters=1000, workers=2, branches=2, gpu=[0,1,2,3], name='my_research', bar=True)

First worker will execute two branches on GPU 0 and 1
and the second on the 2 and 3.

Functions on root
--------------------------------

All functions and pipelines if branches > 0 executed in parallel
threads so sometime it can be a problem. In order to allow run
function in main thread there exists parameter on_root. Function
that will be added with on_root=True will get iteration, experiments
and kwargs. experiments is a list of experiments that was defined above
(OrderedDict of ExecutableUnits). Simple example of usage:

.. code-block:: python

    def on_root(iteration, experiments):
        print("On root", iteration)

    research = (Research()
        .add_function(on_root, on_root=True, execute=10, logging=True)
        .add_pipeline(root=train_root, branch=train_template, variables='loss', name='train')
        .add_pipeline(root=test_root, branch=test_template,
                  variables='accuracy', name='test', run=True, execute='%100', import_from='train', logging=True)
        .add_grid(grid)
        .add_function(get_accuracy, returns='accuracy', name='test_accuracy', execute='%100', pipeline='test')
    )

That function will be executed just one time on 10 iteration
and will be executed one time for all branches in task.

.. code-block:: python

    research.run(n_reps=1, n_iters=100, workers=2, branches=2, gpu=[0,1,2,3], name='my_research', bar=True)

Logfile:

::

    INFO     [2018-05-15 14:18:32,496] Distributor [id:5176] is preparing workers
    INFO     [2018-05-15 14:18:32,497] Create queue of jobs
    INFO     [2018-05-15 14:18:32,511] Run 2 workers
    INFO     [2018-05-15 14:18:32,608] Start Worker 0 [id:26021] (gpu: [0, 1])
    INFO     [2018-05-15 14:18:32,709] Start Worker 1 [id:26022] (gpu: [2, 3])
    INFO     [2018-05-15 14:18:41,722] Worker 0 is creating process for Job 0
    INFO     [2018-05-15 14:18:49,254] Worker 1 is creating process for Job 1
    INFO     [2018-05-15 14:18:53,101] Job 0 was started in subprocess [id:26082] by Worker 0
    INFO     [2018-05-15 14:18:53,118] Job 0 has the following configs:
    {'layout': 'cna', 'model': 'VGG7'}
    {'layout': 'cna', 'model': 'VGG16'}
    INFO     [2018-05-15 14:18:59,267] Job 1 was started in subprocess [id:26130] by Worker 1
    INFO     [2018-05-15 14:18:59,281] Job 1 has the following configs:
    {'layout': 'can', 'model': 'VGG7'}
    {'layout': 'can', 'model': 'VGG16'}
    INFO     [2018-05-15 14:19:02,415] J 0 [26082] I 11: on root 'unit_0' [0]
    INFO     [2018-05-15 14:19:02,415] J 0 [26082] I 11: on root 'unit_0' [1]
    INFO     [2018-05-15 14:19:07,803] J 0 [26082] I 100: dump 'unit_0' [0]
    INFO     [2018-05-15 14:19:07,803] J 0 [26082] I 100: dump 'unit_0' [1]
    INFO     [2018-05-15 14:19:08,761] J 1 [26130] I 11: on root 'unit_0' [0]
    INFO     [2018-05-15 14:19:08,761] J 1 [26130] I 11: on root 'unit_0' [1]
    INFO     [2018-05-15 14:19:12,050] J 0 [26082] I 100: execute 'test' [0]
    INFO     [2018-05-15 14:19:12,051] J 0 [26082] I 100: execute 'test' [1]
    INFO     [2018-05-15 14:19:12,051] J 0 [26082] I 100: dump 'test' [0]
    INFO     [2018-05-15 14:19:12,051] J 0 [26082] I 100: dump 'test' [1]
    INFO     [2018-05-15 14:19:12,056] Job 0 [26082] was finished by Worker 0
    INFO     [2018-05-15 14:19:14,149] J 1 [26130] I 100: dump 'unit_0' [0]
    INFO     [2018-05-15 14:19:14,149] J 1 [26130] I 100: dump 'unit_0' [1]
    INFO     [2018-05-15 14:19:18,819] J 1 [26130] I 100: execute 'test' [0]
    INFO     [2018-05-15 14:19:18,819] J 1 [26130] I 100: execute 'test' [1]
    INFO     [2018-05-15 14:19:18,820] J 1 [26130] I 100: dump 'test' [0]
    INFO     [2018-05-15 14:19:18,820] J 1 [26130] I 100: dump 'test' [1]
    INFO     [2018-05-15 14:19:18,825] Job 1 [26130] was finished by Worker 1
    INFO     [2018-05-15 14:19:18,837] All workers have finished the work

Cross validation
--------------------------------
To run pipelines with cross-validation divide each pipeline
into dataset and pipeline with actions and then add it into research:

.. code-block:: python

    research.add_pipeline(train_template, dataset=mnist, part='train', variables='loss', name='train')

Parameter `part` describe what part of the dataset should be used.
Then run research with additional parameter `n_splits`:

.. code-block:: python

    research.run(workers=4, n_iters=5000, gpu=[4,5,6,7], n_splits=5, name='my_research', bar=True)

In the folder with results will be added additional subfolder and
the full path is
``{research_name}/results/{config_alias}/{repetition_index}/cv{index}/{unitname}_{iteration}``.
The resulting DataFrame will have column `cv_split`.

API
---

See :doc:`Research API <../api/batchflow.research>`.
