# pylint: disable=undefined-variable, no-name-in-module
""" Contains base class for tensorflow models """

import os
import glob
import re
import threading

import dill
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

from ... import Config
from ..utils import unpack_fn_from_config
from ..base import BaseModel
from .layers import mip, conv_block, upsample
from .losses import softmax_cross_entropy, dice
from .train import piecewise_constant


LOSSES = {
    'mse': tf.losses.mean_squared_error,
    'bce': tf.losses.sigmoid_cross_entropy,
    'ce': softmax_cross_entropy,
    'crossentropy': softmax_cross_entropy,
    'absolutedifference': tf.losses.absolute_difference,
    'l1': tf.losses.absolute_difference,
    'cosine': tf.losses.cosine_distance,
    'cos': tf.losses.cosine_distance,
    'hinge': tf.losses.hinge_loss,
    'huber': tf.losses.huber_loss,
    'logloss': tf.losses.log_loss,
    'dice': dice,
}

DECAYS = {
    'exp': tf.train.exponential_decay,
    'invtime': tf.train.inverse_time_decay,
    'naturalexp': tf.train.natural_exp_decay,
    'const': piecewise_constant,
    'poly': tf.train.polynomial_decay,
}


class TFModel(BaseModel):
    r""" Base class for all tensorflow models

    **Configuration**

    ``build`` and ``load`` are inherited from :class:`.BaseModel`.

    device : str or sequence of str
        device name(s), e.g. '/device:GPU:0' (TensorFlow-like format), 'gpu:1:, 'CPU:0'.
        Regular expressions are allowed, e.g. 'GPU:*'.
        Default behaviour is to use the first available GPU (or CPU if no GPUs are detected).
        See `tf.device <https://www.tensorflow.org/api_docs/python/tf/device>`_ for details.

        If multiple devices are at use, batch size must be divisible by the number of used devices.
        If microbatch is also used, then microbatch size must be divisible by the number of devices.

    session : dict
        parameters for session configuration. 'allow_soft_placement' is always True. To learn more, check
        `Tensorflow ConfigProto parameters <https://www.tensorflow.org/api_docs/python/tf/ConfigProto`_.

    inputs : dict
        model inputs (see :meth:`.TFModel._make_inputs`)

    loss - a loss function, might be defined in one of three formats:
        - name
        - tuple (name, args)
        - dict {'name': name, \**args}

        where name might be one of:
            - short name (`'mse'`, `'ce'`, `'l1'`, `'cos'`, `'hinge'`, `'huber'`, `'logloss'`, `'dice'`)
            - a function name from `tf.losses <https://www.tensorflow.org/api_docs/python/tf/losses>`_
              (e.g. `'absolute_difference'` or `'sparse_softmax_cross_entropy'`)
            - callable

        It is possible to compute loss not only with network output and ground truth, but with
        any named Tensors in model by passing `'predictions'` and `'targets'` keywords.

        If loss is a callable, then it should add the result to a loss collection.
        Otherwise, ``add_loss`` should be set to True. An optional collection might also be specified through
        ``loss_collection`` parameter.

        .. note:: Losses from non-default collections won't be detected automatically,
                  so you should process them within your code.

        Examples:

        - ``{'loss': 'mse'}``
        - ``{'loss': {'name': 'sigmoid_cross_entropy', 'label_smoothing': 1e-6}}``
        - ``{'loss': (tf.losses.huber_loss, {'reduction': tf.losses.Reduction.MEAN})}``
        - ``{'loss': {'name': 'dice', 'predictions': 'body_output', 'targets': 'body_targets'}``
        - ``{'loss': external_loss_fn_with_add_loss_inside}``
        - ``{'loss': external_loss_fn_without_add_loss, 'add_loss': True}``
        - ``{'loss': external_loss_fn_to_collection, 'add_loss': True, 'loss_collection': tf.GraphKeys.LOSSES}``

    decay - a learning rate decay algorithm might be defined in one of three formats:
        - name
        - tuple (name, args)
        - dict {'name': name, **args}

        where name might be one of:

        - short name ('exp', 'invtime', 'naturalexp', 'const', 'poly')
        - a function name from `tf.train <https://www.tensorflow.org/api_docs/python/tf/train>`_
          (e.g. 'exponential_decay')
        - a callable

        Examples:

        - ``{'decay': 'exp'}``
        - ``{'decay': ('polynomial_decay', {'decay_steps': 10000})}``
        - ``{'decay': {'name': tf.train.inverse_time_decay, 'decay_rate': .5}``

    scope - subset of variables to optimize during training. Can be either string or sequence of strings.
        Value `''` is reserved for optimizing all trainable variables.
        Putting `-` sign before name stands for complement: optimize everything but the passed scope.

        Examples:

        - ``{'scope': ''}``
        - ``{'scope': 'body/custom_layer'}``
        - ``{'scope': '-body/custom_layer'}``
        - ``{'scope': ['body/custom_layer_1', 'head/custom_layer_2']}``

    optimizer - an optimizer might be defined in one of three formats:
            - name
            - tuple (name, args)
            - dict {'name': name, \**args}

            where name might be one of:

            - short name (e.g. 'Adam', 'Adagrad', any optimizer from
              `tf.train <https://www.tensorflow.org/api_docs/python/tf/train>`_ without a word `Optimizer`)
            - a function name from `tf.train <https://www.tensorflow.org/api_docs/python/tf/train>`_
              (e.g. 'FtlrOptimizer')
            - a callable

        Examples:

        - ``{'optimizer': 'Adam'}``
        - ``{'optimizer': ('Ftlr', {'learning_rate_power': 0})}``
        - ``{'optimizer': {'name': 'Adagrad', 'initial_accumulator_value': 0.01}``
        - ``{'optimizer': functools.partial(tf.train.MomentumOptimizer, momentum=0.95)}``
        - ``{'optimizer': some_optimizer_fn}``

    train_steps - configuration of different training procedures.
        Must be a mapping from string names to dictionary with train parameters like
        loss, decay, scope, optimizer. Those keys support syntax defined above.
        If any of loss, decay, scope, optimizer is defined in config, it serves as default
        value for every train step.
        Optimizer and decay, created at one train step, can be used in another. To do so, one can
        pass 'use' key with value corresponding to the name of train step from which you want to borrow optimizer.
        Note that in this case you are still free to change loss-function or scope.

        In order to use particular train step during train, one must pass `train_mode` argument
        to `train` method.

        Examples:

        - ``{'train_steps': {'all': {'loss': 'ce', 'optimizer': 'Adam', 'scope': ''},
                             'body': {'loss': 'dice', 'optimizer': 'RMSProp', 'scope': 'body'}}
                             'custom': {'use': 'body', 'loss': 'ce', 'scope': 'head'}}``

    microbatch : int
        size of chunks to split every batch in. Allows to process given data sequentially, accumulating gradients
        from microbatches and applying them once in the end. Can be changed later in the `train` method.
        Note that the microbatch size must be a divisor of the batch size.

    common : dict
        default parameters for all :func:`.conv_block`

    initial_block : dict
        parameters for the input block, usually :func:`.conv_block` parameters.

        The only required parameter here is ``initial_block/inputs`` which should contain a name or
        a list of names from ``inputs`` which tensors will be passed to ``initial_block`` as ``inputs``.

        Examples:

        - ``{'initial_block/inputs': 'images'}``
        - ``{'initial_block': dict(inputs='features')}``
        - ``{'initial_block': dict(inputs='images', layout='nac nac', filters=64, kernel_size=[7, 3], strides=[1, 2])}``

    body : dict
        parameters for the base network layers, usually :func:`.conv_block` parameters

    head : dict
        parameters for the head layers, usually :func:`.conv_block` parameters

    predictions : str or callable
        an operation applied to the head output to make the predictions tensor which is used in the loss function.

    output : dict or list
        auxiliary operations

    For more details about predictions and auxiliary output operations see :meth:`.TFModel.output`.

    **How to create your own model**

    #. Take a look at :func:`~.layers.conv_block` since it is widely used as a building block almost everywhere.

    #. Define model defaults (e.g. number of filters, batch normalization options, etc)
       by overriding :meth:`.TFModel.default_config`.
       Or skip it and hard code all the parameters in unpredictable places without the possibility to
       change them easily through model's config.

    #. Define build configuration (e.g. number of classes, etc)
       by overriding :meth:`~.TFModel.build_config`.

    #. Override :meth:`~.TFModel.initial_block`, :meth:`~.TFModel.body` and :meth:`~.TFModel.head`, if needed.
       In many cases defaults and build config are just enough to build a network without additional code writing.

    Things worth mentioning:

    #. Input data and its parameters should be defined in configuration under ``inputs`` key.
       See :meth:`.TFModel._make_inputs` for details.

    #. You might want to use a convenient multidimensional :func:`.conv_block`,
       as well as :func:`~.layers.global_average_pooling`, :func:`~.layers.mip`, or other predefined layers.
       Of course, you can use usual `tensorflow layers <https://www.tensorflow.org/api_docs/python/tf/layers>`_.

    #. If you make dropout, batch norm, etc by hand, you might use a predefined ``is_training`` tensor. You can get it
       by :meth:`~.TFModel.get_from_attr`.

    #. For decay and training control there is a predefined ``global_step`` tensor. You can get it
       by :meth:`~.TFModel.get_from_attr`.

    #. In many cases there is no need to write a loss function, learning rate decay and optimizer
       as they might be defined through config.

    #. If you have defined your own loss function, call `tf.losses.add_loss(...)
       <https://www.tensorflow.org/api_docs/python/tf/losses/add_loss>`_.
    """

    def __init__(self, *args, **kwargs):
        self.session = kwargs.get('session', None)
        self.graph = tf.Graph() if self.session is None else self.session.graph
        self._graph_context = None
        self._full_config = Config()
        self._train_lock = threading.Lock()

        # Parameters of batch processing: splitting batches into parts and/or using multiple devices to process data
        self.microbatch = None
        self.devices = []
        self.leading_device = None
        self.device_to_scope = {}
        self.scope_to_device = {}
        self.multi_device = False

        # Private storage for often used tensors
        self._attrs = dict()

        # Save/load things
        self._saver = None
        self.preserve = ['_attrs', 'microbatch',
                         'devices', 'leading_device', 'device_to_scope', 'scope_to_device', 'multi_device']

        super().__init__(*args, **kwargs)

    def store_to_attr(self, attr, graph_item, device=None):
        """ Store `graph_item` to private container."""
        if device is None:
            self._attrs[attr] = graph_item
        else:
            if self._attrs.get(attr) is None:
                self._attrs[attr] = {device: graph_item}
            else:
                self._attrs[attr].update({device: graph_item})

    def get_from_attr(self, attr, device=None, default=None):
        """ Get item from private container or directly from model graph."""
        device = device or self._get_current_device() or self.leading_device
        if attr in self._attrs:
            if isinstance(self._attrs[attr], dict):
                if device in self._attrs[attr]:
                    return self._attrs[attr][device]
            return self._attrs[attr]
        if default is not None:
            return default
        return self._check_tensor(attr, device)

    def _check_tensor(self, name, device=None):
        prefix = self.__class__.__name__ + '/'
        if device is not None:
            if device in self.device_to_scope.keys():
                prefix += self.device_to_scope[device]
            else:
                prefix += device

        pattern = '^' + prefix + '.*' + name + '.*'
        valid = [item for item in self.graph.get_operations() if re.match(pattern, item.name)]
        if len(valid) > 1:
            valid = [item for item in valid if re.match('.*_output$', item.name)]
            if len(valid) != 1:
                raise KeyError("Too many tensors match the '%s' name in  %s model" % (name, type(self).__name__))

        if len(valid) == 1:
            return valid[0].values()[0]
        raise KeyError("Model %s does not have '%s' tensor" % (type(self).__name__, name))

    def build(self, *args, **kwargs):
        """ Build the model. """
        # Get list of all available devices, infer leading device and number of devices
        self.devices = self._get_devices()
        if len(self.devices) > 1:
            self.multi_device = len(self.devices)
        self.leading_device = self.devices[0]

        self.device_to_scope = {item: item[1:].replace(':', '_') for item in self.devices}
        self.scope_to_device = {v: k for k, v in self.device_to_scope.items()}

        # Create model graph. First of all, `is_training` and `global_step` tensors are defined;
        # then, for each device, model architecture is created (with inputs placeholders and all);
        # finally, individual train steps with desired loss, optimizer, decay and scope are created
        with self.graph.as_default():
            with tf.variable_scope(self.__class__.__name__):
                with tf.variable_scope('globals'):
                    is_training = tf.placeholder(tf.bool, name='is_training')
                    self.store_to_attr('is_training', is_training)

                    global_step = tf.Variable(0, trainable=False, name='global_step')
                    self.store_to_attr('global_step', global_step)

                for device in self.devices:
                    with tf.device(device):
                        with tf.variable_scope(self.device_to_scope[device]):
                            config = self.build_config()
                            self._full_config = config
                            self._build(config)

                self.microbatch = config.get('microbatch')

                if self.session is None:
                    self.create_session(config)

                self._make_train_steps(config)
                self.reset()

    def create_session(self, config=None):
        """ Create TF session """
        config = config if config is not None else self.config
        session_config = config.get('session', default={})
        session_config = {**session_config, **{'allow_soft_placement': True}}
        self.session = tf.Session(config=tf.ConfigProto(**session_config))

    def reset(self):
        """ Reset the trained model to allow a new training from scratch """
        with self.session.graph.as_default():
            self.session.run(tf.global_variables_initializer())

    def _get_devices(self):
        available_devices = device_lib.list_local_devices()

        # Remove internal `XLA` devices, see `using JIT compilation <https://www.tensorflow.org/xla/jit>`_.
        usable_devices = [device.name for device in available_devices
                          if 'XLA' not in device.name]

        if self.config.get('device'):
            devices = self.config.get('device')
            devices = devices if isinstance(devices, list) else [devices]
            devices = [device for name in devices for device in usable_devices
                       if re.search(name.upper(), device.upper()) is not None]
            devices = [device for i, device in enumerate(devices)
                       if device not in devices[:i]]
        else:
            cpu_devices = [device for device in usable_devices
                           if 'CPU' in device]
            gpu_devices = [device for device in usable_devices
                           if 'GPU' in device]
            if gpu_devices:
                devices = [gpu_devices[0]]
            else:
                devices = [cpu_devices[0]]
        return devices

    def _get_current_device(self):
        scope = tf.get_variable_scope().name
        if '/' in scope:
            device_scope = scope.split('/')[1]
            if device_scope in self.scope_to_device:
                return self.scope_to_device[device_scope]
        return None

    def _make_inputs(self, names=None, config=None):
        """ Create model input data from config provided

        In the config's inputs section it looks for ``names``, creates placeholders required, and
        makes some typical transformations (like one-hot-encoding), if needed.

        **Configuration**

        inputs : dict
            - key : str
                a placeholder name
            - values : str or dict or tuple
                each input's config

        Input config:

        ``dtype`` : str or tf.DType (by default 'float32')
            data type

        ``shape`` : int, tuple, list or None (default)
            a tensor shape which includes the number of channels/classes and doesn't include a batch size.

        ``classes`` : int, array-like or None (default)
            a number of class or an array of class labels if data labels are strings or anything else except
            ``np.arange(num_classes)``.

        ``data_format`` : str {'channels_first', 'channels_last'} or {'f', 'l'}
            The ordering of the dimensions in the inputs. Default is 'channels_last'.
            For brevity ``data_format`` may be shortened to ``df``.

        ``transform`` : str or callable
            Predefined transforms are

            - ``ohe`` - one-hot encoding
            - ``mip @ d`` - maximum intensity projection :func:`~.layers.mip` with depth ``d`` (should be int)
            - ``downsample @ d`` - NN downsampling with a factor ``d`` (should be int)

        ``name`` : str
            a name for the transformed tensor.

        If an input config is a tuple, it should contain all items exactly in the order shown above:
        dtype, shape, classes, data_format, transform, name.
        If an item is None, the default value will be used instead.

        **How it works**

        A placholder with ``dtype``, ``shape`` and with a name ``key`` is created first.
        Then it is transformed with a ``transform`` function in accordance with ``data_format``.
        The resulting tensor will have the name ``name``.

        **Aliases**
        If an input config is a string, it should point to another key from inputs dict.
        This creates an alias to another input which might be convenient to substitute tensor names.

        By default, `targets` is aliased to `labels` or `masks` if present.

        Parameters
        ----------
        names : list
            placeholder names that are expected in the config's 'inputs' section

        Raises
        ------
        KeyError if there is any name missing in the config's 'inputs' section.
        ValueError if there are duplicate names.

        Returns
        -------
        placeholders : dict
            key : str
                a placeholder name
            value : tf.Tensor
                placeholder tensor
        tensors : dict
            key : str
                a placeholder name
            value : tf.Tensor
                an input tensor after transformations
        """
        # pylint:disable=too-many-statements
        full_config = config
        config = full_config.get('inputs')

        device = self._get_current_device()

        names = names or []
        missing_names = set(names) - set(config.keys())
        if len(missing_names) > 0:
            raise KeyError("Inputs should contain {} names".format(missing_names))

        placeholder_names = set(config.keys())
        tensor_names = set(x.get('name') for x in config.values() if isinstance(x, dict) and x.get('name'))
        wrong_names = placeholder_names & tensor_names
        if len(wrong_names) > 0:
            raise ValueError('Inputs contain duplicate names:', wrong_names)

        # add default aliases
        if 'labels' in config and 'targets' not in config:
            config['targets'] = 'labels'
        elif 'masks' in config and 'targets' not in config:
            config['targets'] = 'masks'
        # if targets is defined in the input dict, these implicit aliases will be overwritten.

        param_names = ('dtype', 'shape', 'classes', 'data_format', 'transform', 'name')
        defaults = dict(data_format=full_config.get('common/data_format', default='channels_last'))

        placeholders = dict()
        tensors = dict()
        _inputs = dict()
        for input_name, input_config in config.items():
            if isinstance(input_config, str):
                continue
            elif isinstance(input_config, (tuple, list)):
                input_config = list(input_config) + [None for _ in param_names]
                input_config = input_config[:len(param_names)]
                input_config = dict(zip(param_names, input_config))
                input_config = dict((k, v) for k, v in input_config.items() if v is not None)
            input_config = {**defaults, **input_config}

            reshape = None
            shape = input_config.get('shape')
            if isinstance(shape, int):
                shape = (shape,)
            if shape:
                input_config['shape'] = shape
                shape = [None] + list(shape)

            _inputs[input_name] = dict(config=input_config)
            self.store_to_attr('_inputs', _inputs)

            if self.has_classes(input_name):
                dtype = input_config.get('dtype', tf.int64)
                shape = shape or (None,)
            else:
                dtype = input_config.get('dtype', 'float')
            tensor = tf.placeholder(dtype, shape, input_name)
            placeholders[input_name] = tensor
            self.store_to_attr(input_name, tensor, device)

            if 'df' in input_config and 'data_format' not in input_config:
                input_config['data_format'] = input_config['df']
            if input_config.get('data_format') == 'l':
                input_config['data_format'] = 'channels_last'
            elif input_config.get('data_format') == 'f':
                input_config['data_format'] = 'channels_first'

            _inputs[input_name] = dict(config=input_config)
            self.store_to_attr('_inputs', _inputs)
            tensor = self._make_transform(input_name, tensor, input_config)

            if isinstance(reshape, (list, tuple)):
                tensor = tf.reshape(tensor, [-1] + list(reshape))

            name = input_config.get('name')
            if name is not None:
                tensor = tf.identity(tensor, name=name)
                self.store_to_attr(name, tensor, device)

            tensors[input_name] = tensor

            _inputs[input_name] = dict(config=input_config, placeholder=placeholders[input_name], tensor=tensor)
            if name is not None:
                _inputs[name] = _inputs[input_name]
            self.store_to_attr('_inputs', _inputs)

        # check for aliases
        for input_name, input_config in config.items():
            if isinstance(input_config, str) and input_name not in _inputs:
                _inputs[input_name] = _inputs[input_config]
                tensors[input_name] = tensors[input_config]
                placeholders[input_name] = placeholders[input_config]
                tensor = tf.identity(tensors[input_name], name=input_name)
                self.store_to_attr(input_name, tensors[input_name], device)

        self.store_to_attr('_inputs', _inputs)
        self.store_to_attr('inputs', tensors)
        return placeholders, tensors

    def _make_transform(self, input_name, tensor, config):
        if config is not None:
            transforms = {
                'ohe': self._make_ohe,
                'mip': self._make_mip,
                'downsample': self._make_downsample
            }

            transform_names = config.get('transform')
            if not isinstance(transform_names, list):
                transform_names = [transform_names]
            for transform_name in transform_names:
                if isinstance(transform_name, str):
                    kwargs = dict()
                    if transform_name.startswith('mip'):
                        parts = transform_name.split('@')
                        transform_name = 'mip'
                        kwargs['depth'] = int(parts[1])
                    elif transform_name.startswith('downsample'):
                        parts = transform_name.split('@')
                        transform_name = 'downsample'
                        kwargs['factor'] = int(parts[1])
                    tensor = transforms[transform_name](input_name, tensor, config, **kwargs)
                elif callable(transform_name):
                    tensor = transform_name(tensor)
                elif transform_name:
                    raise ValueError("Unknown transform {}".format(transform_name))
        return tensor

    def _make_ohe(self, input_name, tensor, config):
        if config.get('shape') is None and config.get('classes') is None:
            raise ValueError("shape and classes cannot be both None for input " +
                             "'{}' with one-hot-encoding transform".format(input_name))

        num_classes = self.num_classes(input_name)
        axis = -1 if self.data_format(input_name) == 'channels_last' else 1
        tensor = tf.one_hot(tensor, depth=num_classes, axis=axis)
        return tensor

    def _make_downsample(self, input_name, tensor, config, factor):
        """ Perform downsampling with the factor given. """
        _ = input_name, config
        size = self.shape(tensor, False)
        if None in size[1:]:
            size = self.shape(tensor, True)
        size = size / factor
        size = tf.cast(size, tf.int32)
        tensor = tf.expand_dims(tensor, -1)
        tensor = tf.image.resize_nearest_neighbor(tensor, size)
        tensor = tf.squeeze(tensor, [-1])
        return tensor

    def _make_mip(self, input_name, tensor, config, depth):
        # mip has to know shape
        if config.get('shape') is None:
            raise ValueError('mip transform requires shape specified in the inputs config')
        if depth is None:
            raise ValueError("mip should be specified as mip @ depth, e.g. 'mip @ 3'")
        tensor = mip(tensor, depth=depth, data_format=self.data_format(input_name))
        return tensor

    def to_classes(self, tensor, input_name, name=None):
        """ Convert tensor with labels to classes of ``input_name`` """
        if tensor.dtype in [tf.float16, tf.float32, tf.float64]:
            tensor = tf.argmax(tensor, axis=-1, name=name)
        if self.has_classes(input_name):
            self.store_to_attr('_to_classes', input_name, tensor)
        return tensor

    def _make_train_steps(self, config, init=True):
        # Wrap parameters from config root as `train_steps`
        if config.get('train_steps') is None:
            config.update({'train_steps': {'': {key: config.get(key) for key in
                                                ('optimizer', 'decay', 'loss', 'scope')}}})
            total = lambda _: tf.losses.get_total_loss(name='_TOTAL_LOSS')
        else:
            total = lambda loss: loss

        # First pass through the config: pass values from higher level, create (and store) all of the optimizers
        optimizers = {}
        for key, subconfig in config['train_steps'].items():
            subconfig.update({key: subconfig.get(key) or config.get(key)
                              for key in ('optimizer', 'decay', 'loss', 'scope')})
            if subconfig.get('optimizer') is not None:
                if optimizers.get(key) is None:
                    optimizers[key] = self._make_optimizer(subconfig)

        # Second pass through the config: create loss, get scope variables, minimize via chosen optimizer
        train_steps = {}
        for key, subconfig in config['train_steps'].items():
            # Create losses for every device, then combine them into one via summation
            device_grads, device_losses, ops = [], [], {}

            for device in self.devices:
                with tf.device(device):
                    with tf.variable_scope(self.device_to_scope[device]), tf.variable_scope(key):
                        loss_ = total(self._make_loss(subconfig, device))
                        loss_ = tf.identity(loss_, name='_DEVICE_LOSS')
                        device_losses.append(loss_)

                        optimizer = optimizers.get(subconfig.get('use')) or optimizers.get(key)

                        # It is important to control dependencies in order to work with layers like batch-normalization
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        with tf.control_dependencies(update_ops):
                            scope_collection = self._make_scope(subconfig, device)

                            # Simplest operation for training, created always
                            minimize_op = optimizer.minimize(loss_,
                                                             global_step=self.get_from_attr('global_step'),
                                                             var_list=scope_collection)
                            ops['minimize'] = minimize_op

                            # In order to use microbatches, we need to zero-out some storage, then populate it
                            # with computed gradients, and, finally, apply them to the weights at once
                            if self.microbatch:
                                if not self.multi_device:
                                    zero_op, update_op, apply_op = self._make_microbatch_ops(loss_, optimizer,
                                                                                             var_list=scope_collection)
                                    ops.update({'zero_grads': zero_op,
                                                'update_grads': update_op,
                                                'apply_grads': apply_op})

                            # To use multiple devices, we must compute gradients for every device,
                            # combine them on leading device, and apply updates to the weights on every device
                            if self.multi_device:
                                grad_and_vars = optimizer.compute_gradients(loss_,
                                                                            var_list=scope_collection)
                                device_grads.append(grad_and_vars)

            # Store average loss in the attribute, make operation to apply average gradient to the weights
            with tf.device(self.leading_device):
                loss_name = 'loss' if len(key) == 0 else 'loss_' + key
                loss = tf.reduce_mean(tf.stack(device_losses))
                loss = tf.identity(loss, name=loss_name)
                self.store_to_attr(loss_name, loss)

                if self.multi_device:
                    if not self.microbatch:
                        ops['multi_minimize'] = self._make_multi_op(device_grads, optimizer)

                    else:
                        zero_op, update_op, apply_op = self._make_microbatch_multi_ops(device_grads, optimizer)
                        ops.update({'multi_zero_grads': zero_op,
                                    'multi_update_grads': update_op,
                                    'multi_apply_grads': apply_op})

            # We need to explicitly initialize variable for every optimizer in order to not
            # interfere with capability to reuse optimizers for different train_steps
            if init:
                self.session.run(tf.variables_initializer(optimizer.variables()))

            # Store all the created operations
            train_steps.update({key: ops})

        self.store_to_attr('train_steps', train_steps)

    def _make_loss(self, config, device):
        loss, args = unpack_fn_from_config('loss', config)

        add_loss = False
        if loss is None:
            pass
        elif isinstance(loss, str):
            loss = LOSSES.get(re.sub('[-_ ]', '', loss).lower(), None)
        elif isinstance(loss, str) and hasattr(tf.losses, loss):
            loss = getattr(tf.losses, loss)
        elif callable(loss):
            pass
        else:
            raise ValueError("Unknown loss", loss)

        if loss is None:
            if len(tf.losses.get_losses()) == 0:
                raise ValueError("Loss is not defined in the model %s" % self)
        else:
            predictions_name = args.pop('predictions', 'predictions')
            targets_name = args.pop('targets', 'targets')
            predictions = self.get_from_attr(predictions_name, device)
            targets = self.get_from_attr(targets_name, device)

            add_loss = args.pop('add_loss', False)
            if add_loss:
                loss_collection = args.pop('loss_collection', None)
            tensor_loss = loss(targets, predictions, **args)
            if add_loss:
                if loss_collection:
                    tf.losses.add_loss(tensor_loss, loss_collection)
                else:
                    tf.losses.add_loss(tensor_loss)
        return tensor_loss

    def _make_optimizer(self, config):
        optimizer_name, optimizer_args = unpack_fn_from_config('optimizer', config)

        if optimizer_name is None or callable(optimizer_name):
            pass
        elif isinstance(optimizer_name, str) and hasattr(tf.train, optimizer_name):
            optimizer_name = getattr(tf.train, optimizer_name)
        elif isinstance(optimizer_name, str) and hasattr(tf.train, optimizer_name + 'Optimizer'):
            optimizer_name = getattr(tf.train, optimizer_name + 'Optimizer')
        else:
            raise ValueError("Unknown optimizer", optimizer_name)

        decay_name, decay_args = self._make_decay(config)
        if decay_name is not None:
            optimizer_args['learning_rate'] = decay_name(**decay_args,
                                                         global_step=self.get_from_attr('global_step'))

        if optimizer_name:
            optimizer = optimizer_name(**optimizer_args)
        else:
            optimizer = None

        return optimizer

    def _make_decay(self, config):
        decay_name, decay_args = unpack_fn_from_config('decay', config)

        if decay_name is None:
            pass
        elif callable(decay_name):
            pass
        elif isinstance(decay_name, str) and hasattr(tf.train, decay_name):
            decay_name = getattr(tf.train, decay_name)
        elif decay_name in DECAYS:
            decay_name = DECAYS.get(re.sub('[-_ ]', '', decay_name).lower(), None)
        else:
            raise ValueError("Unknown learning rate decay method", decay_name)

        return decay_name, decay_args

    def _make_scope(self, config, device):
        scopes = config.get('scope')
        scopes = [scopes] if isinstance(scopes, str) else scopes
        if not isinstance(scopes, (list, tuple)):
            raise ValueError("'Scope' key should be either string or sequence of strings.")

        total = []
        for scope in scopes:
            model_prefix = self.__class__.__name__ + '/'
            device_prefix = model_prefix + self.device_to_scope[device] + '/'

            if (len(scope) > 0) and (scope[0] in ['-', '_', '^']):
                scope_prefix = device_prefix + scope[1:]
            else:
                scope_prefix = device_prefix + scope

            scope_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 scope_prefix)
            if (len(scope) > 0) and (scope[0] in ['-', '_', '^']):
                scope_collection = [item for item in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, device_prefix)
                                    if item not in scope_collection]
            total.extend(scope_collection)
        return total

    def _make_microbatch_ops(self, loss, optimizer, var_list):
        with tf.variable_scope('microbatch'):
            # Container to store intermediate values of gradients
            count = tf.Variable(0.0, dtype=tf.float32, trainable=False, name='count')
            grad_accum = [tf.Variable(np.empty(var.shape, dtype=np.float32), trainable=False)
                          for var in var_list]

            # Zero-out operation
            with tf.variable_scope('zero_grads'):
                zero_grad_ops = [var.assign(tf.zeros(var.shape)) for var in grad_accum]
                zero_count_op = count.assign(tf.zeros(shape=(), dtype=tf.float32))
                zero_op = zero_grad_ops + [zero_count_op]
                zero_op = tf.group(zero_op, name='zero_grads_op')

            # Compute gradients and add it to the values in the storage
            with tf.variable_scope('update_grads'):
                grad_and_vars = optimizer.compute_gradients(loss, var_list)
                update_grad_ops = [grad_accum[i].assign_add(g) for i, (g, _) in enumerate(grad_and_vars)
                                   if g is not None]
                update_count_op = count.assign_add(tf.constant(1.0, dtype=tf.float32))
                update_op = update_grad_ops + [update_count_op]
                update_op = tf.group(update_op, name='update_grads_op')

            # Apply gradients from the storage to the actual weights
            with tf.variable_scope('apply_grads'):
                grad_and_vars = [(grad_accum[i] / count, v) for i, (_, v) in enumerate(grad_and_vars)]
                apply_op = optimizer.apply_gradients(grad_and_vars,
                                                     global_step=self.get_from_attr('global_step'))
                apply_op = tf.group(apply_op, name='apply_grads_op')
        return zero_op, update_op, apply_op

    def _make_multi_op(self, gradients, optimizer):
        operations = []
        # Each iteration of this loop works with 'copies' of the same variable on different devices
        for grad_and_vars in zip(*gradients):
            # Average gradients from different devices
            expanded = [tf.expand_dims(g, 0) for g, _ in grad_and_vars if g is not None]
            concatted = tf.concat(expanded, axis=0)
            averaged = tf.reduce_mean(concatted, axis=0)

            # Apply gradient on the leading device, then distribute to the others
            leading_device_variable = grad_and_vars[0][1]
            apply_op = optimizer.apply_gradients([(averaged, leading_device_variable)],
                                                 global_step=self.get_from_attr('global_step'))

            distribute_weights = [v.assign(leading_device_variable) for _, v in grad_and_vars[1:]]
            op = tf.group([apply_op] + distribute_weights, name='apply_weights_op')
            operations.append(op)

        # Combine update operations for every variable into single one
        op = tf.group(operations, name='multi_minimize_op')
        return op

    def _make_microbatch_multi_ops(self, gradients, optimizer):
        global_step = self.get_from_attr('global_step')
        zero_ops = []
        update_ops = []
        apply_ops = []

        with tf.variable_scope('microbatch_multi'):
            count = tf.Variable(0.0, dtype=tf.float32, trainable=False, name='count')
            zero_count_op = count.assign(tf.zeros(shape=(), dtype=tf.float32))
            zero_ops.append(zero_count_op)

            update_count_op = count.assign_add(tf.constant(1.0, dtype=tf.float32))
            update_ops.append(update_count_op)

            for grad_and_vars in zip(*gradients):
                # Leading device variable
                var = grad_and_vars[0][1]
                grad_accum = tf.Variable(np.empty(var.shape, dtype=np.float32), trainable=False)
                zero_grad_op = grad_accum.assign(tf.zeros(var.shape))
                zero_ops.append(zero_grad_op)

                # Average gradients from different devices
                expanded = [tf.expand_dims(g, 0) for g, _ in grad_and_vars if g is not None]
                concatted = tf.concat(expanded, axis=0)
                averaged = tf.reduce_mean(concatted, axis=0)
                update_grad_op = grad_accum.assign_add(averaged)
                update_ops.append(update_grad_op)

                apply_grad_op_ = optimizer.apply_gradients([(grad_accum / count, var)],
                                                           global_step=global_step)
                distribute_weights = [v.assign(var) for _, v in grad_and_vars[1:]]
                apply_grad_op = tf.group([apply_grad_op_] + distribute_weights, name='apply_weights_op')
                apply_ops.append(apply_grad_op)

            zero_op = tf.group(zero_ops, name='multi_zero_grads_op')
            update_op = tf.group(update_ops, name='multi_update_grads_op')
            apply_op = tf.group(apply_ops, name='multi_apply_grads_op')

        return zero_op, update_op, apply_op

    def get_number_of_trainable_vars(self):
        """ Return the number of trainable variable in the model graph """
        arr = np.asarray([np.prod(v.get_shape().as_list()) for v in self.graph.get_collection('trainable_variables')])
        return np.sum(arr)

    def get_tensor_config(self, tensor, **kwargs):
        """ Return tensor configuration

        Parameters
        ----------
        tensor : str or tf.Tensor

        Returns
        -------
        dict
            tensor config (see :meth:`.TFModel._make_inputs`)

        Raises
        ------
        ValueError shape in tensor configuration isn't int, tuple or list
        """
        inputs = self.get_from_attr('_inputs')

        if tensor in inputs:
            config = inputs[tensor]['config']
            shape = config.get('shape')
            if isinstance(shape, int):
                shape = (shape,)
            if shape:
                kwargs['shape'] = shape
        elif isinstance(tensor, str):
            try:
                tensor = self.get_from_attr(tensor)
            except KeyError:
                config = {}
            else:
                shape = tensor.get_shape().as_list()[1:]
                data_format = self._full_config.get('common/data_format') or 'channels_last'
                config = dict(dtype=tensor.dtype, shape=shape,
                              name=tensor.name, data_format=data_format)
        else:
            config = {}

        config = {**config, **kwargs}
        return config

    def _map_name(self, name, device=None):
        if isinstance(name, str):
            return self.get_from_attr(name, device)
        return name

    def _fill_feed_dict(self, feed_dict=None, device=None, is_training=True):
        feed_dict = feed_dict or {}
        _feed_dict = {}
        for placeholder, value in feed_dict.items():
            if self.has_classes(placeholder):
                classes = self.classes(placeholder)
                get_indices = np.vectorize(lambda c, arr=classes: np.where(c == arr)[0])
                value = get_indices(value)
            placeholder = self._map_name(placeholder, device)
            value = self._map_name(value, device)
            _feed_dict.update({placeholder: value})
        if self.get_from_attr('is_training') not in _feed_dict:
            _feed_dict.update({self.get_from_attr('is_training'): is_training})
        return _feed_dict

    def _fill_fetches(self, fetches=None, default=None):
        fetches = fetches or default
        if isinstance(fetches, str):
            _fetches = self._map_name(fetches)
        elif isinstance(fetches, (tuple, list)):
            _fetches = []
            for fetch in fetches:
                _fetches.append(self._map_name(fetch))
        elif isinstance(fetches, dict):
            _fetches = dict()
            for key, fetch in fetches.items():
                _fetches.update({key: self._map_name(fetch)})
        else:
            _fetches = fetches
        return _fetches

    def _recast_output(self, out, ix=None, fetches=None):
        if isinstance(out, np.ndarray):
            fetch = fetches[ix] if ix is not None else fetches
            if isinstance(fetch, str):
                fetch = self.graph.get_tensor_by_name(fetch)
            _to_classes = self.get_from_attr('_to_classes', default={})
            if fetch in _to_classes:
                return self.classes(_to_classes[fetch])[out]
        return out

    def _fill_output(self, output, fetches):
        if isinstance(output, (tuple, list)):
            _output = []
            for i, o in enumerate(output):
                _output.append(self._recast_output(o, i, fetches))
            output = type(output)(_output)
        elif isinstance(output, dict):
            _output = type(output)()
            for k, v in output.items():
                _output.update({k: self._recast_output(v, k, fetches)})
        else:
            output = self._recast_output(output, fetches=fetches)

        return output

    def train(self, fetches=None, feed_dict=None, use_lock=False, train_mode='', microbatch=None, **kwargs):
        """ Train the model with the data provided

        Parameters
        ----------
        fetches : tuple, list
            a sequence of `tf.Operation` and/or `tf.Tensor` to calculate
        feed_dict : dict
            input data, where key is a placeholder name and value is a numpy value
        use_lock : bool
            if True, the whole train step is locked, thus allowing for multithreading.
        train_mode : str or sequence of str
            name(s) of train step to optimize. Regular expressions are allowed.
        microbatch : int
            size of chunks to split every batch in. Note that if this option was not specified
            in the model configuration, the first invocation of this method would create additional operations.

        Returns
        -------
        Calculated values of tensors in `fetches` in the same structure

        See also
        --------
        `Tensorflow Session run <https://www.tensorflow.org/api_docs/python/tf/Session#run>`_

        Notes
        -----
        ``feed_dict`` is not required as all placeholder names and their data can be passed directly as named arguments

        Examples
        --------

        ::

            model.train(fetches='loss', feed_dict={'images': B('images'), 'labels': B('labels')})

        The same as::

            model.train(fetches='loss', images=B('images'), labels=B('labels'))
        """
        with self.graph.as_default():
            train_steps = self.get_from_attr('train_steps')
            # Use microbatch size from either args or config, and if
            # necessary ops for microbatch-training are absent, create them
            if microbatch is not False:
                microbatch = microbatch or self.microbatch
            self.microbatch = microbatch

            if (microbatch) and (len(list(train_steps.values())[0]) == 1):
                self._make_train_steps(self._full_config, init=False)

            if microbatch is True: # if config option is set to True, but train option left unspectified,
                microbatch = False # it is faster to pretend that there is no microbatching

            # `feed_dict` processing: updating it with all kwargs,
            # optionally splitting it for microbatch train, resulting in list of feed_dicts,
            # updating every of them with `_fill_feed_dict` so tensorflow can work with it
            feed_dict = feed_dict or {}
            feed_dict = {**feed_dict, **kwargs}

            # `fetches` and `train_mode` processing
            if fetches is None:
                _fetches = tuple()
            else:
                names = [fetches] if isinstance(fetches, str) else fetches
                _fetches = self._fill_fetches(names, default=None)

            if not isinstance(train_mode, (tuple, list)):
                train_mode = [train_mode]

            # Acquire lock so only one `train` is active at a time
            if use_lock:
                self._train_lock.acquire()

            if train_steps:
                for mode in train_mode:
                    if mode in train_steps.keys():
                        train_fetches = [train_steps[mode]]
                    else:
                        train_fetches = [train_step for name, train_step in train_steps.items()
                                         if re.search(mode, name) is not None]

                    if not microbatch:
                        if not self.multi_device:
                            output = self._vanilla_train(train_fetches, _fetches, feed_dict)
                        else:
                            output = self._multi_train(train_fetches, _fetches, feed_dict)
                    else:
                        feed_dicts = self._split_feed_dict(feed_dict, size=microbatch)

                        if not self.multi_device:
                            outputs = self._microbatch_train(train_fetches, _fetches, feed_dicts)
                        else:
                            outputs = self._microbatch_multi_train(train_fetches, _fetches, feed_dicts)

                        outputs = [[item[i] for item in outputs] for i, _ in enumerate(names)]
                        output = [np.mean(outputs[i]) if 'loss' in name else outputs[i][-1]
                                  for i, name in enumerate(names)]

                    output = output[0] if isinstance(fetches, str) else output
            else:
                output = None

            if use_lock:
                self._train_lock.release()
            return self._fill_output(output, _fetches)

    def _split_feed_dict(self, feed_dict, num_parts=None, size=None):
        splitted = {}
        for key, value in feed_dict.items():
            if hasattr(value, '__len__'):
                if num_parts is None:
                    num_parts = len(value) // size
                if len(value) % num_parts != 0:
                    raise ValueError('Batch size must be divisible by {}, but is {}'.format(num_parts, len(value)))
                splitted[key] = np.array_split(value, num_parts)

        splitted_ = [{key: value[i] for key, value in splitted.items()}
                     for i in range(num_parts)]
        return splitted_

    def _vanilla_train(self, train_fetches, fetches, feed_dict):
        # Get list of train operations to run
        all_fetches = [ops['minimize'] for ops in train_fetches]
        if fetches is not None:
            all_fetches += [fetches]

        # Fill feed_dict with placeholders
        _fd = self._fill_feed_dict(feed_dict, is_training=True)
        *_, output = self.session.run(all_fetches, feed_dict=_fd)

        return output

    def _multi_train(self, train_fetches, _fetches, feed_dict):
        # Get list of train operations to run
        all_fetches = [ops['multi_minimize'] for ops in train_fetches]
        if _fetches is not None:
            all_fetches += [_fetches]

        # Split batch into even parts for every device, then run complex operation
        # that computes gradients on every device, combines them on the leading one,
        # and finally sends updates back to devices
        _feed_dicts = self._split_feed_dict(feed_dict, num_parts=self.multi_device)

        _fd = {}
        for part, device in zip(_feed_dicts, self.devices):
            _fd = {**_fd, **self._fill_feed_dict(part, device)}
        *_, output = self.session.run(all_fetches, feed_dict=_fd)

        return output

    def _microbatch_train(self, train_fetches, _fetches, feed_dicts):
        _feed_dicts = [self._fill_feed_dict(part, is_training=True) for part in feed_dicts]

        outputs = []
        for ops in train_fetches:
            # Get train operations to run
            zero_op, update_op, apply_op = ops['zero_grads'], \
                                           ops['update_grads'], \
                                           ops['apply_grads']
            all_fetches = [update_op]
            if _fetches is not None:
                all_fetches += [_fetches]

            # For every train step, zero out gradient accumulators,then update them with gradients,
            # computed on each of `feed_dicts`, and finally apply accumulated values to weights
            self.session.run(zero_op, feed_dict=_feed_dicts[0])
            for _fd in _feed_dicts:
                _, _output = self.session.run(all_fetches, feed_dict=_fd)
                outputs += [_output]
            self.session.run(apply_op, feed_dict=_feed_dicts[-1])
        return outputs

    def _microbatch_multi_train(self, train_fetches, _fetches, feed_dicts):
        outputs = []
        for ops in train_fetches:
            # Get train operations to run
            zero_op, update_op, apply_op = ops['multi_zero_grads'], \
                                           ops['multi_update_grads'], \
                                           ops['multi_apply_grads']
            all_fetches = [update_op]
            if _fetches is not None:
                all_fetches += [_fetches]

            # For every microbatch run complex operation that computes gradients on every device,
            # combines them on the leading one, and stores into accumulator. When the last
            # microbatch is processed, accumulated value is applied to the weights on leading device,
            # and finally distributed to other devices
            for i, feed_dict in enumerate(feed_dicts):
                _feed_dicts = self._split_feed_dict(feed_dict, num_parts=self.multi_device)
                _fd = {}
                for part, device in zip(_feed_dicts, self.devices):
                    _fd = {**_fd, **self._fill_feed_dict(part, device)}

                if i == 0:
                    self.session.run(zero_op, feed_dict=_fd)

                _, _output = self.session.run(all_fetches, feed_dict=_fd)
                outputs += [_output]
            self.session.run(apply_op, feed_dict=_fd)
        return outputs

    def predict(self, fetches=None, feed_dict=None, **kwargs):
        """ Get predictions on the data provided

        Parameters
        ----------
        fetches : tuple, list
            a sequence of `tf.Operation` and/or `tf.Tensor` to calculate
        feed_dict : dict
            input data, where key is a placeholder name and value is a numpy value

        Returns
        -------
        Calculated values of tensors in `fetches` in the same structure

        See also
        --------
        `Tensorflow Session run <https://www.tensorflow.org/api_docs/python/tf/Session#run>`_

        Notes
        -----
        ``feed_dict`` is not required as all placeholder names and their data can be passed directly.

        Examples
        --------

        ::

            model.predict(fetches='loss', feed_dict={'images': B('images'), 'labels': B('labels')})

        The same as::

            model.predict(fetches='loss', images=B('images'), labels=B('labels'))

        """
        with self.graph.as_default():
            feed_dict = {} if feed_dict is None else feed_dict
            feed_dict = {**feed_dict, **kwargs}
            _feed_dict = self._fill_feed_dict(feed_dict, is_training=False)
            _fetches = self._fill_fetches(fetches, default='predictions')
            output = self.session.run(_fetches, _feed_dict)
        return self._fill_output(output, _fetches)

    def save(self, path, *args, **kwargs):
        """ Save tensorflow model and most of important attributes.

        Note
        ----
        All of tuples (for example, shapes) are converted to lists due to usage of JSON format.

        Parameters
        ----------
        path : str
            a path to a directory where all model files will be stored

        Examples
        --------
        >>> tf_model = ResNet34()

        Now save the model

        >>> tf_model.save('/path/to/models/resnet34')

        The model will be saved to /path/to/models/resnet34
        """
        with self.graph.as_default():
            if not os.path.exists(path):
                os.makedirs(path)
            if self._saver is None:
                self._saver = tf.train.Saver()
            self._saver.save(self.session, os.path.join(path, 'model'), *args,
                             global_step=self.get_from_attr('global_step'), **kwargs)

        preserved = dict()
        for attribute_name in self.preserve:
            attribute = getattr(self, attribute_name)
            preserved[attribute_name] = self._to_names(attribute)
        with open(os.path.join(path, 'attributes.dill'), 'wb') as f:
            dill.dump(preserved, f)

    def _to_names(self, graph_item):
        # Base cases
        if isinstance(graph_item, tf.Tensor):
            return ('Tensor', graph_item.name)
        if isinstance(graph_item, tf.Operation):
            return ('Operation', graph_item.name)
        if isinstance(graph_item, tf.Variable):
            return ('Variable', graph_item.op.name)
        if isinstance(graph_item, (bool, str, int, float)) or graph_item is None:
            return graph_item

        # Handle different containers
        if isinstance(graph_item, (list, tuple, np.ndarray)):
            return type(graph_item)([self._to_names(item) for item in graph_item])
        if isinstance(graph_item, (dict, Config)):
            return type(graph_item)({key: self._to_names(graph_item[key]) for key in graph_item.keys()})
        raise ValueError('Unrecognized type of value.')

    def load(self, path, graph=None, checkpoint=None, *args, **kwargs):
        """ Load a TensorFlow model and most important attributes from files

        Note
        ----
        All of tuples (for example, shapes) are loaded as lists due to usage of JSON format.

        Parameters
        ----------
        path : str
            a directory where a model is stored
        graph : str
            a filename for a metagraph file
        checkpoint : str
            a checkpoint file name or None to load the latest checkpoint

        Examples
        --------
        >>> resnet = ResNet34(load=dict(path='/path/to/models/resnet34'))

        >>> tf_model.load(path='/path/to/models/resnet34')
        """
        _ = args, kwargs
        self.graph = tf.Graph()

        with self.graph.as_default():
            if graph is None:
                graph_files = glob.glob(os.path.join(path, '*.meta'))
                graph_files = [os.path.splitext(os.path.basename(graph))[0] for graph in graph_files]
                all_steps = []
                for _graph in graph_files:
                    try:
                        step = int(_graph.split('-')[-1])
                    except ValueError:
                        pass
                    else:
                        all_steps.append(step)
                graph = '-'.join(['model', str(max(all_steps))]) + '.meta'

            graph_path = os.path.join(path, graph)
            saver = tf.train.import_meta_graph(graph_path)

            if checkpoint is None:
                checkpoint_path = tf.train.latest_checkpoint(path)
            else:
                checkpoint_path = os.path.join(path, checkpoint)

            self.create_session()
            saver.restore(self.session, checkpoint_path)

        with open(os.path.join(path, 'attributes.dill'), 'rb') as dill_file:
            restored = dill.load(dill_file)

        for attribute_name, value in restored.items():
            setattr(self, attribute_name, self._to_graph_items(value))
        self.preserve = list(restored.keys())

    def _to_graph_items(self, name):
        # Base cases
        if isinstance(name, (bool, str, int, float)) or name is None:
            return name

        # Handle different containers
        if isinstance(name, (list, tuple, np.ndarray)):
            if len(name) == 2:
                type_, name_ = name
                if type_ == 'Variable':
                    with self.graph.as_default():
                        return tf.global_variables(name_)[0]
                if type_ == 'Tensor':
                    return self.graph.get_tensor_by_name(name_)
                if type_ == 'Operation':
                    return self.graph.get_operation_by_name(name_)
            return type(name)([self._to_graph_items(item) for item in name])

        if isinstance(name, (dict, Config)):
            return type(name)({key: self._to_graph_items(name[key]) for key in name.keys()})
        raise ValueError('Unrecognized type of value.')

    @classmethod
    def crop(cls, inputs, resize_to, data_format='channels_last'):
        """ Crop input tensor to a shape of a given image.
        If resize_to does not have a fully defined shape (resize_to.get_shape() has at least one None),
        the returned tf.Tensor will be of unknown shape except the number of channels.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        resize_to : tf.Tensor
            a tensor which shape the inputs should be resized to
        data_format : str {'channels_last', 'channels_first'}
            data format
        """

        static_shape = cls.spatial_shape(resize_to, data_format, False)
        dynamic_shape = cls.spatial_shape(resize_to, data_format, True)

        if None in cls.shape(inputs) + static_shape:
            return cls._dynamic_crop(inputs, static_shape, dynamic_shape, data_format)
        return cls._static_crop(inputs, static_shape, data_format)

    @classmethod
    def _static_crop(cls, inputs, shape, data_format='channels_last'):
        input_shape = np.array(cls.spatial_shape(inputs, data_format))

        if np.abs(input_shape - shape).sum() > 0:
            begin = [0] * inputs.shape.ndims
            if data_format == "channels_last":
                size = [-1] + shape + [-1]
            else:
                size = [-1, -1] + shape
            x = tf.slice(inputs, begin=begin, size=size)
        else:
            x = inputs
        return x

    @classmethod
    def _dynamic_crop(cls, inputs, static_shape, dynamic_shape, data_format='channels_last'):
        input_shape = cls.spatial_shape(inputs, data_format, True)
        n_channels = cls.num_channels(inputs, data_format)
        if data_format == 'channels_last':
            slice_size = [(-1,), dynamic_shape, (n_channels,)]
            output_shape = [None] * (len(static_shape) + 1) + [n_channels]
        else:
            slice_size = [(-1, n_channels), dynamic_shape]
            output_shape = [None, n_channels] + [None] * len(static_shape)

        begin = [0] * len(inputs.get_shape().as_list())
        size = tf.concat(slice_size, axis=0)
        cond = tf.reduce_sum(tf.abs(input_shape - dynamic_shape)) > 0
        x = tf.cond(cond, lambda: tf.slice(inputs, begin=begin, size=size), lambda: inputs)
        x.set_shape(output_shape)
        return x

    @classmethod
    def initial_block(cls, inputs, name='initial_block', **kwargs):
        """ Transform inputs with a convolution block

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name

        Notes
        -----
        For other parameters see :func:`.conv_block`.

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('initial_block', **kwargs)
        if kwargs.get('layout'):
            return conv_block(inputs, name=name, **kwargs)
        return inputs

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Base layers which produce a network embedding

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name

        Notes
        -----
        For other parameters see :func:`.conv_block`.

        Returns
        -------
        tf.Tensor

        Examples
        --------
        ::

            MyModel.body(inputs, layout='ca ca ca', filters=[128, 256, 512], kernel_size=3)
        """
        kwargs = cls.fill_params('body', **kwargs)
        if kwargs.get('layout'):
            return conv_block(inputs, name=name, **kwargs)
        return inputs

    @classmethod
    def head(cls, inputs, name='head', **kwargs):
        """ The last network layers which produce predictions

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name

        Notes
        -----
        For other parameters see :func:`.conv_block`.

        Returns
        -------
        tf.Tensor

        Examples
        --------
        A fully convolutional head with 3x3 and 1x1 convolutions and global max pooling:

            MyModel.head(network_embedding, layout='cacaP', filters=[128, num_classes], kernel_size=[3, 1])

        A fully connected head with dropouts, a dense layer with 1000 units and final dense layer with class logits::

            MyModel.head(network_embedding, layout='dfadf', units=[1000, num_classes], dropout_rate=.15)
        """
        kwargs = cls.fill_params('head', **kwargs)
        if kwargs.get('layout'):
            return conv_block(inputs, name=name, **kwargs)
        return inputs

    def output(self, inputs, predictions=None, ops=None, prefix=None, **kwargs):
        """ Add output operations to the model graph, like predicted probabilities or labels, etc.

        Parameters
        ----------
        inputs : tf.Tensor or a sequence of tf.Tensors
            input tensors

        predictions : str or callable
            an operation applied to inputs to get `predictions` tensor which is used in a loss function:

            - 'sigmoid' - ``sigmoid(inputs)``
            - 'proba' - ``softmax(inputs)``
            - 'labels' - ``argmax(inputs)``
            - 'softplus' - ``softplus(inputs)``
            - callable - a user-defined operation

        ops : a sequence of operations or an ordered dict
            auxiliary operations

            If dict:

            - key - a prefix for each input
            - value - a sequence of aux operations

        Raises
        ------
        ValueError if the number of inputs does not equal to the number of prefixes
        TypeError if inputs is not a Tensor or a sequence of Tensors

        Examples
        --------

        ::

            config = {
                'output': ['proba', 'labels']
            }

        However, if one of the placeholders also has a name 'labels', then it will be lost as the model
        will rewrite the name 'labels' with an output.

        That is where a dict might be convenient::

            config = {
                'output': {'predicted': ['proba', 'labels']}
            }

        Now the output will be stored under names 'predicted_proba' and 'predicted_labels'.

        For multi-output models ensure that an ordered dict is used (e.g. :class:`~collections.OrderedDict`).
        """
        if ops is None:
            ops = []
        elif not isinstance(ops, (dict, tuple, list)):
            ops = [ops]
        if not isinstance(ops, dict):
            ops = {'': ops}

        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]

        for i, tensor in enumerate(inputs):
            if not isinstance(tensor, tf.Tensor):
                raise TypeError("Network output is expected to be a Tensor, but given {}".format(type(inputs)))

            prefix = [*ops.keys()][i]
            if prefix:
                ctx = tf.variable_scope(prefix)
                ctx.__enter__()
            else:
                ctx = None
            attr_prefix = prefix + '_' if prefix else ''

            self._add_output_op(tensor, predictions, 'predictions', '', **kwargs)
            for oper in ops[prefix]:
                self._add_output_op(tensor, oper, oper, attr_prefix, **kwargs)

            if ctx:
                ctx.__exit__(None, None, None)

    def _add_output_op(self, inputs, oper, name, attr_prefix, **kwargs):
        device = self._get_current_device()
        if oper is None:
            self._add_output_identity(inputs, name, attr_prefix, device, **kwargs)
        elif oper == 'softplus':
            self._add_output_softplus(inputs, name, attr_prefix, device, **kwargs)
        elif oper == 'sigmoid':
            self._add_output_sigmoid(inputs, name, attr_prefix, device, **kwargs)
        elif oper == 'proba':
            self._add_output_proba(inputs, name, attr_prefix, device, **kwargs)
        elif oper == 'labels':
            self._add_output_labels(inputs, name, attr_prefix, device, **kwargs)
        elif callable(oper):
            self._add_output_callable(inputs, oper, None, attr_prefix, device, **kwargs)

    def _add_output_identity(self, inputs, name, attr_prefix, device, **kwargs):
        _ = kwargs
        x = tf.identity(inputs, name=name)
        self.store_to_attr(attr_prefix + name, x, device)
        return x

    def _add_output_softplus(self, inputs, name, attr_prefix, device, **kwargs):
        _ = kwargs
        proba = tf.nn.softplus(inputs, name=name)
        self.store_to_attr(attr_prefix + name, proba, device)

    def _add_output_sigmoid(self, inputs, name, attr_prefix, device, **kwargs):
        _ = kwargs
        proba = tf.sigmoid(inputs, name=name)
        self.store_to_attr(attr_prefix + name, proba, device)

    def _add_output_proba(self, inputs, name, attr_prefix, device, **kwargs):
        axis = self.channels_axis(kwargs['data_format'])
        proba = tf.nn.softmax(inputs, name=name, axis=axis)
        self.store_to_attr(attr_prefix + name, proba, device)

    def _add_output_labels(self, inputs, name, attr_prefix, device, **kwargs):
        class_axis = self.channels_axis(kwargs.get('data_format'))
        predicted_classes = tf.argmax(inputs, axis=class_axis, name=name)
        self.store_to_attr(attr_prefix + name, predicted_classes, device)

    def _add_output_callable(self, inputs, oper, name, attr_prefix, device, **kwargs):
        _ = kwargs
        x = oper(inputs)
        name = name or oper.__name__
        self.store_to_attr(attr_prefix + name, x, device)
        return x

    @classmethod
    def default_config(cls):
        """ Define model defaults

        You need to override this method if you expect your model or its blocks to serve as a base for other models
        (e.g. VGG for FCN, ResNet for LinkNet, etc).

        Put here all constants (like the number of filters, kernel sizes, block layouts, strides, etc)
        specific to the model, but independent of anything else (like image shapes, number of classes, etc).

        These defaults can be changed in :meth:`~.TFModel.build_config` or when calling :meth:`.Pipeline.init_model`.

        Usually, it looks like::

            @classmethod
            def default_config(cls):
                config = TFModel.default_config()
                config['initial_block'] = dict(layout='cnap', filters=16, kernel_size=7, strides=2,
                                               pool_size=3, pool_strides=2)
                config['body/filters'] = 32
                config['head'] = dict(layout='cnadV', dropout_rate=.2)
                return config
        """
        config = Config()
        config['inputs'] = {}
        config['initial_block'] = {}
        config['body'] = {}
        config['head'] = {}
        config['predictions'] = None
        config['output'] = None
        config['optimizer'] = ('Adam', dict())
        config['decay'] = (None, dict())
        config['scope'] = ''
        config['common'] = {'batch_norm': {'momentum': .1}}

        return config

    @classmethod
    def fill_params(cls, _name, **kwargs):
        """ Fill block params from default config and kwargs """
        config = cls.default_config()
        _config = config.get(_name)
        config = {**config['common'], **_config, **kwargs}
        return config

    def build_config(self, names=None):
        """ Define a model architecture configuration

        It takes just 2 steps:

        #. Define names for all placeholders and make input tensors by calling ``super().build_config(names)``.

           If the model config does not contain any name from ``names``, :exc:`KeyError` is raised.

           See :meth:`.TFModel._make_inputs` for details.

        #. Define parameters for :meth:`.TFModel.initial_block`, :meth:`.TFModel.body`, :meth:`.TFModel.head`
           which depend on inputs.

        #. Don't forget to return ``config``.

        Typically it looks like this::

            def build_config(self, names=None):
                names = names or ['images', 'labels']
                config = super().build_config(names)
                config['head']['num_classes'] = self.num_classes('targets')
                return config
        """
        config = self.default_config()

        config = config + self.config

        if config.get('inputs'):
            with tf.variable_scope('inputs'):
                self._make_inputs(names, config)
            inputs = config.get('initial_block/inputs')

            if isinstance(inputs, str):
                if not config.get('common/data_format'):
                    config['common/data_format'] = self.data_format(inputs)
                config['initial_block/inputs'] = self.get_from_attr('inputs')[inputs]

            elif isinstance(inputs, list):
                config['initial_block/inputs'] = [self.get_from_attr('inputs')[name]
                                                  for name in inputs]
            else:
                raise ValueError('initial_block/inputs should be specified with a name or a list of names.')

        return config

    def _add_block(self, name, config, inputs):
        defaults = {'is_training': self.get_from_attr('is_training'),
                    'global_step': self.get_from_attr('global_step'),
                    **config['common']}
        if callable(config[name]):
            block = config[name](inputs, **defaults)
        elif isinstance(config[name], dict):
            block = getattr(self, name)(inputs=inputs, **{**defaults, **config[name]})
        else:
            raise TypeError('block can be configured as a function or a dict with parameters')
        return block

    def _build(self, config=None):
        inputs = config.pop('initial_block/inputs')
        x = self._add_block('initial_block', config, inputs=inputs)
        x = self._add_block('body', config, inputs=x)
        output = self._add_block('head', config, inputs=x)
        self.output(output, predictions=config['predictions'], ops=config['output'], **config['common'])

    def data_format(self, tensor, **kwargs):
        """ Return the tensor data format (channels_last or channels_first)

        Parameters
        ----------
        tensor : str or tf.Tensor

        Returns
        -------
        data_format : str
        """
        config = self.get_tensor_config(tensor, **kwargs)
        return config.get('data_format')

    def has_classes(self, tensor):
        """ Check if a tensor has classes defined in the config """
        config = self.get_tensor_config(tensor)
        has = config.get('classes') is not None
        return has

    def classes(self, tensor):
        """ Return the classes """
        config = self.get_tensor_config(tensor)
        classes = config.get('classes')
        if isinstance(classes, int):
            return np.arange(classes)
        return np.asarray(classes)

    def num_classes(self, tensor):
        """ Return the  number of classes """
        if self.has_classes(tensor):
            classes = self.classes(tensor)
            return classes if isinstance(classes, int) else len(classes)
        return self.get_num_channels(tensor)

    def get_num_channels(self, tensor, **kwargs):
        """ Return the number of channels in the tensor

        Parameters
        ----------
        tensor : str or tf.Tensor

        Returns
        -------
        number of channels : int
        """
        config = self.get_tensor_config(tensor, **kwargs)
        shape = (None,) + config.get('shape')
        channels_axis = self.channels_axis(tensor, **kwargs)
        return shape[channels_axis] if shape else None

    @classmethod
    def num_channels(cls, tensor, data_format='channels_last'):
        """ Return number of channels in the input tensor

        Parameters
        ----------
        tensor : tf.Tensor

        Returns
        -------
        shape : tuple of ints
        """
        shape = tensor.get_shape().as_list()
        axis = cls.channels_axis(data_format)
        return shape[axis]

    def get_shape(self, tensor, **kwargs):
        """ Return the tensor shape without batch dimension

        Parameters
        ----------
        tensor : str or tf.Tensor

        Returns
        -------
        shape : tuple
        """
        config = self.get_tensor_config(tensor, **kwargs)
        return config.get('shape')

    @classmethod
    def shape(cls, tensor, dynamic=False):
        """ Return shape of the input tensor without batch size

        Parameters
        ----------
        tensor : tf.Tensor

        dynamic : bool
            if True, returns tensor which represents shape. If False, returns list of ints and/or Nones

        Returns
        -------
        shape : tf.Tensor or list
        """
        if dynamic:
            shape = tf.shape(tensor)
        else:
            shape = tensor.get_shape().as_list()
        return shape[1:]

    def get_spatial_dim(self, tensor, **kwargs):
        """ Return the tensor spatial dimensionality (without batch and channels dimensions)

        Parameters
        ----------
        tensor : str or tf.Tensor

        Returns
        -------
        number of spatial dimensions : int
        """
        config = self.get_tensor_config(tensor, **kwargs)
        return len(config.get('shape')) - 1

    @classmethod
    def spatial_dim(cls, tensor):
        """ Return spatial dim of the input tensor (without channels and batch dimension)

        Parameters
        ----------
        tensor : tf.Tensor

        Returns
        -------
        dim : int
        """
        return len(tensor.get_shape().as_list()) - 2

    def get_spatial_shape(self, tensor, **kwargs):
        """ Return the tensor spatial shape (without batch and channels dimensions)

        Parameters
        ----------
        tensor : str or tf.Tensor

        Returns
        -------
        spatial shape : tuple
        """
        config = self.get_tensor_config(tensor, **kwargs)
        data_format = config.get('data_format')
        shape = config.get('shape')[:-1] if data_format == 'channels_last' else config.get('shape')[1:]
        return shape

    @classmethod
    def spatial_shape(cls, tensor, data_format='channels_last', dynamic=False):
        """ Return spatial shape of the input tensor

        Parameters
        ----------
        tensor : tf.Tensor

        dynamic : bool
            if True, returns tensor which represents shape. If False, returns list of ints and/or Nones

        Returns
        -------
        shape : tf.Tensor or list
        """
        if dynamic:
            shape = tf.shape(tensor)
        else:
            shape = tensor.get_shape().as_list()
        axis = slice(1, -1) if data_format == "channels_last" else slice(2, None)
        return shape[axis]

    def get_batch_size(self, tensor):
        """ Return batch size (the length of the first dimension) of the input tensor

        Parameters
        ----------
        tensor : str or tf.Tensor

        Returns
        -------
        batch size : int or None
        """
        if isinstance(tensor, tf.Tensor):
            pass
        elif isinstance(tensor, str):
            tensor = self.get_from_attr(tensor)
        else:
            raise TypeError("Tensor can be tf.Tensor or string, but given %s" % type(tensor))

        return tensor.get_shape().as_list()[0]

    @classmethod
    def batch_size(cls, tensor):
        """ Return batch size (the length of the first dimension) of the input tensor

        Parameters
        ----------
        tensor : tf.Tensor

        Returns
        -------
        batch size : int or None
        """
        return tensor.get_shape().as_list()[0]


    @classmethod
    def channels_axis(cls, data_format='channels_last'):
        """ Return the channels axis for the tensor

        Parameters
        ----------
        data_format : str {'channels_last', 'channels_first', 'N***'} or None

        Returns
        -------
        int
        """
        return 1 if data_format == "channels_first" or data_format.startswith("NC") else -1

    @classmethod
    def se_block(cls, inputs, ratio, name='se', **kwargs):
        """ Squeeze and excitation block

        Hu J. et al. "`Squeeze-and-Excitation Networks <https://arxiv.org/abs/1709.01507>`_"

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        ratio : int
            squeeze ratio for the number of filters

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            data_format = kwargs.get('data_format')
            in_filters = cls.num_channels(inputs, data_format)
            x = conv_block(inputs,
                           **{**kwargs, 'layout': 'Vfafa', 'units': [in_filters//ratio, in_filters],
                              'name': 'se', 'activation': [tf.nn.relu, tf.nn.sigmoid]})

            shape = [-1] + [1] * (cls.spatial_dim(inputs) + 1)
            axis = cls.channels_axis(data_format)
            shape[axis] = in_filters
            scale = tf.reshape(x, shape)
            x = inputs * scale
        return x

    @classmethod
    def upsample(cls, inputs, factor=None, resize_to=None, layout='b', name='upsample', **kwargs):
        """ Upsample input tensor

        Parameters
        ----------
        inputs : tf.Tensor or tuple of two tf.Tensor
            a tensor to resize and a tensor which size to resize to
        factor : int
            an upsamping scale
        resize_to : tf.Tensor
            a tensor which shape the output should be resized to
        layout : str
            a resizing technique, a sequence of:

            - R - use residual connection with bilinear additive upsampling (must be the first symbol)
            - b - bilinear resize
            - B - bilinear additive upsampling
            - N - nearest neighbor resize
            - t - transposed convolution
            - X - subpixel convolution

        Returns
        -------
        tf.Tensor
        """
        if np.all(factor == 1):
            return inputs

        if kwargs.get('filters') is None:
            kwargs['filters'] = cls.num_channels(inputs, kwargs['data_format'])

        x = upsample(inputs, factor=factor, layout=layout, name=name, **kwargs)
        if resize_to is not None:
            x = cls.crop(x, resize_to, kwargs['data_format'])
        return x
