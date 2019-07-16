""" Cotains base class for Torch models """
import os
import re
import threading
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from ... import Config
from .. import BaseModel
from ..utils import unpack_fn_from_config
from .layers import ConvBlock
from .losses import CrossEntropyLoss


LOSSES = {
    'mse': nn.MSELoss,
    'bce': nn.BCEWithLogitsLoss,
    'ce': CrossEntropyLoss,
    'crossentropy': CrossEntropyLoss,
    'absolutedifference': nn.L1Loss,
    'l1': nn.L1Loss,
    'cosine': nn.CosineSimilarity,
    'cos': nn.CosineSimilarity,
    'hinge': nn.HingeEmbeddingLoss,
    'huber': nn.SmoothL1Loss,
    'logloss': CrossEntropyLoss,
}


DECAYS = {
    'exp': torch.optim.lr_scheduler.ExponentialLR,
}


class TorchModel(BaseModel):
    r""" Base class for torch models

    **Configuration**

    ``build`` and ``load`` are inherited from :class:`.BaseModel`.

    device : str or torch.device
        if str, a device name (e.g. 'cpu' or 'gpu:0').

    inputs : dict
        model inputs (see :meth:`~.TorchModel._make_inputs`)

    loss - a loss function, might be defined in one of three formats:
        - name
        - tuple (name, args)
        - dict {'name': name, \**args}

        where name might be one of:
            - short name (`'mse'`, `'ce'`, `'l1'`, `'cos'`, `'hinge'`, `'huber'`, `'logloss'`, `'dice'`)
            - a class name from `torch losses <https://pytorch.org/docs/stable/nn.html#loss-functions>`_
              (e.g. `'PoissonNLL'` or `'TripletMargin'`)
            - a module class
            - callable

        Examples:

        - ``{'loss': 'mse'}``
        - ``{'loss': ('KLDiv', {'reduction': 'none'})``
        - ``{'loss': {'name': MyCustomLoss, 'epsilon': 1e-6}}``
        - ``{'loss': my_custom_loss_fn}``

    decay - a learning rate decay algorithm might be defined in one of three formats:
        - name
        - tuple (name, args)
        - dict {'name': name, **args}

        where name might be one of:

        - short name ('exp')
        - a class name from `torch.optim.lr_scheduler
          <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_
          (e.g. 'LambdaLR')
        - a class with ``_LRScheduler`` interface
        - a callable which takes optimizer and optional args

        Examples:

        - ``{'decay': 'exp'}``
        - ``{'decay': ('StepLR', {'steps_size': 10000})}``
        - ``{'decay': {'name': MyCustomDecay, 'decay_rate': .5}``

    optimizer - an optimizer might be defined in one of three formats:
            - name
            - tuple (name, args)
            - dict {'name': name, \**args}

            where name might be one of:

            - short name (e.g. 'Adam', 'Adagrad', any optimizer from
              `torch.optim <https://pytorch.org/docs/stable/optim.html#algorithms>`_)
            - a class with ``Optimizer`` interface
            - a callable which takes model parameters and optional args.

        Examples:

        - ``{'optimizer': 'Adam'}``
        - ``{'optimizer': ('SparseAdam', {'lr': 0.01})}``
        - ``{'optimizer': {'name': 'Adagrad', 'initial_accumulator_value': 0.01}``
        - ``{'optimizer': {'name': MyCustomOptimizer, momentum=0.95}}``

    microbatch : int
        make forward/backward pass with microbatches of a given size, but apply gradients after the whole batch.
        Batch size should be evenly divisible by microbatch size.

    common : dict
        default parameters for all blocks (see :class:`.ConvBlock`)

    initial_block : dict or nn.Module
        a user-defined module or parameters for the input block, usually :class:`.ConvBlock` parameters.

        The only required parameter here is ``initial_block/inputs`` which should contain a name or
        a list of names from ``inputs`` which tensors will be passed to ``initial_block`` as ``inputs``.

        Examples:

        - ``{'initial_block/inputs': 'images'}``
        - ``{'initial_block': dict(inputs='features')}``
        - ``{'initial_block': dict(inputs='images', layout='nac nac', filters=64, kernel_size=[7, 3], strides=[1, 2])}``
        - ``{'initial_block': MyCustomModule(some_param=1, another_param=2)}``

    body : dict or nn.Module
        a user-defined module or parameters for the base network layers, usually :class:`.ConvBlock` parameters

    head : dict or nn.Module
        a user-defined module or parameters for the head layers, usually :class:`.ConvBlock` parameters

    predictions : str or callable
        an operation applied to the head output to make the predictions tensor which is used in the loss function.

    output : dict or list
        auxiliary operations

    For more details about predictions and auxiliary output operations see :meth:`.TorchModel.output`.

    **How to create your own model**

    #. Take a look at :class:`~.ConvBlock` since it is widely used as a building block almost everywhere.

    #. Define model defaults (e.g. number of filters, batch normalization options, etc)
       by overriding :meth:`.TorchModel.default_config`.
       Or skip it and hard code all the parameters in unpredictable places without the possibility to
       change them easily through model's config.

    #. Define build configuration (e.g. number of classes, etc)
       by overriding :meth:`~.TorchModel.build_config`.

    #. Override :meth:`~.TorchModel.initial_block`, :meth:`~.TorchModel.body` and :meth:`~.TorchModel.head`, if needed.
       In many cases defaults and build config are just enough to build a network without additional code writing.

    Things worth mentioning:

    #. Input data and its parameters should be defined in configuration under ``inputs`` key.
       See :meth:`.TorchModel._make_inputs` for details.

    #. You might want to use a convenient multidimensional :class:`.ConvBlock`,
       as well as other predefined layers from ``dataset.models.torch.layers``.
       Of course, you can use usual `Torch layers <https://pytorch.org/docs/stable/nn.html>`_.

    #. In many cases there is no need to write a loss function, learning rate decay and optimizer
       as they might be defined through config.

    #. For a configured loss to work one of the inputs should have a name ``targets`` and
       the model output is considered ``predictions``.
       They will be passed to a loss function.
    """
    def __init__(self, *args, **kwargs):
        self._train_lock = threading.Lock()
        self.device = None
        self.loss_fn = None
        self.lr_decay = None
        self.optimizer = None
        self.model = None
        self._inputs = dict()
        self.predictions = None
        self.loss = None
        self.microbatch = None
        self._full_config = None

        super().__init__(*args, **kwargs)

    def build(self, *args, **kwargs):
        """ Build the model """
        config = self.build_config()
        self._full_config = config

        self.device = self._get_device()

        self._build(config)

        if self.loss_fn is None:
            self._make_loss(config)
        if self.optimizer is None:
            self._make_optimizer(config)

        self.microbatch = config.get('microbatch', None)


    def _make_inputs(self, names=None, config=None):
        """ Create model input data from config provided

        **Configuration**

        inputs : dict
            - key : str
                a placeholder name
            - values : dict or tuple
                each input's config

        Input config:

        ``dtype`` : str or torch.dtype (by default 'float32')
            data type

        ``shape`` : int, tuple, list or None (default)
            a tensor shape which includes the number of channels/classes and doesn't include a batch size.

        ``classes`` : int, array-like or None (default)
            an array of class labels if data labels are strings or anything else except ``np.arange(num_classes)``

        ``data_format`` : str {'channels_first', 'channels_last'} or {'f', 'l'}
            The ordering of the dimensions in the inputs. Default is 'channels_last'.
            For brevity ``data_format`` may be shortened to ``df``.

        ``name`` : str
            a name for the transformed and reshaped tensor.

        If an input config is a tuple, it should contain all items exactly in the order shown above:
        dtype, shape, classes, data_format, transform, name.
        If an item is None, the default value will be used instead.


        Parameters
        ----------
        names : list
            tensor names that are expected in the config's 'inputs' section

        Raises
        ------
        KeyError if there is any name missing in the config's 'inputs' section.
        ValueError if there are duplicate names.
        """
        config = config.get('inputs')

        names = names or []
        missing_names = set(names) - set(config.keys())
        if len(missing_names) > 0:
            raise KeyError("Inputs should contain {} names".format(missing_names))

        placeholder_names = set(config.keys())
        tensor_names = set(x.get('name') for x in config.values() if isinstance(x, dict) and x.get('name'))
        wrong_names = placeholder_names & tensor_names
        if len(wrong_names) > 0:
            raise ValueError('Inputs contain duplicate names:', wrong_names)

        param_names = ('dtype', 'shape', 'classes', 'transform', 'name')

        for input_name, input_config in config.items():
            if isinstance(input_config, (tuple, list)):
                input_config = list(input_config) + [None for _ in param_names]
                input_config = input_config[:len(param_names)]
                input_config = dict(zip(param_names, input_config))
                input_config = dict((k, v) for k, v in input_config.items() if v is not None)

            shape = input_config.get('shape')
            if isinstance(shape, int):
                shape = (shape,)
            if shape:
                input_config['shape'] = tuple([None] + list(shape))

            self._inputs[input_name] = dict(config=input_config)

        # add default aliases
        if 'targets' not in config:
            if 'labels' in config:
                self._inputs['targets'] = self._inputs['labels']
            elif 'masks' in config:
                self._inputs['targets'] = self._inputs['masks']

    def _get_device(self):
        device = self.config.get('device')
        if device is torch.device or device is None:
            _device = device
        elif isinstance(device, str):
            _device = device.split(':')
            unit, index = _device if len(_device) > 1 else (device, '0')
            if unit.lower() in ['gpu', 'cpu']:
                unit = 'cuda' if unit.lower() == 'gpu' else 'cpu'
                _device = torch.device(unit, int(index))
            else:
                raise ValueError('Unknown device type: ', device)
        else:
            raise TypeError('Wrong device type: ', type(device))
        return _device

    def _make_loss(self, config):
        loss, args = unpack_fn_from_config('loss', config)

        if isinstance(loss, str):
            loss = LOSSES.get(re.sub('[-_ ]', '', loss).lower(), None)
        elif isinstance(loss, str) and hasattr(nn, loss):
            loss = getattr(nn, loss)
        elif isinstance(loss, str) and hasattr(nn, loss + "Loss"):
            loss = getattr(nn, loss + "Loss")
        elif isinstance(loss, type):
            pass
        elif callable(loss):
            loss = lambda **a: partial(loss, **args)
            args = {}
        else:
            raise ValueError("Loss is not defined in the model %s" % self.__class__.__name__)

        self.loss_fn = loss(**args)

    def _make_optimizer(self, config):
        optimizer_name, optimizer_args = unpack_fn_from_config('optimizer', config)

        if optimizer_name is None or callable(optimizer_name) or isinstance(optimizer_name, type):
            pass
        elif isinstance(optimizer_name, str) and hasattr(torch.optim, optimizer_name):
            optimizer_name = getattr(torch.optim, optimizer_name)
        else:
            raise ValueError("Unknown optimizer", optimizer_name)

        if optimizer_name:
            self.optimizer = optimizer_name(self.model.parameters(), **optimizer_args)
        else:
            raise ValueError("Optimizer is not defined", optimizer_name)

        decay_name, decay_args = self._make_decay(config)
        if decay_name is not None:
            self.lr_decay = decay_name(self.optimizer, **decay_args)

    def _make_decay(self, config):
        decay_name, decay_args = unpack_fn_from_config('decay', config)

        if decay_name is None or callable(decay_name) or isinstance(decay_name, type):
            pass
        elif isinstance(decay_name, str) and hasattr(torch.optim.lr_scheduler, decay_name):
            decay_name = getattr(torch.optim.lr_scheduler, decay_name)
        elif decay_name in DECAYS:
            decay_name = DECAYS.get(re.sub('[-_ ]', '', decay_name).lower(), None)
        else:
            raise ValueError("Unknown learning rate decay method", decay_name)

        return decay_name, decay_args

    def get_tensor_config(self, tensor):
        """ Return tensor configuration """
        if isinstance(tensor, str):
            return self._inputs[tensor]['config']
        raise TypeError("tensor is expected to be a name for config's inputs section")

    def shape(self, tensor):
        """ Return the tensor's shape """
        if isinstance(tensor, (list, tuple)):
            return tuple(self.get_tensor_config(t)['shape'] for t in tensor)
        return self.get_tensor_config(tensor)['shape']

    def num_channels(self, tensor, data_format='channels_first'):
        """ Return number of channels in the input tensor """
        shape = self.shape(tensor)
        axis = self.channels_axis(data_format)
        return shape[axis]

    def spatial_dim(self, tensor):
        shape = self.shape(tensor)
        return len(shape) - 2


    @classmethod
    def channels_axis(cls, data_format='channels_first'):
        """ Return the channels axis for the tensor

        Parameters
        ----------
        data_format : str {'channels_last', 'channels_first', 'N***'} or None

        Returns
        -------
        int
        """
        data_format = data_format if data_format else 'channels_first'

        return 1 if data_format == "channels_first" or data_format.startswith("NC") else -1

    def has_classes(self, tensor):
        """ Check if a tensor has classes defined in the config """
        config = self.get_tensor_config(tensor)
        has = config.get('classes') is not None
        return has

    def classes(self, tensor):
        """ Return the  number of classes """
        config = self.get_tensor_config(tensor)
        classes = config.get('classes')
        if isinstance(classes, int):
            return np.arange(classes)
        return np.asarray(classes)

    def num_classes(self, tensor):
        """ Return the number of classes """
        if self.has_classes(tensor):
            classes = self.classes(tensor)
            return classes if isinstance(classes, int) else len(classes)
        return self.num_channels(tensor)

    @classmethod
    def default_config(cls):
        """ Define model defaults

        You need to override this method if you expect your model or its blocks to serve as a base for other models
        (e.g. VGG for FCN, ResNet for LinkNet, etc).

        Put here all constants (like the number of filters, kernel sizes, block layouts, strides, etc)
        specific to the model, but independent of anything else (like image shapes, number of classes, etc).

        These defaults can be changed in :meth:`~.TorchModel.build_config` or when calling :meth:`.Pipeline.init_model`.

        Usually, it looks like::

            @classmethod
            def default_config(cls):
                config = TorchModel.default_config()
                config['initial_block'] = dict(layout='cnap', filters=16, kernel_size=7, strides=2,
                                               pool_size=3, pool_strides=2)
                config['body/filters'] = 32
                config['head'] = dict(layout='cnadV', dropout_rate=.2)
                return config
        """
        config = Config()
        config['inputs'] = {}
        config['common'] = {}
        config['initial_block'] = {}
        config['body'] = {}
        config['head'] = {}
        config['predictions'] = None
        config['output'] = None
        config['optimizer'] = ('Adam', dict())
        config['microbatch'] = None

        return config

    @classmethod
    def get_defaults(cls, name, kwargs):
        """ Fill block params from default config and kwargs """
        config = cls.default_config()
        _config = config.get(name)
        kwargs = kwargs or {}
        config = {**config['common'], **_config, **kwargs}
        return config

    def build_config(self, names=None):
        """ Define a model architecture configuration

        It takes just 2 steps:

        #. Define names for input data and make input tensors by calling ``super().build_config(names)``.

           If the model config does not contain any name from ``names``, :exc:`KeyError` is raised.

           See :meth:`._TorchModel.make_inputs` for details.

        #. Define parameters for :meth:`~.TorchModel.initial_block`, :meth:`~.TorchModel.body`,
           :meth:`~.TorchModel.head` which depend on inputs.

        #. Don't forget to return ``config``.

        Typically it looks like this::

            def build_config(self, names=None):
                names = names or ['images', 'labels']
                config = super().build_config(names)
                config['head/num_classes'] = self.num_classes('targets')
                return config
        """
        config = self.default_config()
        config = config + self.config

        if config.get('inputs'):
            self._make_inputs(names, config)
            inputs = self.get('initial_block/inputs', config)
            if isinstance(inputs, str):
                config['common/data_format'] = config['inputs'][inputs].get('data_format')

        return config

    def _add_block(self, blocks, name, config, inputs):
        if isinstance(config[name], nn.Module):
            block = config[name]
        elif isinstance(config[name], dict):
            block = getattr(self, name)(inputs=inputs, **{**config['common'], **config[name]})
        else:
            raise TypeError('block can be configured as a Module or a dict with parameters')
        if block is not None:
            blocks.append(block)
        return block

    def _build(self, config=None):
        initial_inputs = self.shape(config['initial_block/inputs'])
        config.pop('initial_block/inputs')

        blocks = []
        initial_block = self._add_block(blocks, 'initial_block', config, initial_inputs)
        body = self._add_block(blocks, 'body', config, initial_block or initial_inputs)
        self._add_block(blocks, 'head', config, body or initial_block or initial_inputs)

        self.model = nn.Sequential(*blocks)

        if self.device:
            self.model.to(self.device)

    @classmethod
    def initial_block(cls, **kwargs):
        """ Transform inputs with a convolution block

        Notes
        -----
        For parameters see :class:`.ConvBlock`.

        Returns
        -------
        torch.nn.Module or None
        """
        kwargs = cls.get_defaults('initial_block', kwargs)
        if kwargs.get('layout'):
            return ConvBlock(**kwargs)
        return None

    @classmethod
    def body(cls, **kwargs):
        """ Base layers which produce a network embedding

        Notes
        -----
        For parameters see :class:`.ConvBlock`.

        Returns
        -------
        torch.nn.Module or None
        """
        kwargs = cls.get_defaults('body', kwargs)
        if kwargs.get('layout'):
            return ConvBlock(**kwargs)
        return None

    @classmethod
    def head(cls, **kwargs):
        """ The last network layers which produce predictions

        Notes
        -----
        For parameters see :class:`.ConvBlock`.

        Returns
        -------
        torch.nn.Module or None
        """
        kwargs = cls.get_defaults('head', kwargs)
        if kwargs.get('layout'):
            return ConvBlock(**kwargs)
        return None

    def output(self, inputs, predictions=None, ops=None, prefix=None, **kwargs):
        """ Add output operations to the model, like predicted probabilities or labels, etc.

        Parameters
        ----------
        inputs : torch.Tensor or a sequence of torch.Tensors
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
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("Network output is expected to be a Tensor, but given {}".format(type(inputs)))

            prefix = [*ops.keys()][i]
            attr_prefix = prefix + '_' if prefix else ''

            self._add_output_op(tensor, predictions, 'predictions', '', **kwargs)
            for oper in ops[prefix]:
                self._add_output_op(tensor, oper, oper, attr_prefix, **kwargs)

    def _add_output_op(self, inputs, oper, name, attr_prefix, **kwargs):
        if oper is None:
            self._add_output_identity(inputs, name, attr_prefix, **kwargs)
        elif oper == 'softplus':
            self._add_output_softplus(inputs, name, attr_prefix, **kwargs)
        elif oper == 'sigmoid':
            self._add_output_sigmoid(inputs, name, attr_prefix, **kwargs)
        elif oper == 'proba':
            self._add_output_proba(inputs, name, attr_prefix, **kwargs)
        elif oper == 'labels':
            self._add_output_labels(inputs, name, attr_prefix, **kwargs)
        elif callable(oper):
            self._add_output_callable(inputs, oper, None, attr_prefix, **kwargs)

    def _add_output_identity(self, inputs, name, attr_prefix, **kwargs):
        _ = kwargs
        setattr(self, attr_prefix + name, inputs)
        return inputs

    def _add_output_softplus(self, inputs, name, attr_prefix, **kwargs):
        _ = kwargs
        proba = torch.nn.functional.softplus(inputs)
        setattr(self, attr_prefix + name, proba)

    def _add_output_sigmoid(self, inputs, name, attr_prefix, **kwargs):
        _ = kwargs
        proba = torch.nn.functional.sigmoid(inputs)
        setattr(self, attr_prefix + name, proba)

    def _add_output_proba(self, inputs, name, attr_prefix, **kwargs):
        axis = self.channels_axis(kwargs.get('data_format'))
        proba = torch.nn.functional.softmax(inputs, dim=axis)
        setattr(self, attr_prefix + name, proba)

    def _add_output_labels(self, inputs, name, attr_prefix, **kwargs):
        class_axis = self.channels_axis(kwargs.get('data_format'))
        predicted_classes = inputs.argmax(dim=class_axis)
        setattr(self, attr_prefix + name, predicted_classes)

    def _add_output_callable(self, inputs, oper, name, attr_prefix, **kwargs):
        _ = kwargs
        x = oper(inputs)
        name = name or oper.__name__
        setattr(self, attr_prefix + name, x)
        return x

    def _fill_value(self, inputs):
        inputs = torch.from_numpy(inputs)
        if self.device:
            inputs = inputs.to(self.device)
        return inputs

    def _fill_param(self, inputs):
        if inputs is None:
            pass
        elif isinstance(inputs, tuple):
            inputs_list = []
            for i in inputs:
                v = self._fill_value(i)
                inputs_list.append(v)
            inputs = inputs_list
        else:
            inputs = self._fill_value(inputs)
        return inputs

    def _fill_input(self, *args):
        inputs = []
        for arg in args:
            inputs.append(self._fill_param(arg))
        return tuple(inputs)

    def _fill_output(self, fetches):
        _fetches = [fetches] if isinstance(fetches, str) else fetches

        output = []
        for f in _fetches:
            if hasattr(self, f):
                v = getattr(self, f)
                if isinstance(v, (torch.Tensor, torch.autograd.Variable)):
                    v = v.detach().cpu().numpy()
                output.append(v)
            else:
                raise KeyError('Unknown value to fetch', f)

        output = output[0] if isinstance(fetches, str) else type(fetches)(output)

        return output

    def train(self, *args, fetches=None, use_lock=False, microbatch=None):    # pylint: disable=arguments-differ
        """ Train the model with the data provided

        Parameters
        ----------
        args
            arguments to be passed directly into the model

        fetches : tuple, list
            a sequence of `tf.Operation` and/or `tf.Tensor` to calculate

        use_lock : bool
            if True, the whole train step is locked, thus allowing for multithreading.

        microbatch : int or None
            make forward/backward pass with microbatches of a given size, but apply gradients after the whole batch.
            Batch size should be evenly divisible by microbatch size.

        Returns
        -------
        Calculated values of tensors in `fetches` in the same structure


        Examples
        --------

        ::

            model.train(B('images'), B('labels'), fetches='loss')
        """
        if use_lock:
            self._train_lock.acquire()

        *inputs, targets = self._fill_input(*args)

        self.model.train()

        if self.lr_decay:
            self.lr_decay()

        if microbatch is not False:
            if microbatch is True:
                microbatch = self.microbatch
            else:
                microbatch = microbatch or self.microbatch
        if microbatch:
            if len(inputs[0]) % microbatch != 0:
                raise ValueError("Inputs size should be evenly divisible by microbatch size: %d and %d" %
                                 (len(inputs), microbatch))

            self.optimizer.zero_grad()

            steps = len(inputs[0]) // microbatch
            predictions = []
            for i in range(0, len(inputs[0]), microbatch):
                inputs_ = [data[i: i + microbatch] for data in inputs]
                targets_ = targets[i: i + microbatch]

                predictions.append(self.model(*inputs_))
                self.loss = self.loss_fn(predictions[-1], targets_)
                self.loss.backward()

            self.optimizer.step()
            self.predictions = torch.cat(predictions)
            self.loss = self.loss / steps
        else:
            self.optimizer.zero_grad()
            self.predictions = self.model(*inputs)
            self.loss = self.loss_fn(self.predictions, targets)
            self.loss.backward()
            self.optimizer.step()

        if use_lock:
            self._train_lock.release()

        config = self._full_config
        self.output(inputs=self.predictions, predictions=config['predictions'],
                    ops=config['output'], **config['common'])
        output = self._fill_output(fetches)

        return output

    def predict(self, *args, targets=None, fetches=None):    # pylint: disable=arguments-differ
        """ Get predictions on the data provided

        Parameters
        ----------
        args
            arguments to be passed directly into the model

        targets
            (optional) targets to calculate loss

        fetches : tuple, list
            a sequence of tensors to fetch from the model

        use_lock : bool
            if True, the whole train step is locked, thus allowing for multithreading.

        microbatch : int or None
            make forward/backward pass with microbatches of a given size, but apply gradients after the whole batch.
            Batch size should be evenly divisible by microbatch size.

        Returns
        -------
        Calculated values of tensors in `fetches` in the same structure


        Examples
        --------

        ::

            model.predict(B('images'), targets=B('labels'), fetches='loss')
        """
        inputs = self._fill_input(*args)
        if targets is not None:
            targets = self._fill_input(targets)[0]

        self.model.eval()

        with torch.no_grad():
            self.predictions = self.model(*inputs)
            if targets is None:
                self.loss = None
            else:
                self.loss = self.loss_fn(self.predictions, targets)

        config = self._full_config
        self.output(inputs=self.predictions, predictions=config['predictions'],
                    ops=config['output'], **config['common'])
        output = self._fill_output(fetches)
        return output

    def save(self, path, *args, **kwargs):
        """ Save torch model.

        Parameters
        ----------
        path : str
            a path to a file where the model data will be stored

        Examples
        --------
        >>> torch_model = ResNet34()

        Now save the model

        >>> torch_model.save('/path/to/models/resnet34')

        The model will be saved to /path/to/models/resnet34
        """
        _ = args, kwargs
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save({
            'model_state_dict': self.model,
            'optimizer_state_dict': self.optimizer,
            'loss': self.loss_fn,
            'config': self.config,
            'full_config': self._full_config
            }, path)

    def load(self, path, *args, eval=False, **kwargs):
        """ Load a torch model from files

        Parameters
        ----------
        path : str
            a file path where a model is stored

        eval : bool
            whether to switch the model to eval mode

        Examples
        --------
        >>> resnet = ResNet34(load=dict(path='/path/to/models/resnet34'))

        >>> torch_model.load(path='/path/to/models/resnet34')

        >>> TorchModel(config={'device': 'gpu:2', 'load/path': '/path/to/models/resnet34'})

        **How to move the model to device**

        The model will be moved to device specified in the model config by key `device`.
        """
        _ = args, kwargs
        device = self._get_device()
        if device:
            checkpoint = torch.load(path, map_location=device)
        else:
            checkpoint = torch.load(path)
        self.model = checkpoint['model_state_dict']
        self.optimizer = checkpoint['optimizer_state_dict']
        self.loss_fn = checkpoint['loss']
        self.config = self.config + checkpoint['config']
        self._full_config = checkpoint['full_config']

        self.device = device

        if device:
            self.model.to(device)

        if eval:
            self.model.eval()
