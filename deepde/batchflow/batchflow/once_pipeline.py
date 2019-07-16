""" Once pipeline """
import sys
import copy as cp
from functools import partial
import numpy as np

from .named_expr import NamedExpression, eval_expr
from ._const import ACTIONS, LOAD_MODEL_ID, SAVE_MODEL_ID


class OncePipeline:
    """ Pipeline that runs only once before or after the main pipeline """
    def __init__(self, pipeline=None, *namespaces):
        self.pipeline = pipeline
        self._namespaces = list(namespaces)
        self._actions = []

    @classmethod
    def concat(cls, pipe1, pipe2):
        """ Concatenate two pipelines """
        # pylint: disable=protected-access
        new_p = OncePipeline(pipe1.pipeline)
        new_p._actions = pipe1._actions + pipe2._actions
        new_p._namespaces = pipe1._namespaces + [a for a in pipe2._namespaces if a not in pipe1._namespaces]
        return new_p

    def __getstate__(self):
        state = dict(actions=self._actions, namespaces=self._namespaces, pipeline=self.pipeline)
        return state

    def __setstate__(self, state):
        self._actions = state['actions']
        self._namespaces = state['namespaces']
        self.pipeline = state['pipeline']

    def copy(self):
        """ Make a shallow copy of the dataset object """
        return cp.copy(self)

    def __add__(self, other):
        if isinstance(other, OncePipeline):
            return self.pipeline + other
        return other + self

    @property
    def _all_namespaces(self):
        return [sys.modules["__main__"]] + self._namespaces

    def has_method(self, name):
        return any(hasattr(namespace, name) for namespace in self._all_namespaces)

    def get_method(self, name):
        """ Return a method by the name """
        for namespace in self._all_namespaces:
            if hasattr(namespace, name):
                return getattr(namespace, name)
        return None

    def _add_action(self, name, *args, _args=None, save_to=None, **kwargs):
        action = {'name': name, 'args': args, 'kwargs': kwargs, 'save_to': save_to}
        if _args:
            action.update(**_args)
        self._actions.append(action)
        return self

    def __getattr__(self, name):
        if self.has_method(name):
            return partial(self._add_action, name)
        raise AttributeError("Unknown name: %s" % name)

    def add_namespace(self, *namespaces):
        self._namespaces.extend(namespaces)
        return self

    def _exec_action(self, action):
        args_value = eval_expr(action['args'], pipeline=self.pipeline)
        kwargs_value = eval_expr(action['kwargs'], pipeline=self.pipeline)

        if action['name'] in ACTIONS:
            method = getattr(self, ACTIONS[action['name']])
            method(action)
        else:
            method = self.get_method(action['name'])
            if method is None:
                raise ValueError("Unknown method: %s" % action['name'])

            res = method(*args_value, **kwargs_value)

            if isinstance(action['save_to'], NamedExpression):
                action['save_to'].set(res, pipeline=self.pipeline)
            elif isinstance(action['save_to'], np.ndarray):
                action['save_to'][:] = res

    def run(self):
        """ Execute all actions """
        for action in self._actions:
            self._exec_action(action)
        return self

    def init_variable(self, name, default=None, lock=True, **kwargs):
        """ Create a variable if not exists.
        If the variable exists, does nothing.

        Parameters
        ----------
        name : string
            a name of the variable
        default
            an initial value for the variable set when pipeline is created
        init_on_each_run
            an initial value for the variable to set before each run
        lock : bool
            whether to lock a variable before each update (default: True)

        Returns
        -------
        self - in order to use it in the pipeline chains

        Examples
        --------
        >>> pp = dataset.p.before
                    .init_variable("iterations", default=0)
                    .init_variable("accuracy", init_on_each_run=0)
                    .init_variable("loss_history", init_on_each_run=list)
        """
        self.pipeline.variables.create(name, default, lock=lock, pipeline=self, **kwargs)
        return self

    def init_model(self, mode, model_class=None, name=None, config=None):
        """ Initialize a static or dynamic model

        Parameters
        ----------
        mode : {'static', 'dynamic'}
        model_class : class
            a model class
        name : str
            a name for the model. Default - a model class name.
        config : dict
            model configurations parameters, where each key and value could be named expressions.

        Examples
        --------
        >>> pipeline.before.init_model('static', MyModel)

        >>> pipeline.before
              .init_variable('images_shape', [256, 256])
              .init_model('static', MyModel, config={'input_shape': V('images_shape')})

        >>> pipeline.before
              .init_variable('shape_name', 'images_shape')
              .init_model('dynamic', C('model'), config={V('shape_name)': B('images_shape')})

        >>> pipeline.before
              .init_model('dynamic', MyModel, config={'input_shape': C(lambda batch: batch.images.shape[1:])})
        """
        self.pipeline.models.init_model(mode, model_class, name, config=config)
        return self

    def save_model(self, name, *args, **kwargs):
        """ Save a model """
        return self._add_action(SAVE_MODEL_ID, *args, _args=dict(model_name=name), **kwargs)

    def _exec_save_model(self, action):
        self.pipeline._exec_save_model(None, action)        # pylint:disable=protected-access

    def load_model(self, mode, model_class=None, name=None, *args, **kwargs):
        """ Load a model """
        if mode == 'static':
            self.pipeline.models.load_model(mode, model_class, name, *args, **kwargs)
            return self
        return self._add_action(LOAD_MODEL_ID, *args,
                                _args=dict(mode=mode, model_class=model_class, model_name=name),
                                **kwargs)

    def _exec_load_model(self, action):
        self.pipeline._exec_load_model(None, action)        # pylint:disable=protected-access
