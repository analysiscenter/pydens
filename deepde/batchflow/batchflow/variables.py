""" Contains Variable class and Variables storage class """
import copy as cp
import threading
import logging

from .named_expr import eval_expr, L


class Variable:
    """ Pipeline variable """
    def __init__(self, default=None, lock=True, pipeline=None, **kwargs):
        self.default = default
        if 'init_on_each_run' in kwargs:
            init_on_each_run = kwargs.get('init_on_each_run')
            if callable(init_on_each_run):
                self.default = L(init_on_each_run)
            else:
                self.default = init_on_each_run
            self._init_on_each_run = True
        else:
            self._init_on_each_run = kwargs.get('_init_on_each_run', False)
        self._lock = threading.Lock() if lock else None
        self.value = None
        if not self.init_on_each_run:
            self.initialize(pipeline=pipeline)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_lock'] = state['_lock'] is not None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock() if state['_lock'] else None

    @property
    def init_on_each_run(self):
        return self._init_on_each_run

    def get(self):
        """ Return a variable value """
        return self.value

    def set(self, value):
        """ Assign a variable value """
        self.value = value

    def initialize(self, pipeline=None):
        """ Initialize a variable value """
        value = eval_expr(self.default, pipeline=pipeline)
        self.set(value)

    def lock(self):
        """ Acquire lock """
        if self._lock:
            self._lock.acquire()

    def unlock(self):
        """ Release lock """
        if self._lock:
            self._lock.release()


class VariableDirectory:
    """ Storage for pipeline variables """
    def __init__(self):
        self.variables = {}
        self._lock = threading.Lock()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_lock')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def lock(self, name=None):
        """ Lock the directory itself or a variable """
        if name is None:
            if self._lock:
                self._lock.acquire()
        else:
            self.variables[name].lock()

    def unlock(self, name=None):
        """ Unlock the directory itself or a variable """
        if name is None:
            if self._lock:
                self._lock.release()
        else:
            self.variables[name].unlock()

    def copy(self):
        return cp.copy(self)

    def __copy__(self):
        """ Make a shallow copy of the directory """
        new_dir = VariableDirectory()
        new_dir.variables = {**self.variables}
        return new_dir

    def __add__(self, other):
        if not isinstance(other, VariableDirectory):
            raise TypeError("VariableDirectory is expected, but given '%s'" % type(other).__name__)

        new_dir = self.copy()
        new_dir.variables.update(other.variables)
        return new_dir

    def items(self):
        """ Return a sequence of (name, params) for all variables """
        for v in self.variables:
            var = self.variables[v].__getstate__()
            var.pop('value')
            var['lock'] = var['_lock']
            var.pop('_lock')
            yield v, var

    def exists(self, name):
        """ Checks if a variable already exists """
        return name in self.variables

    def create(self, name, *args, pipeline=None, **kwargs):
        """ Create a variable """
        if not self.exists(name):
            with self._lock:
                if not self.exists(name):
                    self.variables[name] = Variable(*args, pipeline=pipeline, **kwargs)

    def create_many(self, variables, pipeline=None):
        """ Create many variables at once """
        if isinstance(variables, (tuple, list)):
            variables = dict(zip(variables, [None] * len(variables)))

        for name, var in variables.items():
            var = var or {}
            var.pop('args', ())
            kwargs = var.pop('kwargs', {})
            self.create(name, **var, **kwargs, pipeline=pipeline)

    def init_on_run(self, pipeline=None):
        """ Initialize all variables before a pipeline is run """
        with self._lock:
            for v in self.variables:
                if self.variables[v].init_on_each_run:
                    self.variables[v].initialize(pipeline=pipeline)

    def get(self, name, *args, create=False, pipeline=None, **kwargs):
        """ Return a variable value """
        create = create or len(args) + len(kwargs) > 0
        if not self.exists(name):
            if create:
                self.variable.create(name, *args, pipeline=pipeline, **kwargs)
            else:
                raise KeyError("Variable '%s' does not exists" % name)
        var = self.variables[name].get()
        return var

    def set(self, name, value):
        """ Set a variable value """
        if not self.exists(name):
            raise KeyError("Variable '%s' does not exist" % name)
        self.variables[name].set(value)

    def delete(self, name):
        """ Remove the variable with a given name """
        if not self.exists(name):
            logging.warning("Variable '%s' does not exist", name)
        else:
            self.variables.pop(name)
