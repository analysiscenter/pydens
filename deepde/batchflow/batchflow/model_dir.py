""" Pipeline decorators """
import threading

from .models import BaseModel
from .named_expr import NamedExpression, eval_expr


class NonInitializedModel:
    """ Reference to a dynamic model that has not been created yet """
    def __init__(self, model_class, config=None):
        self.model_class = model_class
        self.config = config

    @property
    def default_name(self):
        """: str - the model class name (serve as a default for a model name) """
        if isinstance(self.model_class, NamedExpression):
            raise ValueError("Model name should be explicitly set if a model class is a named expression",
                             self.model_class)
        return self.model_class.__name__


class ModelDirectory:
    """ Model storage """
    def __init__(self):
        self.models = {}
        self.lock = threading.Lock()

    def __getstate__(self):
        return {'models': self.models}

    def __setstate__(self, state):
        self.models = state['models']
        self.lock = threading.Lock()

    def __repr__(self):
        return repr(self.models)

    def copy(self):
        """ Make a shallow copy of the directory """
        new_md = ModelDirectory()
        new_md.models = {**self.models}
        return new_md

    def eval_expr(self, expr, batch=None):
        """ Evaluate all named expressions in a given data structure """
        return eval_expr(expr, batch=batch)

    def get(self, name):
        """ Retrieve a model from a directory without building it

        Parameters
        ----------
        name : str
            model name

        Returns
        -------
        model
        """
        return self.models.get(name)

    def get_model_by_name(self, name, batch=None):
        """ Retrieve a model from a directory

        Parameters
        ----------
        name : str
            model name

        Returns
        -------
        model

        Raises
        ------
        KeyError
            if there is no model with a given name
        """
        model = self.get(name)
        if model is None:
            raise KeyError("Model '%s' does not exist" % name)
        if isinstance(model, NonInitializedModel):
            with self.lock:
                model = self.get(name)
                if isinstance(model, NonInitializedModel):
                    config = self.eval_expr(model.config, batch=batch) or {}
                    model_class = self.eval_expr(model.model_class, batch=batch)
                    model = self.create_model(model_class, config)
                    self.models[name] = model
        return model

    def create_model(self, model_class, config=None):
        """ Create a model """
        model = model_class(config=config)
        return model

    def add_model(self, name, model):
        """ Add a model to the directory """
        if name is None:
            name = model.default_name
        with self.lock:
            self.models.update({name: model})

    def init_model(self, mode, model_class, name=None, config=None):
        """ Initialize a static or dynamic model

        Parameters
        ----------
        mode : {'static', 'dynamic'}
        model_class : class or named expression
            a model class
        name : str
            a name for the model. Default - a model class name.
        config : dict
            model configurations parameters, where each key and value could be named expressions
        """
        if mode == 'static':
            model = self.create_model(model_class, config)
        else:
            model = NonInitializedModel(model_class, config)
        self.add_model(name, model)

    def import_model(self, source, pipeline=None, name=None):
        """ Import model from another pipeline or a model itself """
        if isinstance(source, BaseModel):
            model = source
        else:
            model = pipeline.models.get(source)
            name = name or source
        self.add_model(name, model)

    def save_model(self, name, *args, **kwargs):
        model = self.get_model_by_name(name)
        model.save(*args, **kwargs)

    def load_model(self, mode, model_class, name=None, *args, build=None, **kwargs):
        """ Load a model """
        _ = args
        config = {'load': kwargs, 'build': build}
        self.init_model(mode, model_class, name, config=config)

    def __add__(self, other):
        if not isinstance(other, ModelDirectory):
            raise TypeError("ModelDirectory is expected, but given '%s'" % type(other).__name__)

        new_md = self.copy()
        new_md.models.update(other.models)
        return new_md
