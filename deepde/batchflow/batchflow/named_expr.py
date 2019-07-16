""" Contains named expression classes"""
import numpy as np


class _DummyBatch:
    """ A fake batch for static models """
    def __init__(self, pipeline):
        self.pipeline = pipeline


class NamedExpression:
    """ Base class for a named expression

    Attributes
    ----------
    name : str
        a name
    mode : str
        a default assignment method: write, append, extend, update.
        Can be shrotened to jiust the first letter: w, a, e, u.

        - 'w' - overwrite with a new value. This is a default mode.
        - 'a' - append a new value
                (see list.append https://docs.python.org/3/tutorial/datastructures.html#more-on-lists)
        - 'e' - extend with a new value
                (see list.extend https://docs.python.org/3/tutorial/datastructures.html#more-on-lists)
        - 'u' - update with a new value
                (see dict.update https://docs.python.org/3/library/stdtypes.html#dict.update
                or set.update https://docs.python.org/3/library/stdtypes.html#frozenset.update)

    """
    def __init__(self, name, mode='w'):
        self.name = name
        self.mode = mode

    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a named expression

        Parameters
        ----------
        batch
            a batch which should be used to calculate a value
        pipeline
            a pipeline which should be used to calculate a value
            (might be omitted if batch is passed)
        model
            a model which should be used to calculate a value
            (usually omitted, but might be useful for F- and L-expressions)
        """
        if isinstance(self.name, NamedExpression):
            return self.name.get(batch=batch, pipeline=pipeline, model=model)
        return self.name

    def set(self, value, batch=None, pipeline=None, model=None, mode=None, eval=True):
        """ Set a value to a named expression

        Parameters
        ----------
        batch
            a batch which should be used to calculate a value
        pipeline
            a pipeline which should be used to calculate a value
            (might be omitted if batch is passed)
        model
            a model which should be used to calculate a value
            (usually omitted, but might be useful for F- and L-expressions)
        mode : str
            an assignment method: write, append, extend, update.
            A default mode may be specified when instantiating an expression.
        eval : bool
            whether to evaluate value before assigning it to the expression
            (as value might contain other named expressions,
            so it should be processed recursively)
        """
        mode = mode or self.mode
        if eval:
            value = eval_expr(value, batch=batch, pipeline=pipeline, model=model)
        if mode in ['a', 'append']:
            self.append(value, batch=batch, pipeline=pipeline, model=model)
        elif mode in ['e', 'extend']:
            self.extend(value, batch=batch, pipeline=pipeline, model=model)
        elif mode in ['u', 'update']:
            self.update(value, batch=batch, pipeline=pipeline, model=model)
        else:
            self.assign(value, batch=batch, pipeline=pipeline, model=model)

    def assign(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a named expression """
        raise NotImplementedError("assign should be implemented in child classes")

    def append(self, value, *args, **kwargs):
        """ Append a value to a named expression

        if a named expression is a dict or set, `update` is called, or `append` otherwise.

        See also
        --------
        list.append https://docs.python.org/3/tutorial/datastructures.html#more-on-lists
        dict.update https://docs.python.org/3/library/stdtypes.html#dict.update
        set.update https://docs.python.org/3/library/stdtypes.html#frozenset.update
        """
        var = self.get(*args, **kwargs)
        if var is None:
            self.assign(value, *args, **kwargs)
        elif isinstance(var, (set, dict)):
            var.update(value)
        else:
            var.append(value)

    def extend(self, value, *args, **kwargs):
        """ Extend a named expression with a new value
        (see list.extend https://docs.python.org/3/tutorial/datastructures.html#more-on-lists) """
        var = self.get(*args, **kwargs)
        if var is None:
            self.assign(value, *args, **kwargs)
        else:
            var.extend(value)

    def update(self, value, *args, **kwargs):
        """ Update a named expression with a new value
        (see dict.update https://docs.python.org/3/library/stdtypes.html#dict.update
        or set.update https://docs.python.org/3/library/stdtypes.html#frozenset.update) """
        var = self.get(*args, **kwargs)
        if var is not None:
            var.update(value)
        else:
            self.assign(value, *args, **kwargs)

    def __repr__(self):
        return type(self).__name__ + '(' + str(self.name) + ')'


class W(NamedExpression):
    """ A wrapper which returns the wrapped named expression without evaluating it

    Examples
    --------
    ::

        W(V('variable'))
        W(B(copy=True))
        W(R('normal', 0, 1, size=B('size')))
    """
    def get(self, batch=None, pipeline=None, model=None):
        """ Return a wrapped named expression """
        _ = batch, pipeline, model
        return self.name

    def assign(self, *args, **kwargs):
        """ Assign a value """
        _ = args, kwargs
        raise NotImplementedError("Assigning a value to a wrapper is not supported")


def eval_expr(expr, batch=None, pipeline=None, model=None):
    """ Evaluate a named expression recursively """
    if batch is None:
        batch = _DummyBatch(pipeline)
    args = dict(batch=batch, pipeline=pipeline, model=model)

    if isinstance(expr, NamedExpression):
        _expr = expr.get(**args)
        if isinstance(_expr, NamedExpression) and not isinstance(expr, W):
            expr = eval_expr(_expr, **args)
        else:
            expr = _expr
    elif isinstance(expr, (list, tuple)):
        _expr = []
        for val in expr:
            _expr.append(eval_expr(val, **args))
        expr = type(expr)(_expr)
    elif isinstance(expr, dict):
        _expr = type(expr)()
        for key, val in expr.items():
            key = eval_expr(key, **args)
            val = eval_expr(val, **args)
            _expr.update({key: val})
        expr = _expr
    return expr


class B(NamedExpression):
    """ Batch component or attribute name

    Notes
    -----
    ``B()`` return the batch itself.

    To avoid unexpected data changes the copy of the batch may be returned, if ``copy=True``.

    Examples
    --------
    ::

        B('size')
        B('images_shape')
        B(copy=True)
    """
    def __init__(self, name=None, mode='w', copy=False):
        super().__init__(name, mode)
        self.copy = copy

    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a batch component """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        if isinstance(batch, _DummyBatch):
            raise ValueError("Batch expressions are not allowed in static models: B('%s')" % name)
        if name is None:
            return batch.copy() if self.copy else batch
        return getattr(batch, name)

    def assign(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a batch component """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        if name is not None:
            setattr(batch, name, value)


class C(NamedExpression):
    """ A pipeline config option

    Examples
    --------
    ::

        C('model_class')
        C('GPU')
    """
    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a pipeline config """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        config = pipeline.config or {}
        try:
            value = config[name]
        except KeyError:
            raise KeyError("Name is not found in the config: %s" % name) from None
        return value

    def assign(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a pipeline config """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        config = pipeline.config or {}
        config[name] = value


class F(NamedExpression):
    """ A function, method or any other callable that takes a batch or a pipeline and possibly other arguments

    Examples
    --------
    ::

        F(MyBatch.rotate, angle=30)
        F(prepare_data, 115, item=10)
    """
    def __init__(self, name, *args, mode='w', _pass=True, **kwargs):
        super().__init__(name, mode)
        self.args = args
        self.kwargs = kwargs
        self._pass = _pass

    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value from a callable """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        args = []
        if self._pass:
            if isinstance(batch, _DummyBatch) or batch is None:
                _pipeline = batch.pipeline if batch is not None else pipeline
                args += [_pipeline]
            else:
                args += [batch]
            if model is not None:
                args += [model]
        fargs = eval_expr(self.args, batch=batch, pipeline=pipeline, model=model)
        fkwargs = eval_expr(self.kwargs, batch=batch, pipeline=pipeline, model=model)
        return name(*args, *fargs, **fkwargs)

    def assign(self, *args, **kwargs):
        """ Assign a value by calling a callable """
        _ = args, kwargs
        raise NotImplementedError("Assigning a value with a callable is not supported")

class L(F):
    """ A function, method or any other callable """
    def __init__(self, name, *args, mode='w', **kwargs):
        super().__init__(name, *args, mode=mode, _pass=False, **kwargs)


class V(NamedExpression):
    """ Pipeline variable name

    Examples
    --------
    ::

        V('model_name')
        V('loss_history')
    """
    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a pipeline variable """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        value = pipeline.get_variable(name)
        return value

    def assign(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a pipeline variable """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        pipeline.assign_variable(name, value, batch=batch)

class D(NamedExpression):
    """ Dataset attribute

    Examples
    --------
    ::

        D('classes')
        D('organization')
    """
    def _get_name_dataset(self, batch=None, pipeline=None, model=None):
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline or pipeline
        dataset = pipeline.dataset if pipeline is not None else None
        dataset = dataset or batch.dataset
        if dataset is None:
            raise ValueError("Dataset is not set", self)
        return name, dataset

    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a dataset attribute """
        name, dataset = self._get_name_dataset(batch=batch, pipeline=pipeline, model=model)
        if hasattr(dataset, name):
            value = getattr(dataset, name)
        else:
            raise KeyError("Attribute does not exist in the dataset", name)
        return value

    def assign(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a dataset attribute """
        name, dataset = self._get_name_dataset(batch=batch, pipeline=pipeline, model=model)
        setattr(dataset, name, value)


class R(NamedExpression):
    """ A random value

    Notes
    -----
    If `size` is needed, it should be specified as a named, not a positional argument.

    Examples
    --------
    ::

        R('normal', 0, 1)
        R('poisson', lam=5.5, seed=42, size=3)
        R(['metro', 'taxi', 'bike'], p=[.6, .1, .3], size=10)
    """
    def __init__(self, name, *args, state=None, seed=None, size=None, **kwargs):
        if not (callable(name) or isinstance(name, (str, NamedExpression))):
            args = (name,) + args
            name = 'choice'
        super().__init__(name)
        if isinstance(state, np.random.RandomState):
            self.random_state = state
        else:
            self.random_state = np.random.RandomState(seed)
        self.args = args
        self.kwargs = kwargs
        self.size = size

    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a random variable """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        if callable(name):
            pass
        elif isinstance(name, str) and hasattr(self.random_state, name):
            name = getattr(self.random_state, name)
        else:
            raise TypeError('Random distribution should be a callable or a numpy distribution')
        args = eval_expr(self.args, batch=batch, pipeline=pipeline, model=model)
        if self.size is not None:
            self.kwargs['size'] = self.size
        kwargs = eval_expr(self.kwargs, batch=batch, pipeline=pipeline, model=model)

        return name(*args, **kwargs)

    def assign(self, *args, **kwargs):
        """ Assign a value """
        _ = args, kwargs
        raise NotImplementedError("Assigning a value to a random variable is not supported")

    def __repr__(self):
        repr_str = 'R(' + str(self.name) + ', ' + str(self.args) + ', ' + str(self.kwargs)
        return repr_str + ', size=' + str(self.size) + ')' if self.size else ')'


class P(W):
    """ A wrapper for parallel actions

    Examples
    --------
    Each image in the batch will be rotated at its own angle::

        pipeline
            .rotate(angle=P(R('normal', 0, 1)))

    Without ``P`` all images in the batch will be rotated at the same angle,
    as an angle randomized across batches only::

        pipeline
            .rotate(angle=R('normal', 0, 1))

    Generate 3 categorical random samples for each batch item::

        pipeline
            .calc_route(P(R(['metro', 'taxi', 'bike'], p=[.6, 0.1, 0.3], size=3))

    Notes
    -----
    As P-wrapper is often used for ``R``-expressions, ``R`` can be omitted for brevity.
    So ``P('normal', 0, 1))`` is equivalent to ``P(R('normal', 0, 1)))``, but a bit shorter.
    """
    def __init__(self, name, *args, **kwargs):
        if not isinstance(name, NamedExpression):
            name = R(name, *args, **kwargs)
        if isinstance(name, R):
            if name.size is None:
                name.size = B('size')
            elif isinstance(name.size, int):
                name.size = B('size'), name.size
            else:
                name.size = (B('size'),) + tuple(name.size)
        super().__init__(name)

    def get(self, batch=None, pipeline=None, model=None, parallel=False):   # pylint:disable=arguments-differ
        """ Return a wrapped named expression """
        if parallel:
            return self.name.get(batch=batch, pipeline=pipeline, model=model)
        return self
