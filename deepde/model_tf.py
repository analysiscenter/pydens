""" TensorFlow implementation of model for solving partial differential equations. Inspired by
Sirignano J., Spiliopoulos K. "`DGM: A deep learning algorithm for solving partial differential equations
<http://arxiv.org/abs/1708.07469>`_"
"""
from inspect import signature

import tensorflow as tf

#pylint: disable=no-name-in-module, import-error
from .batchflow.models.tf import TFModel
from .batchflow.models.tf.layers import conv_block
from .syntax import get_num_parameters



class TFDeep(TFModel):
    r"""TensorFlow model for solving partial differential equations (PDEs) of up to the second order
    on rectangular domains using neural networks.

    In addition, allows to solve random (parametric) PDEs in strong form as discussed in Nabian M.A., Meidani H.
    "`A Deep Neural Network Surrogate for High-Dimensional Random Partial Differential Equations
    <https://arxiv.org/abs/1806.02957>`_"

    **Configuration**

    Inherited from :class:`.TFModel`. Supports all config options from  :class:`.TFModel`,
    including the choice of `device`, `session`, `inputs`-configuration, `loss`-function . Also
    allows to set up the network-architecture using options `initial_block`, `body`, `head`. See
    docstring of :class:`.TFModel` for more detail.

    Left-hand-side (lhs), domain and other properties of PDE are defined in `pde`-dict:

    pde : dict
        dictionary of parameters of PDE. Must contain keys
        - form : callable
            defines diferential form in lhs of the PDE. Composed from predefined tokens including
            differential operator `D(u, x)` and unary operations like `sin` and `cos`. Can also
            include coefficients R(e) to make the whole equation a parametric family of equations
            rather than a simple PDE.
        - domain : list
            defines the rectangular domain of the equation as a sequence of coordinate-wise bounds.
        - bind_bc_ic : bool
            If True, modifies the network-output to bind boundary and initial conditions.
        - initial_condition : callable or const or None or list
            If supplied, defines the initial state of the system as a function of
            spatial coordinates (and, possibly, parametric coefficients R(e)). In that case, PDE
            is considered to be an evolution equation (heat-equation or wave-equation, e.g.). Then,
            first (n - 1) coordinates are spatial, while the last one is the time-variable. If the
            lhs of PDE contains second-order derivative w.r.t time, initial evolution-rate of the
            system must also be supplied. In this case, the arg is a `list` with two callables
            (constants). Also written using the set of predefined tokens.
        - time_multiplier : str or callable
            Can be either 'sigmoid', 'polynomial' or callable. Needed if `initial_condition`
            is supplied. Defines the multipliers applied to network for binding initial conditions.
            `sigmoid` works better in problems with asymptotic steady states (heat equation, e.g.).

    track : dict
        allows for logging of differentials of the solution-approximator. Can be used for
        keeping track on the model-training process.

    Examples
    --------

        config = dict(
            pde = dict(
                form=lambda u, x, t: D(u, t) - D(D(u, x), x) - 5,
                initial_condition=lambda t: sin(2 * np.pi * t),
                domain=[[0, 1], [0, 3]],
                time_multiplier='sigmoid'),
            track=dict(dt=lambda u, x, t: D(u, t)))

        stands for PDE given by
            \begin{multline}
                \frac{\partial f}{\partial t} - \frac{\partial^2 f}{\partial x^2} = 5, \\
                f(x, 0) = \sin(2 \pi x), \\
                \Omega = [0, 1] \times [0, 3], \\
                f(0, t) = 0 = f(1, t).
            \end{multline}
        while the solution to the equation is searched in the form
            \begin{equation}
                f(x, t) = (\sigma(x / w) - 0.5) * network(x, t) + \sin(x).
            \end{equation}
        We also track
            $$ \frac{\partial f}{\partial t} $$
    """
    @classmethod
    def default_config(cls):
        """ Overloads :meth:`.TFModel.default_config`. """
        config = super().default_config()
        config['ansatz'] = {}
        config['common/time_multiplier'] = 'sigmoid'
        config['common/bind_bc_ic'] = True
        return config

    def build_config(self, names=None):
        """ Overloads :meth:`.TFModel.build_config`.
        PDE-problem is fetched from 'pde' key in 'self.config', and then
        is passed to 'common' so that all of the subsequent blocks get it as 'kwargs'.
        """
        pde = self.config.get('pde')
        if pde is None:
            raise ValueError("The PDE-problem is not specified. Use 'pde' config to set up the problem.")

        # Get the dimensionality
        n_dims = pde.get('n_dims')
        n_funs = pde.get('n_funs', 1)
        n_eqns = pde.get('n_eqns', n_funs)

        # Make sure that `form` describes necessary number of equations
        form = pde.get('form')
        form = form if isinstance(form, (tuple, list)) else [form]
        assert len(form) == n_eqns
        pde.update({'form': form})

        # Count unique usages of `P`
        n_parameters = get_num_parameters(form[0])

        # Convert each expression to track to list
        track = pde.get('track')
        if track:
            track = {value if isinstance(value, (tuple, list)) else [value]
                     for value in track.values()}
            pde.update({'track': track})

        # Make sure that PDE dimensionality is consistent
        n_args = len(signature(form[0]).parameters)
        assert n_dims + n_parameters + n_funs == n_args
        pde.update({'n_dims': n_dims,
                    'n_funs': n_funs,
                    'n_eqns': n_eqns,
                    'n_parameters': n_parameters,
                    'n_vars': n_dims + n_parameters})

        # Make sure points-tensor is created
        self.config.update({'initial_block/inputs': 'points',
                            'inputs': dict(points={'shape': (n_dims + n_parameters, )})})

        # Default values for domain
        if pde.get('domain') is None:
            self.config.update({'pde/domain': [[0, 1]] * n_dims})

        # Make sure that initial conditions are callable
        init_conds = pde.get('initial_condition', None)
        if init_conds is not None:
            init_conds = self._make_nested_list(init_conds, n_funs, 'initial')
            self.config.update({'pde/initial_condition': init_conds})

        # make sure that boundary condition is callable
        bound_cond = pde.get('boundary_condition', [0.0]*n_funs)
        bound_cond = self._make_nested_list(bound_cond, n_funs, 'boundary')
        self.config.update({'pde/boundary_condition': bound_cond})

        # 'common' is updated with PDE-problem
        config = super().build_config(names)
        config['common'].update(self.config['pde'])

        config = self._make_ops(config)
        config['ansatz/coordinates'] = self.get_from_attr('coordinates')
        return config

    def _make_nested_list(self, list_cond, n_funs, name=None):
        if n_funs == 1:
            list_cond = list_cond if isinstance(list_cond, (tuple, list)) else [list_cond]
            list_cond = [list_cond]
        else:
            if isinstance(list_cond, (tuple, list)):
                if not isinstance(list_cond[0], (tuple, list)):
                    list_cond = [[cond] for cond in list_cond]
            else:
                raise ValueError('Multiple functions must have multiple {} conditions.'.format(name))
        assert len(list_cond) == n_funs

        results = []
        for i in range(n_funs):
            result = []
            for cond in list_cond[i]:
                if callable(cond):
                    result.append(cond)
                else:
                    result.append(lambda *args, value=cond: value)
            results.append(result)
        return results

    def _make_ops(self, config):
        """ Stores necessary operations in 'config'. """
        # retrieving variables
        ops = config.get('output')
        track = config.get('track')
        coordinates = self.get_from_attr('coordinates')

        # ensuring that 'ops' is of the needed type
        if ops is None:
            ops = []
        elif not isinstance(ops, (dict, tuple, list)):
            ops = [ops]
        if not isinstance(ops, dict):
            ops = {'': ops}
        prefix = list(ops.keys())[0]
        _ops = dict()
        _ops[prefix] = list(ops[prefix])

        # form for output-transformation
        config['predictions'] = self._make_form_calculator(config.get("common/form"), coordinates,
                                                           name='predictions', pde=config['common'])
        # forms for tracking
        if track is not None:
            for op in track.keys():
                _compute_op = self._make_form_calculator(track[op], coordinates, name=op, pde=config['common'])
                _ops[prefix].append(_compute_op)

        config['output'] = _ops
        return config

    def _make_inputs(self, names=None, config=None):
        """ Create necessary placeholders. """
        n_dims = config['pde/n_dims']
        n_parameters = config['pde/n_parameters']
        n_eqns = config['pde/n_eqns']
        placeholders_, tensors_ = super()._make_inputs(names, config)

        # split input so we can access individual variables later
        coordinates = tf.split(tensors_['points'][:, :n_dims], n_dims, axis=1, name='coordinates')
        if n_parameters > 0:
            perturbations = tf.split(tensors_['points'][:, n_dims:], n_parameters, axis=1, name='perturbations')
        else:
            perturbations = []

        tensors_['points'] = tf.concat(coordinates + perturbations, axis=1)
        self.store_to_attr('coordinates', coordinates + perturbations)
        self.store_to_attr('inputs', tensors_)

        # make targets-tensor from zeros
        points = self.get_from_attr('inputs').get('points')
        self.store_to_attr('targets', tf.zeros(shape=(tf.shape(points)[0], n_eqns)))
        return placeholders_, tensors_

    @classmethod
    def _make_form_calculator(cls, form, coordinates, name='_callable', pde=None):
        """ Get callable that computes differential form of a tf.Tensor
        with respect to coordinates.
        """
        n_funs = pde.get('n_funs')
        form = form if isinstance(form, (tuple, list)) else [form]

        # `_callable` should be a function of `net`-tensor only
        def _callable(net):
            net_list = tf.split(net, n_funs, axis=-1, name='net-splitted')

            results = []
            for eqn in form:
                result = eqn(*net_list, *coordinates)
                results.append(result)

            results = tf.reshape(results, shape=(-1, len(form)))
            return results

        setattr(_callable, '__name__', name)
        return _callable

    def _build(self, config=None):
        """ Overloads :meth:`.TFModel._build`: adds ansatz-block for binding
        boundary and initial conditions.
        """
        inputs = config.pop('initial_block/inputs')
        x = self._add_block('initial_block', config, inputs=inputs)
        x = self._add_block('body', config, inputs=x)
        x = self._add_block('head', config, inputs=x)
        output = self._add_block('ansatz', config, inputs=x)
        self.store_to_attr('solution', output)
        self.output(output, predictions=config['predictions'], ops=config['output'], **config['common'])

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Body of the neural network.
        Shared between all of the unknown functions in the PDE."""
        return super().body(inputs, name, **kwargs)

    @classmethod
    def head(cls, inputs, name='head', **kwargs):
        """ Head (final) block of the neural network.
        Makes one individual output branch for each unknown function in the PDE-system.
        """
        n_funs = kwargs.get('n_funs')
        kwargs = cls.fill_params('head', **kwargs)

        if not kwargs.get('layout'):
            return [inputs]

        with tf.variable_scope(name):
            heads = []
            for i in range(n_funs):
                branch = conv_block(inputs, name=('branch-' + str(i)), **kwargs)
                heads.append(branch)
        return heads

    @classmethod
    def ansatz(cls, inputs, coordinates, **kwargs):
        """ Binds `initial_condition` or `boundary_condition`, if these are supplied in the config
        of the model. Does so by:
        1. Applying one of preset multipliers to the network output
           (effectively zeroing it out on boundaries and $t=t_0$)
        2. Adding passed condition, so it is satisfied on boundaries and/or at $t=t_0$.
        Creates a tf.Tensor `solution` - the final output of the model.
        """
        if kwargs["bind_bc_ic"]:
            # Retrieving variables
            n_dims = kwargs['n_dims']
            n_funs = kwargs['n_funs']

            init_cond = kwargs.get("initial_condition")
            bound_cond = kwargs["boundary_condition"]
            domain = kwargs["domain"]
            time_mode = kwargs["time_multiplier"]

            # Separate variables and perturbations
            coordinates = coordinates[:n_dims]
            perturbations = coordinates[n_dims:]

            lower, upper = [[bounds[i] for bounds in domain] for i in range(2)]
            n_dims_xs = n_dims if init_cond is None else n_dims - 1
            xs_spatial = coordinates[:n_dims_xs] if n_dims_xs > 0 else []
            xs_spatial_ = tf.concat(xs_spatial, axis=1) if n_dims_xs > 0 else None
            xs_spatial_es = xs_spatial + perturbations

            # Multiplicator for binding boundary conditions
            binding_multiplier = 1
            if n_dims_xs > 0:
                lower_tf, upper_tf = [tf.constant(bounds[:n_dims_xs], shape=(1, n_dims_xs), dtype=tf.float32)
                                      for bounds in (lower, upper)]
                binding_multiplier *= tf.reduce_prod((xs_spatial_ - lower_tf) * (upper_tf - xs_spatial_) /
                                                     (upper_tf - lower_tf)**2,
                                                     axis=1, name='ansatz/xs_multiplier', keepdims=True)

            # Apply ansatz to each branch of head to obtain solution-approximation for each pde
            solution = []
            for i in range(n_funs):
                add_term = 0
                multiplier = 1
                add_bind = 0

                # Ignore boundary condition as it is automatically set by initial condition
                if init_cond is not None:
                    shifted = coordinates[-1] - tf.constant(lower[-1], shape=(1, 1), dtype=tf.float32)
                    time_mode = kwargs["time_multiplier"]

                    add_term += init_cond[i][0](*xs_spatial_es)
                    multiplier *= cls._make_time_multiplier(time_mode,
                                                            '0' if len(init_cond[i]) == 1 else '00')(shifted)

                    # multiple initial conditions
                    if len(init_cond[i]) > 1:
                        add_term += (init_cond[i][1](*xs_spatial_es)
                                     * cls._make_time_multiplier(time_mode, '01')(shifted))

                # If there are no initial conditions, boundary conditions are used (default value is 0)
                else:
                    add_term += bound_cond[i][0](*xs_spatial_es)

                # Sometimes you need it
                if kwargs.get('do_that_strange_magic'):
                    if n_dims_xs > 0:
                        lower_tf, upper_tf = [tf.constant(bounds[:n_dims_xs],
                                                          shape=(1, n_dims_xs), dtype=tf.float32)
                                              for bounds in (lower, upper)]
                        binding_multiplier *= tf.reduce_prod(((xs_spatial_ - lower_tf)
                                                              * (upper_tf - xs_spatial_)) /
                                                             (upper_tf - lower_tf)**2,
                                                             axis=1, name='ansatz/xs_multiplier', keepdims=True)

                        add_bind = ((bound_cond[i][0](coordinates[-1]) - init_cond[i][0](lower_tf)
                                     / (multiplier + 1e1))
                                    * ((upper_tf - xs_spatial) / (upper_tf - lower_tf)))
                        add_bind = tf.reshape(add_bind, shape=(-1, 1))

                result = add_term + multiplier * (inputs[i]*binding_multiplier + add_bind)
                solution.append(result)
        return tf.concat(solution, axis=-1, name='ansatz/_output')

    @classmethod
    def _make_time_multiplier(cls, family, order=None):
        r""" Produce time multiplier: a callable, applied to an arbitrary function to bind its value
        and, possibly, first order derivataive w.r.t. to time at $t=0$.

        Parameters
        ----------
        family : str or callable
            defines the functional form of the multiplier, can be either `polynomial` or `sigmoid` or generic callable.
        order : str or None
            sets the properties of the multiplier, can be either `0` or `00` or `01`. '0'
            fixes the value of multiplier as $0$ at $t=0$, while '00' sets both value and derivative to $0$.
            In the same manner, '01' sets the value at $t=0$ to $0$ and the derivative to $1$.

        Returns
        -------
        callable

        Examples
        --------
        Form an `solution`-tensor binding the initial value (at $t=0$) of the `network`-tensor to $sin(2 \pi x)$::

            solution = network * TFDeep._make_time_multiplier('sigmoid', '0')(t) + tf.sin(2 * np.pi * x)

        Bind the initial value to $sin(2 \pi x)$ and the initial rate to $cos(2 \pi x)$::

            solution = (network * TFDeep._make_time_multiplier('polynomial', '00')(t) +
                            tf.sin(2 * np.pi * x) +
                            tf.cos(2 * np.pi * x) * TFDeep._make_time_multiplier('polynomial', '01')(t))
        """
        if family == "sigmoid":
            if order == '0':
                def _callable(shifted_time):
                    log_scale = tf.Variable(0.0, name='time_scale')
                    return tf.sigmoid(shifted_time * tf.exp(log_scale)) - 0.5
            elif order == '00':
                def _callable(shifted_time):
                    log_scale = tf.Variable(0.0, name='time_scale')
                    scale = tf.exp(log_scale)
                    return tf.sigmoid(shifted_time * scale) - tf.sigmoid(shifted_time) * scale - 1 / 2 + scale / 2
            elif order == '01':
                def _callable(shifted_time):
                    log_scale = tf.Variable(0.0, name='time_scale')
                    scale = tf.exp(log_scale)
                    return 4 * tf.sigmoid(shifted_time * scale) / scale - 2 / scale
            else:
                raise ValueError("Order " + str(order) + " is not supported.")

        elif family == "polynomial":
            if order == '0':
                def _callable(shifted_time):
                    log_scale = tf.Variable(0.0, name='time_scale')
                    return shifted_time * tf.exp(log_scale)
            elif order == '00':
                def _callable(shifted_time):
                    return shifted_time ** 2 / 2
            elif order == '01':
                def _callable(shifted_time):
                    return shifted_time
            else:
                raise ValueError("Order " + str(order) + " is not supported.")

        elif callable(family):
            _callable = family
        else:
            raise ValueError("'family' should be either 'sigmoid', 'polynomial' or callable.")

        return _callable

    def predict(self, fetches=None, feed_dict=None, **kwargs):
        """ Get network-approximation of PDE-solution on a set of points. Overloads :meth:`.TFModel.predict` :
        `solution`-tensor is now considered to be the main model-output.
        """
        fetches = 'solution' if fetches is None else fetches
        predicted = super().predict(fetches, feed_dict, **kwargs)

        if hasattr(predicted, '__len__'):
            if len(predicted) == 1:
                predicted = predicted[0]
        return predicted
