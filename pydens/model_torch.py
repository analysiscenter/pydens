""" Contains classes for solving differential equations with neural networks. """

from abc import ABC, abstractmethod
from contextvars import ContextVar, copy_context

import numpy as np
import torch
from torch import nn
from torch.autograd import grad
from tqdm import tqdm

from batchflow.models.torch import Block # pylint: disable=import-error


current_model = ContextVar("current_model")

class TorchModel(ABC, nn.Module):
    """ Pytorch model for solving differential equations with neural networks. """
    def __init__(self, ndims, initial_condition=None, boundary_condition=None, domain=(0, 1), nparams=0, **kwargs):
        _ = kwargs
        super().__init__()

        # Store the number of variables and parameters - dimensionality of the problem.
        self.ndims = ndims
        self.ndims_spatial = ndims if initial_condition is None else ndims - 1
        self.nparams = nparams
        self.total = ndims + nparams
        self.variables = {}

        # Parse and store initial condition, boundary condition and domain of the problem.
        self.initial_condition = initial_condition
        self.boundary_condition = boundary_condition
        if isinstance(domain, (tuple, list)):
            if isinstance(domain[0], (float, int)):
                domain = [domain] * ndims
            elif isinstance(domain[0], (tuple, list)):
                pass
            else:
                raise ValueError('Should be either 1d or 2d-sequence of float/ints.')
        else:
            raise ValueError('Should be either 1d or 2d-sequence of float/ints.')
        self.domain = domain

        # Initialize trainable variables for anzatc-trasform to bind initial
        # and boundary conditions.
        self.log_scale = nn.Parameter(torch.tensor(0.0, requires_grad=True))

    @abstractmethod
    def forward(self, xs):
        """ Forward of the model-network. """

    def freeze_trainable(self, layers=None, variables=None):
        """ Freeze layers and trainable variables.

        Parameters
        ----------
        layers : tuple or list
            Contains names of layers that need not be trained during the run of `Solver.fit`.
            In other words, frozen.
        variables : tuple or list
            Contains names of variables that need to be frozen.

            Examples:

            - ``variables=['log_scale'] # Freezes trainable multiplier that is included in the anzatc.``
        """
        layers = layers or []
        variables = variables or []

        # Freeze layers.
        for layer in layers:
            for param in getattr(self, layer).parameters():
                param.requires_grad = False

        # Freeze variables.
        for variable in variables:
            param = getattr(self, variable)
            param.requires_grad = False

    def unfreeze_trainable(self, layers=None, variables=None):
        """ Unfreeze layers and trainable variables. Reverses the effect of method `TorchModel.freeze_trainable`.

        Parameters
        ----------
        layers : tuple or list
            Names of layers that will be made trainable again.
        variables : tuple or list
            Names of variables that will be made trainable again.
        """
        layers = layers or []
        variables = variables or []

        # Unfreeze layers.
        for layer in layers:
            for param in getattr(self, layer).parameters():
                param.requires_grad = True

        # Unfreeze variables.
        for variable in variables:
            param = getattr(self, variable)
            param.requires_grad = True

    def anzatc(self, u, xs):
        """ Anzatc-transformation of the model-output needed for binding initial and boundary conditions. """
        # Get tensor of spatial variables and time-tensor.
        xs_spatial = xs[:, :self.ndims_spatial]
        t = xs[:, self.ndims - 1:self.ndims]
        lower, upper = [lims[0] for lims in self.domain], [lims[1] for lims in self.domain]
        lower_spatial, upper_spatial = [torch.Tensor(lst[:self.ndims_spatial]).reshape(1, -1).float().to(xs.device)
                                        for lst in (lower, upper)]
        t0 = lower[-1]

        # Apply transformation to bind the boundary condition.
        if self.boundary_condition is not None:
            u = (u * (torch.prod((xs_spatial - lower_spatial) / (upper_spatial - lower_spatial), dim=1, keepdim=True) *
                      torch.prod((upper_spatial - xs_spatial) / (upper_spatial - lower_spatial), dim=1, keepdim=True))
                     + self.boundary_condition)

        # Apply transformation to bind the initial condition.
        if self.initial_condition is not None:
            _xs_spatial = [xs_spatial[:, i] for i in range(xs_spatial.shape[1])]
            u = ((nn.Sigmoid()((t - t0) / torch.exp(self.log_scale)) - .5) * u
                 + self.initial_condition(*_xs_spatial).view(-1, 1))
        return u

class ConvBlockModel(TorchModel):
    """ Model that can create large family of neural networks in a line of code.
    The class is based on "`convolutional block from batchflow <https://github.com/analysiscenter/batchflow>`_".

    The class allows to easily implement fully connected neural networks as well as convolutional
    neural networks with many branches and skip connections. For purposes of solving simple
    differential equations we suggest using fully connected networks with skip connections.

    See also docstring of `Solver`.

    Parameters
    ----------
    layout : str
        String defining the sequence of layers - for instance, 'fa R fa + f'.
        Letter 'f' stands for fully connected layer while letter 'c' - for convolutional layer;
        'a' inserts activation, 'R' defines the start of the skip connection, while '+' shows
        where the skip ends through sum-operation.
    features : sequence
        Sequence configuring the amount of units in all dense layers of the architecture,
        if any present in the layout.
    activation : sequence
        Sequence of callables, str, for instance: `[torch.Sin, torch.nn.Sigmoid, 'Sigmoid']`.

    Examples:

        - ``layout, features, activation = 'fa fa f', [5, 10, 1], 'Sigmoid' # Fully-conn with 2 hidden``
        - ``layout, features, activation = 'faR fa fa+ f', [5, 10, 5, 1], 'Sigmoid # Fully-conn with 3 hidden and skip'``
    """
    def __init__(self, ndims, initial_condition=None, boundary_condition=None, domain=(0, 1), nparams=0,
                 layout='fafaf', features=(20, 30, 1), activation='Sigmoid', **kwargs):
        super().__init__(ndims=ndims, initial_condition=initial_condition, boundary_condition=boundary_condition,
                         domain=domain, nparams=nparams, **kwargs)

        # Prepare kwargs for conv-block.
        kwargs.update(layout=layout, features=list(features), activation=activation)

        # Assemble conv-block.
        fake_inputs = torch.rand((2, self.total), dtype=torch.float32)
        self.conv_block = Block(inputs=fake_inputs, **kwargs)

    def forward(self, xs):
        u = self.conv_block(xs)
        return self.anzatc(u, xs)

def D(y, x):
    """ Differentiation token.
    """
    res = grad(y.sum(), x, retain_graph=True, create_graph=True)[0]
    return res

def V(name, *args, **kwargs):
    """ Token for a trainable variable.
    """
    # If the variable does not exist yet - create it and register in the model.
    # Alternatively, fetch the variable from the model - if already created.
    model = current_model.get()
    if not hasattr(model, name):
        setattr(model, name, nn.Parameter(*args, **kwargs))
    return getattr(model, name)


class FourierNetwork(TorchModel):
    """ ... """
    def __init__(self, ndims, initial_condition=None, boundary_condition=None, domain=(0, 1), nparams=0,
                 layout='faf', features=(20, 30, 1), activation='Sigmoid', fourier_sigma=1, fourier_bias=False,
                 freeze_fourier_features=True, **kwargs):
        super().__init__(ndims=ndims, initial_condition=initial_condition, boundary_condition=boundary_condition,
                         domain=domain, nparams=nparams, **kwargs)

        # Prepare kwargs for conv-block.
        kwargs.update(layout=layout, features=list(features[1:]), activation=activation)

        # Set up encoding block
        self.encoding = nn.Linear(self.total, features[0], bias=fourier_bias)
        if freeze_fourier_features:
            self.encoding.requires_grad_(False)
        if fourier_sigma is not None:
            nn.init.normal_(self.encoding.weight, std=fourier_sigma)

        # Assemble conv-block.
        fake_inputs = torch.rand((2, 2 * features[0]), dtype=torch.float32) # sin and cos are concatted -> take 2 * features[0]
        self.conv_block = Block(inputs=fake_inputs, **kwargs)

    def forward(self, xs):
        # make encoding
        enc = [activation(2 * torch.pi * self.encoding(xs))
               for activation in (torch.sin, torch.cos)]
        enc = torch.cat(enc, dim=1)

        # apply conv block
        u = self.conv_block(enc)
        return self.anzatc(u, xs)

class MultiscaleFourierNetwork(TorchModel):
    """ ... """
    def __init__(self, ndims, initial_condition=None, boundary_condition=None, domain=(0, 1), nparams=0, nscales=5,
                 layout='faf', features=(20, 30, 1), activation='Sigmoid', fourier_sigma=1, fourier_bias=False,
                 freeze_fourier_features=True, **kwargs):
        super().__init__(ndims=ndims, initial_condition=initial_condition, boundary_condition=boundary_condition,
                         domain=domain, nparams=nparams, **kwargs)
        #
        if isinstance(fourier_sigma, (int, float)):
            fourier_sigma = (fourier_sigma, ) * nscales

        # Prepare kwargs for conv-block.
        kwargs.update(layout=layout.replace(' ', '')[:-1], features=list(features[1:-1]), activation=activation)

        # Set up encoding block
        self.encodings = nn.ModuleList([nn.Linear(self.total, features[0], bias=fourier_bias) for _ in range(nscales)])
        for i, encoding in enumerate(self.encodings):
            if fourier_sigma is not None:
                nn.init.normal_(encoding.weight, std=fourier_sigma[i])
            if freeze_fourier_features:
                encoding.requires_grad_(False)

        # Assemble conv-block.
        fake_inputs = torch.rand((2, 2 * features[0]), dtype=torch.float32) # sin and cos are concatted -> take 2 * features[0]
        self.conv_block = Block(inputs=fake_inputs, **kwargs)

        # Last linear layer stacked outputs -> single output
        self.last_linear = nn.Linear(nscales * features[-2], features[-1])

    def forward(self, xs):
        # make encoding
        encodings = [
            [activation(2 * torch.pi * encoding(xs)) for activation in (torch.sin, torch.cos)]
            for encoding in self.encodings
        ]
        encodings = [torch.cat(enc, dim=1) for enc in encodings]

        # apply conv block
        us = [self.conv_block(enc) for enc in encodings]
        us = torch.cat(us, dim=-1)

        #
        u = self.last_linear(us)

        return self.anzatc(u, xs)

class SpatialTemporalFourierNetwork(TorchModel):
    """ ... """
    def __init__(self, ndims, initial_condition=None, boundary_condition=None, domain=(0, 1), nparams=0, nscales=(5, 3),
                 layout='faf', features=(20, 30, 1), activation='Sigmoid', fourier_sigma=1, fourier_bias=False,
                 freeze_fourier_features=True, **kwargs):
        super().__init__(ndims=ndims, initial_condition=initial_condition, boundary_condition=boundary_condition,
                         domain=domain, nparams=nparams, **kwargs)
        self.nscales = nscales
        #
        if isinstance(fourier_sigma, (int, float)):
            fourier_sigma = (fourier_sigma, ) * nscales[0] + (fourier_sigma, ) * nscales[1]

        # Prepare kwargs for conv-block.
        kwargs.update(layout=layout.replace(' ', '')[:-1], features=list(features[1:-1]), activation=activation)

        # Set up encoding block
        self.encodings = (
            nn.ModuleList([nn.Linear(self.total - 1, features[0], bias=fourier_bias) for _ in range(nscales[0])]) + # spatial encoding matrices
            nn.ModuleList([nn.Linear(1, features[0], bias=fourier_bias) for _ in range(nscales[1])])  # temporal encoding matrices
        )
        for i, encoding in enumerate(self.encodings):
            if fourier_sigma is not None:
                nn.init.normal_(encoding.weight, std=fourier_sigma[i])
            if freeze_fourier_features:
                encoding.requires_grad_(False)

        # Assemble conv-block.
        fake_inputs = torch.rand((2, 2 * features[0]), dtype=torch.float32) # sin and cos are concatted -> take 2 * features[0]
        self.conv_block = Block(inputs=fake_inputs, **kwargs)

        # Last linear layer stacked outputs -> single output
        self.last_linear = nn.Linear(nscales[0] * nscales[1] * features[-2], features[-1])

    def forward(self, xs):
        # create spatial and temporal tensors
        xs_spatial = torch.cat([xs[:, :self.ndims_spatial], xs[:, self.ndims:]], dim=1)
        t = xs[:, self.ndims - 1:self.ndims]
        coordinates = (xs_spatial, ) * self.nscales[0] + (t, ) * self.nscales[1]

        # make encoding
        encodings = [
            [activation(2 * torch.pi * encoding(coords)) for activation in (torch.sin, torch.cos)]
            for encoding, coords in zip(self.encodings, coordinates)
        ]
        encodings = [torch.cat(enc, dim=1) for enc in encodings]

        # apply conv block
        hs = [self.conv_block(enc) for enc in encodings]
        multiplications = []
        for i in range(self.nscales[0]):
            for j in range(self.nscales[1]):
                multiplications.append(hs[i] * hs[self.nscales[0] + j])

        us = torch.cat(multiplications, dim=-1)

        #
        u = self.last_linear(us)

        return self.anzatc(u, xs)

class Solver():
    r""" Solver of differential equations with neural networks. Allows to solve wide variety of
    differential equations including (i) common ODEs and PDEs (ii) parametric families of equations
    (iii) inverse partial differential equations.

    Based on Sirignano J., Spiliopoulos K. "`DGM: A deep learning algorithm for solving partial
    differential equations <https://arxiv.org/abs/1708.07469>`_".

    Parameters
    ----------
    equation : callable
        Callable that uses tokens `D`(differentiation operation), `V`(trainable variable) and
        common mathematical operations from `torch` to setup a PDE-problem.

        Examples:

        - ``lambda f, x: D(f, x) + torch.log(x)``
        - .. code-block:: python

            def ode(f, x):
                return D(f, x) - 2 * np.pi * torch.cos(2 * np.pi * x)

        - .. code-block:: python

            def pde(f, x, y):
                return D(D(f, x), x) + D(D(f, y), y) - torch.sin(np.pi * (x + y))

        - .. code-block:: python

            def pde(f, x, e):
                return D(f, x) - e * np.pi * torch.cos(e * np.pi * x)

            Note that the last example doesn't include differentiation with respect
            to `e`. This means that `e` is a parameter with uncertainty rather than a variable.
            So, the value of `nparams`-parameter should be set to 1, whie the value of
            `ndims` - to 2.

    ndims : int
        The dimensionality of the problem. Equals the number of variables. For instance,
        when the problem includes `x`, `y` and `t`, the dimensionality is equal to 3.

    initial_condition : callable or float
        Defines initial condition for a problem that includes time-variable.

        Examples:

        - ``lambda x: x * (1 - x)``
            Can define initial condition for a wave equation for a pertubed string with `x` and `t`
            - variables. The initial condition is basically a perturbation of a string at `t=0`.
        - ``lambda x, y: 10 * x * y * (1 - x) * (1 - y)``
            Can define initial condition for a heat equation describing temperature evolution of a
            2d plate.

    boundary_condition : float
        Defines boundary condition for the problem.

    domain : tuple or list
        Configures rectangular domain of the problem. Can be either a 2d-sequence of form
        `[(x_low, x_high), (y_low, y_high), ..., (t_low, t_high)]` of `length=ndims`; or
        a 1d-sequence `(low, high)`. In this case, limits for all dimensions are considered
        to be the same.

    nparams : int
        The number of parameters with uncertainty in the model.

    model : class
        Class inheriting `TorchModel`. The default value is `ConvBlockModel`. The class allows
        to implement fully connected/convolutional architectures with multiple branches
        and skip-connections. All of the `kwargs` supplied into `Solver`-initialization
        go into the `model`-class.

        Note:
        `ConvBlockModel` is based on `Block` from framework
        "`BatchFlow <https://github.com/analysiscenter/batchflow>`_".

    device : str or torch.device
        ..............

    constraints : sequence or callable
        Either sequence of callables or one callable. Each callable is an additional constraint
        that can be used during `fit`-run to form loss function.

        Examples:

        - ``constraints=lambda f, x: f(0.5) + 1``
            This constraints allows to bind `f(0.5) = -1` for a simple ODE.
        - ``constraints=(lambda f, x: f(0.5) + 1, lambda f, x: f(x)**2)``
            This set of additional constraints can be used to (i) bind `f(0.5) = -1`
            along with minimization of a L2-norm of `f`.

    kwargs : dict
        Keyword-arguments used for initialization of the model-instance. When `model`-parameter
        is set to its default value - `ConvBlockModel`, use these arguments to configure the
        architecture:

        layout : str
            String defining the sequence of layers - for instance, 'fa R fa + f'.
            Letter 'f' stands for fully connected layer while letter 'c' - for convolutional layer;
            'a' inserts activation, 'R' defines the start of the skip connection, while '+' shows
            where the skip ends through sum-operation.
        features : sequence
            Sequence configuring the amount of units in all dense layers of the architecture,
            if any present in layout.
        activation : sequence
            Sequence of callables, str, for instance: `[torch.Sin, torch.nn.Sigmoid, 'Sigmoid']`.

        Examples:

            - ``layout, features, activation = 'fa fa f', [5, 10, 1], 'Sigmoid'``
            - ``layout, features, activation = 'faR fa fa+ f', [5, 10, 5, 1], 'Sigmoid'``
    """
    def __init__(self, equation, ndims, initial_condition=None, boundary_condition=None, domain=(0, 1),
                 nparams=0, model=ConvBlockModel, device='cpu', constraints=None, **kwargs):
        self.equation = equation
        if constraints is None:
            self.constraints = ()
        elif isinstance(constraints, (tuple, list)):
            self.constraints = constraints
        else:
            self.constraints = (constraints, )
        self.losses = []
        self.optimizer = None
        self.device = device

        # Initialize neural network for solving the equation.
        if initial_condition is not None:
            if not callable(initial_condition):
                value = initial_condition
                initial_condition = lambda *args: torch.tensor(value, dtype=torch.float32).to(device)
        self.model = model(**kwargs, ndims=ndims, initial_condition=initial_condition,
                           boundary_condition=boundary_condition, domain=domain, nparams=nparams).to(device)

        # Bind created model to a global context variable.
        current_model.set(self.model)
        self.ctx = copy_context()

        # Perform fake run of the model to create all the variables (coming from V-token).
        xs = [torch.rand((1, 1)).to(device) for _ in range(self.model.total)]
        for x in xs:
            x.requires_grad_()
        xs_concat = self.reshape_and_concat(xs)
        u_hat = self.model(xs_concat)
        _ = self.ctx.run(self.equation, u_hat, *xs)

    def reshape_and_concat(self, tensors):
        """ Cast, reshape and concatenate sequence of incoming tensors. Returns `torch.Tensor`
        of size (N X D).

        Parameters
        ----------
        tensors : sequence
            Sequence of elements. Each element can be an numpy-array, `torch.Tensor` or a number.
            The function either casts or tiles and casts each array to a `torch.Tensor` of shape (N X 1).

        Returns
        -------
        torch.Tensor
            Tensor of shape (N X D), where D is defined as maximum length of tensor in the incoming
            sequence.
        """
        # Determine batch size as max-len of a tensor in included in the sequence.
        xs = list(tensors)
        sizes = ([np.prod(tensor.shape) for tensor in xs if isinstance(tensor, (np.ndarray, torch.Tensor))] +
                 [np.prod(np.array(tensor).shape) for tensor in xs if isinstance(tensor, (tuple, list))])
        batch_size = np.max(sizes) if len(sizes) > 0 else 1

        # Perform cast and reshape of all tensors in the list.
        for i, x in enumerate(xs):
            if isinstance(x, (int, float)):
                xs[i] = torch.Tensor(np.tile(x, (batch_size, 1))).float().to(self.device)
            if isinstance(x, np.ndarray):
                if x.size != batch_size:
                    x = np.tile(x.squeeze()[0], (batch_size, 1))
                xs[i] = torch.Tensor(x.reshape(batch_size, 1)).float().to(self.device)
            if isinstance(x, (list, tuple)):
                xs[i] = torch.Tensor(x).float().to(self.device).view(-1, 1)
            if isinstance(x, torch.Tensor):
                xs[i] = x.view(-1, 1)
        return torch.cat(xs, dim=1)

    def fit(self, niters, batch_size, sampler=None, loss_terms='equation', optimizer='Adam',
            criterion=nn.MSELoss(), lr=0.005, **kwargs):
        """ Perform fit procedure. Trains the model attributed to the `Solver`-instance.

        Parameters
        ----------
        niters : int
            Number of iterations of optimization.

        batch_size : int
            The number of points sampled for each iteration.

        sampler : Sampler or None
            An object that generates batches of points for the training prosedure.
            Shoild contain `sample`-method that accepts `size`-argument. In its turm, the method `sample`
            should produce the requested number of nd-points. Usually, `sampler`-object is an instance
            of class `Sampler` from framework "`BatchFlow <https://github.com/analysiscenter/batchflow>`_".

        loss_terms : str/sequence of str
            Defines loss terms used for constructing total loss-function. Each item from the sequence can be
            equal to:
            - 'equation' - the term penalizes the difference between l.h.s and r.h.s of the equation. The default value
                of the parameter. Should be used in all cases, where one runs `fit` to better solve the equation.
            - 'constraint_{k}' - the value 'constraint_0' adds the first additional constraint to the total loss,
                while 'constraint_1' penalizes the second etc.; whenever used, `constraints`-argument should be
                supplied during the `Solver`-initialization.

        optimizer : str
            Name of an optimizer from `torch.optim`. If set to `None`, uses already existing optimizer.
            Whenever supplied, creates a new optimizer-instance.
            See "`Docstring of torch.optim <https://pytorch.org/docs/stable/optim.html>`_".

        criterion : callable
            Criterion-function. Accepts two arguments; is used to form terms of the total
            loss-function. Frequently is a method from `torch.nn`.

            Examples:

            - .. code-block:: python

                kwargs = {criterion=torch.nn.MSELoss(),
                          loss_terms=('equation', 'constraint_0')}
                solver.fit(**kwargs)

                # Yields loss given by
                # [MSELoss(l.h.s of the equation - r.h.s of the equation) + MSELoss(constraint_0(f, x, ...))].
                # In other words, this loss penalizes both requested terms using supplied criterion.

        lr : float
            Starting learning rate for the optimizer-initialization.

        kwargs : dict
            Other keyword-arguments for the the optimizer-initialization.
        """
        # Initialize the optimizer if supplied.
        if optimizer is not None:
            self.optimizer = getattr(torch.optim, optimizer)([p for p in self.model.parameters()
                                                              if p.requires_grad],
                                                              lr=lr, **kwargs)

        loss_terms = loss_terms if isinstance(loss_terms, (tuple, list)) else (loss_terms, )
        nums_constraints = [int(term_name.replace('constraint', '').replace('_', ''))
                            for term_name in loss_terms if 'constraint' in term_name]

        # Perform `niters`-iterations of optimizer steps.
        self.model.train()
        for _ in tqdm(range(niters)):
            if optimizer == "LBFGS":
                def closure():
                    self.optimizer.zero_grad()
                    # Sample batch of points and compute solution-approximation on the batch.
                    if sampler is None:
                        xs = [torch.rand((batch_size, 1)).to(self.device) for _ in range(self.model.total)]
                    else:
                        xs_array = sampler.sample(batch_size).astype(np.float32)
                        xs = [torch.from_numpy(xs_array[:, i:i+1]).to(self.device) for i in range(xs_array.shape[1])]
                    for x in xs:
                        x.requires_grad_()
                    xs_concat = self.reshape_and_concat(xs)
                    u_hat = self.ctx.run(self.model, xs_concat)
                    # Compute loss: form it summing equation-loss and constraints-loss.
                    loss  = 0

                    # Include equation loss.
                    if 'equation' in loss_terms:
                        loss += criterion(self.ctx.run(self.equation, u_hat, *xs), torch.zeros_like(xs[0]).to(self.device))

                    # Include additional constraints' loss.
                    def _forward(*xs):
                        """ Concat and apply the model. """
                        xs = self.reshape_and_concat(xs)
                        return self.model(xs)

                    for num in nums_constraints:
                        loss += criterion(self.ctx.run(self.constraints[num], _forward, *xs), torch.Tensor([0.0]))

                    # Optimizer step.
                    loss.backward()
                    return loss
                self.optimizer.step(closure)
                loss = closure()
            else:
                self.optimizer.zero_grad()

                # Sample batch of points and compute solution-approximation on the batch.
                if sampler is None:
                    xs = [torch.rand((batch_size, 1)).to(self.device) for _ in range(self.model.total)]
                else:
                    xs_array = sampler.sample(batch_size).astype(np.float32)
                    xs = [torch.from_numpy(xs_array[:, i:i+1]).to(self.device) for i in range(xs_array.shape[1])]
                for x in xs:
                    x.requires_grad_()
                xs_concat = self.reshape_and_concat(xs)
                u_hat = self.ctx.run(self.model, xs_concat)

                # Compute loss: form it summing equation-loss and constraints-loss.
                # loss_terms = loss_terms if isinstance(loss_terms, (tuple, list)) else (loss_terms, )
                nums_constraints = [int(term_name.replace('constraint', '').replace('_', ''))
                                    for term_name in loss_terms if 'constraint' in term_name]
                loss  = 0

                # Include equation loss.
                if 'equation' in loss_terms:
                    loss += criterion(self.ctx.run(self.equation, u_hat, *xs), torch.zeros_like(xs[0]).to(self.device))

                # Include additional constraints' loss.
                def _forward(*xs):
                    """ Concat and apply the model. """
                    xs = self.reshape_and_concat(xs)
                    return self.model(xs)

                for num in nums_constraints:
                    loss += criterion(self.ctx.run(self.constraints[num], _forward, *xs), torch.Tensor([0.0]))

                # Optimizer step.
                loss.backward()
                self.optimizer.step()

            # Gather and store training stats.
            self.losses.append(loss.detach().cpu().numpy())

    def predict(self, *xs):
        """ Get predictions of the solution to the problem in supplied points `*xs`. Each
        `x` is a tensor/array/number that the method will either cast (or cast and tile) to a (1 X D) tensor.

        Parameters
        ----------
        tensors : sequence
            Sequence of elements. Each element can be an numpy-array, `torch.Tensor` or a number.
            The function either casts (or tiles and casts) each array to a `torch.Tensor` of shape (N X 1).

        Returns
        -------
        np.ndarray
            Array containing solution-prediction.
        """
        # Reshape and concat.
        xs = self.reshape_and_concat(xs)

        # Perform inference.
        self.model.eval()
        result = self.ctx.run(self.model, xs)
        return result.detach().cpu().numpy()
