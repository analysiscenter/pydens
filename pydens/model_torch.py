""" Contains classes for solving differential equations with neural networks. """

from abc import ABC, abstractmethod
from contextvars import ContextVar, copy_context

import numpy as np
import torch
from torch import nn
from torch.autograd import grad
from tqdm import tqdm

from .batchflow.batchflow.models.torch.layers import ConvBlock # pylint: disable=import-error


current_model = ContextVar("current_model")

class TorchModel(ABC, nn.Module):
    """ Pytorch model for solving differential equations with neural networks. """
    def __init__(self, initial_condition=None, boundary_condition=None, ndims=1, nparams=0, **kwargs):
        _ = kwargs
        super().__init__()

        # Store the number of variables and parameters - dimensionality of the problem.
        self.ndims = ndims
        self.nparams = nparams
        self.total = ndims + nparams
        self.variables = {}

        # Parse and store initial and boundary condition.
        if initial_condition is None:
            self.initial_condition = None
        else:
            self.initial_condition = (initial_condition if callable(initial_condition)
                                      else lambda *args: torch.tensor(initial_condition, dtype=torch.float32))
        self.boundary_condition = boundary_condition

        # Initialize trainable variables for anzatc-trasform to bind initial
        # and boundary conditions.
        self.log_scale = nn.Parameter(torch.tensor(0.0, requires_grad=True))

    @abstractmethod
    def forward(self, xs):
        """ Forward of the model-network. """

    def freeze_trainable(self, layers=None, variables=None):
        """ Freeze layers and trainable variables.
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
        """ Unfreeze layers and trainable variables.
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
        xs_spatial = xs[:, :self.ndims] if self.initial_condition is None else xs[:, :self.ndims - 1]
        t = xs[:, self.ndims - 1:self.ndims]

        # Apply transformation to bind the boundary condition.
        if self.boundary_condition is not None:
            u = u * (torch.prod(xs_spatial, dim=1, keepdim=True) *
                     torch.prod((1 - xs_spatial), dim=1, keepdim=True)) + self.boundary_condition

        # Apply transformation to bind the initial condition.
        if self.initial_condition is not None:
            _xs_spatial = [xs_spatial[:, i] for i in range(xs_spatial.shape[1])]
            u = ((nn.Sigmoid()(t / torch.exp(self.log_scale)) - .5) * u
                 + self.initial_condition(*_xs_spatial).view(-1, 1))
        return u

class ConvBlockModel(TorchModel):
    """ Model that uses capabilities of batchflow.models.torch.conv_block. Can create
    large family of neural networks in a line of code.
    """
    def __init__(self, layout='fafaf', units=(20, 30, 1), activation='Sigmoid', **kwargs):
        super().__init__(**kwargs)

        # Prepare kwargs for conv-block.
        for key in ['initial_condition', 'ndims', 'nparams', 'boundary_condition']:
            _ = kwargs.pop(key, None)
        kwargs.update(layout=layout, units=list(units), activation=activation)

        # Assemble conv-block.
        fake_inputs = torch.rand((2, self.total), dtype=torch.float32)
        self.conv_block = ConvBlock(inputs=fake_inputs, **kwargs)

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

    model : class
        Class inheriting `TorchModel`. The default value is `ConvBlockModel`. The class allows
        to implement fylly connected/convolutional architectures with multiple branches
        and skip-connections. All of the `kwargs` supplied into `Solver`-initialization
        go into the `model`-class.

        Note:
        `ConvBlockModel` is based on `ConvBlock` from framework
        "`BatchFlow <https://github.com/analysiscenter/batchflow>`_".

    kwargs : dict
        Keyword-arguments used for initialization of the model-instance. When `model`-parameter
        is set to its default value - `ConvBlockModel`, use these arguments to configure the
        architecture:

        layout : str
            String defining the sequence of layers - for instance, 'fa R fa + f'.
            Letter 'f' stands for fully connected layer while letter 'c' - for convolutional layer;
            'a' inserts activation, 'R' defines the start of the skip connection, while '+' shows
            where the skip ends through sum-operation.
        units : sequence
            Sequence configuring the amount of units in all dense layers of the architecture,
            if any present in layout.
        activation : sequence
            Sequence of callables, str, for instance: [torch.Sin, torch.nn.Sigmoid, 'Sigmoid'].

        Examples:

            - ``layout, units, activation = 'fa fa f', [5, 10, 1], 'Sigmoid'``
            - ``layout, units, activation = 'faR fa fa+ f', [5, 10, 5, 1], 'Sigmoid'``
    """
    def __init__(self, equation, model=ConvBlockModel, constraints=None, **kwargs):
        self.equation = equation
        if constraints is None:
            self.constraints = ()
        elif isinstance(constraints, (tuple, list)):
            self.constraints = constraints
        else:
            self.constraints = (constraints, )
        self.losses = []
        self.optimizer = None

        # Initialize neural network for solving the equation.
        self.model = model(**kwargs)

        # Bind created model to a global context variable.
        current_model.set(self.model)
        self.ctx = copy_context()

        # Perform fake run of the model to create all the variables (coming from V-token).
        xs = [torch.rand((1, 1)) for _ in range(self.model.total)]
        for x in xs:
            x.requires_grad_()
        xs_concat = self.reshape_and_concat(xs)
        u_hat = self.model(xs_concat)
        _ = self.ctx.run(self.equation, u_hat, *xs)

    @classmethod
    def reshape_and_concat(cls, tensors):
        """ Cast, reshape and concatenate sequence of incoming tensors. """
        # Determine batch size as max-len of a tensor in included in the sequence.
        xs = list(tensors)
        sizes = ([np.prod(tensor.shape) for tensor in xs if isinstance(tensor, (np.ndarray, torch.Tensor))] +
                 [np.prod(np.array(tensor).shape) for tensor in xs if isinstance(tensor, (tuple, list))])
        batch_size = np.max(sizes) if len(sizes) > 0 else 1

        # Perform cast and reshape of all tensors in the list.
        for i, x in enumerate(xs):
            if isinstance(x, (int, float)):
                xs[i] = torch.Tensor(np.tile(x, (batch_size, 1))).float()
            if isinstance(x, np.ndarray):
                if x.size != batch_size:
                    x = np.tile(x.squeeze()[0], (batch_size, 1))
                xs[i] = torch.Tensor(x.reshape(batch_size, 1)).float()
            if isinstance(x, (list, tuple)):
                xs[i] = torch.Tensor(x).float().view(-1, 1)
            if isinstance(x, torch.Tensor):
                xs[i] = x.view(-1, 1)
        return torch.cat(xs, dim=1)

    def fit(self, niters, batch_size, sampler=None, loss_terms='equation', optimizer='Adam',
            criterion=nn.MSELoss(), lr=0.001, **kwargs):
        """ Fit the model inside the solver-instance. """
        # Initialize the optimizer if supplied.
        if optimizer is not None:
            self.optimizer = getattr(torch.optim, optimizer)([p for p in self.model.parameters()
                                                              if p.requires_grad],
                                                              lr=lr, **kwargs)

        # Perform `niters`-iterations of optimizer steps.
        self.model.train()
        for _ in tqdm(range(niters)):
            self.optimizer.zero_grad()

            # Sample batch of points and compute solution-approximation on the batch.
            if sampler is None:
                xs = [torch.rand((batch_size, 1)) for _ in range(self.model.total)]
            else:
                xs_array = sampler.sample(batch_size).astype(np.float32)
                xs = [torch.from_numpy(xs_array[:, i:i+1]) for i in range(xs_array.shape[1])]
            for x in xs:
                x.requires_grad_()
            xs_concat = self.reshape_and_concat(xs)
            u_hat = self.ctx.run(self.model, xs_concat)

            # Compute loss: form it summing equation-loss and constraints-loss.
            loss_terms = loss_terms if isinstance(loss_terms, (tuple, list)) else (loss_terms, )
            nums_constraints = [int(term_name.replace('constraint', '').replace('_', ''))
                                for term_name in loss_terms if 'constraint' in term_name]
            loss  = 0

            # Include equation loss.
            if 'equation' in loss_terms:
                loss += criterion(self.ctx.run(self.equation, u_hat, *xs), torch.zeros_like(xs[0]))

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
        """ Get approximation to a solution in a set of points.
        Points are given by a list of tensors.
        """
        # Reshape and concat.
        xs = self.reshape_and_concat(xs)

        # Perform inference.
        self.model.eval()
        result = self.ctx.run(self.model, xs)
        return result.detach().cpu().numpy()
