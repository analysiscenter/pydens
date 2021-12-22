import torch
from torch import nn
import numpy as np
from torch.autograd import grad
from tqdm import tqdm
from contextvars import Context, ContextVar, copy_context
from .batchflow.batchflow.models.torch.layers import ConvBlock


current_model = ContextVar("current_model")

class TorchModel(nn.Module):
    def __init__(self, initial_condition=None, boundary_condition=None, ndims=1, nparams=0, **kwargs):
        """ Pytorch model for solving differential equations with neural networks.
        """
        super().__init__()

        # Store the number of variables and parameters - dimensionality of the problem.
        self.ndims = ndims
        self.nparams = nparams
        self._total = ndims + nparams
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

    def freeze_trainable(self, layers=None, variables=None):
        """ Freeze layers and trainable variables.
        """
        layers = () if layers is None else layers
        variables = () if variables is None else variables

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
        layers = () if layers is None else layers
        variables = () if variables is None else variables

        # Unfreeze layers.
        for layer in layers:
            for param in getattr(self, layer).parameters():
                param.requires_grad = True

        # Unfreeze variables.
        for variable in variables:
            param = getattr(self, variable)
            param.requires_grad = True

    def forward(self, *xs):
        xs = [x.view(-1, 1) for x in xs]
        xs = torch.cat(xs, dim=1)
        return xs

    def anzatc(self):
        """ Make transform of the model-output needed for binding initial and boundary conditions.
        """
        def func(u, xs):
            """ Anzatc-transformation itself. """
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
        return func

class ConvBlockModel(TorchModel):
    def __init__(self, layout='fafaf', units=[20, 30, 1], activation='Sigmoid', **kwargs):
        """ Using capabilities of conv-block.
        """
        super().__init__(**kwargs)

        # Prepare kwargs for conv-block.
        for key in ['initial_condition', 'ndims', 'nparams', 'boundary_condition']:
            _ = kwargs.pop(key, None)
        kwargs.update(layout=layout, units=units, activation=activation)

        # Assemble conv-block.
        fake_inputs = torch.rand((2, self._total), dtype=torch.float32)
        self.conv_block = ConvBlock(inputs=fake_inputs, **kwargs)

    def forward(self, *xs):
        xs = super().forward(*xs)
        u = self.conv_block(xs)
        return self.anzatc()(u, xs)

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
    """ Solver-class for handling differential equations with neural networks.
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

        # Initialize neural network for solving the equation.
        self.model = model(**kwargs)

        # Bind created model to a global context variable.
        current_model.set(self.model)
        self.ctx = copy_context()

        # Perform fake run of the model to create all the variables (coming from V-token).
        xs = [torch.rand((1, 1)) for _ in range(self.model._total)]
        for x in xs:
            x.requires_grad_()
        u_hat = self.model(*xs)
        _ = self.ctx.run(self.equation, u_hat, *xs)
        # _ = self.equation(u_hat, *xs)


    def fit(self, niters, batch_size, sampler=None, losses='equation', optimizer='Adam', criterion=nn.MSELoss(),
            lr=0.001, **kwargs):
        # Initialize the optimizer if supplied.
        if optimizer is not None:
            self.optimizer = getattr(torch.optim, optimizer)([p for p in self.model.parameters()
                                                              if p.requires_grad],
                                                              lr=lr, **kwargs)

        # Perform `niters`-iterations of optimizer steps.
        for _ in tqdm(range(niters)):
            self.optimizer.zero_grad()

            # Sample batch of points and compute solution-approximation on the batch.
            if sampler is None:
                xs = [torch.rand((batch_size, 1)) for _ in range(self.model._total)]
            else:
                xs_concat = sampler.sample(batch_size).astype(np.float32)
                xs = [torch.from_numpy(xs_concat[:, i:i+1]) for i in range(xs_concat.shape[1])]
            for x in xs:
                x.requires_grad_()
            u_hat = self.ctx.run(self.model, *xs)

            # Compute loss: form it summing equation-loss and constraints-loss.
            losses = losses if isinstance(losses, (tuple, list)) else (losses, )
            nums_constraints = [int(loss_name.replace('constraint', '').replace('_', ''))
                                for loss_name in losses if 'constraint' in loss_name]
            loss  = 0
            if 'equation' in losses:
                loss += criterion(self.ctx.run(self.equation, u_hat, *xs), torch.zeros_like(xs[0]))
            for num in nums_constraints:
                loss += criterion(self.ctx.run(self.constraints[num], self.model, *xs),
                                  torch.Tensor([0.0]))

            # Optimizer step.
            loss.backward()
            self.optimizer.step()

            # Gather and store training stats.
            self.losses.append(loss.detach().numpy())

    def solve(self, *xs):
        """ Get approximation to a solution in a set of points.
        Points are given by a list of tensors.
        """
        result = self.ctx.run(self.model, *xs)
        return result.cpu().detach().numpy()
