import torch
from torch import nn
import numpy as np
from torch.autograd import grad
from tqdm import tqdm
from contextvars import Context, ContextVar, copy_context


current_model = ContextVar("current_model")

class TorchModel(nn.Module):
    def __init__(self, initial_condition=None, boundary_condition=None, ndims=1, nparams=0, **kwargs):
        """ Pytorch model for solving differential equations with neural networks.
        """
        super().__init__(**kwargs)
        # problem's dimensionality
        self.ndims = ndims
        self.nparams = nparams
        self._total = ndims + nparams
        self.variables = {}

        if initial_condition is None:
            self.initial_condition = None
        else:
            self.initial_condition = initial_condition if callable(initial_condition) else lambda: initial_condition
        self.boundary_condition = boundary_condition

        # variables for anzatc
        self.log_scale = nn.Parameter(torch.tensor(0.0, requires_grad=True))

    def freeze_layers(self, layers=None, variables=None):
        """ Freeze layers
        """
        layers = () if layers is None else layers
        variables = () if variables is None else variables
        for layer in layers:
            for param in getattr(self, layer).parameters():
                param.requires_grad = False
        for variable in variables:
            param = getattr(self, variable)
            param.requires_grad = False

    def unfreeze_layers(self, layers=None, variables=None):
        """ Unfreeze layers
        """
        layers = () if layers is None else layers
        variables = () if variables is None else variables
        for layer in layers:
            for param in getattr(self, layer).parameters():
                param.requires_grad = True
        for variable in variables:
            param = getattr(self, variable)
            param.requires_grad = True

    def forward(self, *xs):
        xs = [x.view(-1, 1) for x in xs]
        xs = torch.cat(xs, dim=1)
        return xs

    def anzatc(self):
        """ Transform of the model-output needed for binding initial and boundary conditions.
        """
        def func(u, xs):
            # get spatial variables and time separately from incoming xs
            xs_spatial = xs[:, :self.ndims] if self.initial_condition is None else xs[:, :self.ndims - 1]
            t = xs[:, self.ndims-1:self.ndims]
            # get rid of parameters in anzatc
            if self.boundary_condition is not None:
                u = u * (torch.prod(xs_spatial, dim=1, keepdim=True) *
                         torch.prod((1 - xs_spatial), dim=1, keepdim=True)) + self.boundary_condition
            if self.initial_condition is not None:
                u = (nn.Sigmoid()(t / torch.exp(self.log_scale)) - .5) * u + self.initial_condition()
            return u
        return func


class CustomModel(TorchModel):
    def __init__(self, **kwargs):
        """ Simple model for solving PDEs - inherits base torch-model.
        """
        super().__init__(**kwargs)

        # define network layers
        self.fc1 = nn.Linear(self.ndims + self.nparams, 20)
        self.ac1 = nn.Sigmoid()
        self.fc2 = nn.Linear(20, 30)
        self.ac2 = nn.Sigmoid()
        self.fc3 = nn.Linear(30, 1)

    def forward(self, *xs):
        xs = super().forward(*xs)
        u = self.fc1(xs)
        u = self.ac1(u)
        u = self.fc2(u)
        u = self.ac2(u)
        u = self.fc3(u)

        return self.anzatc()(u, xs)


def D(y, x):
    """ Differentiation token.
    """
    res = grad(y.sum(), x, retain_graph=True, create_graph=True)[0]
    return res

def V(name, *args, **kwargs):
    """ Token for a trainable variable.
    """
    # if does not exist create ow just take what's been created from the model
    model = current_model.get()
    if not hasattr(model, name):
        setattr(model, name, nn.Parameter(*args, **kwargs))
    return getattr(model, name)


class Solver():
    def __init__(self, equation, model=CustomModel, constraints=None, **kwargs):
        self.equation = equation
        if constraints is None:
            self.constraints = ()
        elif isinstance(constraints, (tuple, list)):
            self.constraints = constraints
        else:
            self.constraints = (constraints, )

        self.model = model(**kwargs)
        self.losses = []

        current_model.set(self.model)
        self.ctx = copy_context()

        # fake run to create all variables and the making an optimizer
        xs = [torch.rand((1, 1)) for _ in range(self.model._total)]
        for x in xs:
            x.requires_grad_()
        u_hat = self.model(*xs)
        _ = self.ctx.run(self.equation, u_hat, *xs)
        _ = self.equation(u_hat, *xs)


    def fit(self, niters, batch_size, sampler=None, losses='equation', optimizer='Adam', criterion=nn.MSELoss(),
            lr=0.001, **kwargs):
        # form the optimizer
        if optimizer is not None:
            self.optimizer = getattr(torch.optim, optimizer)([p for p in self.model.parameters() if p.requires_grad],
                                                             lr=lr, **kwargs)
        for _ in tqdm(range(niters)):
            # sampling points and passing it through the net
            self.optimizer.zero_grad()

            if sampler is None:
                xs = [torch.rand((batch_size, 1)) for _ in range(self.model._total)]
            else:
                xs_concat = sampler.sample(batch_size).astype(np.float32)
                xs = [torch.from_numpy(xs_concat[:, i:i+1]) for i in range(xs_concat.shape[1])]
            for x in xs:
                x.requires_grad_()
            u_hat = self.ctx.run(self.model, *xs)

            # form loss function optimizing some of equations and constraints
            losses = losses if isinstance(losses, (tuple, list)) else (losses, )
            nums_constraints = [int(loss_name.replace('constraint', '').replace('_', ''))
                                for loss_name in losses if 'constraint' in loss_name]
            loss  = 0
            if 'equation' in losses:
                loss += criterion(self.ctx.run(self.equation, u_hat, *xs), torch.zeros_like(xs[0]))
            for num in nums_constraints:
                loss += criterion(self.ctx.run(self.constraints[num], self.model, *xs),
                                  torch.Tensor([0.0]))

            loss.backward()
            self.optimizer.step()

            # gather stats
            self.losses.append(loss.detach().numpy())
