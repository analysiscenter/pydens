import torchvision
import torch
from torch import nn
import numpy as np
from torch.autograd import grad


class TorchModel(nn.Module):
    def __init__(self, initial_condition=None, boundary_condition=None, ndims=1, nparams=0, **kwargs):
        """ Pytorch model for solving differential equations with neural networks.
        """
        super().__init__(**kwargs)
        # problem's dimensionality
        self.ndims = ndims
        self.nparams = nparams
        self.variables = {}

        self.initial_condition = initial_condition
        self.boundary_condition = boundary_condition

        # variables for anzatc
        self.log_scale = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        pass

    def forward(self, *xs):
        xs = [x.view(-1, 1) for x in xs]
        xs = torch.cat(xs, dim=1)
        return xs

    def anzatc(self):
        """ Transform of the model-output needed for binding initial and boundary conditions.
        """
        def func(u, xs):
            if self.boundary_condition is not None:
                u = u * (torch.prod(xs, dim=1, keepdim=True) *
                         torch.prod((1 - xs), dim=1, keepdim=True)) + self.boundary_condition
            if self.initial_condition is not None:
                t = xs[:, -1:]
                u = (nn.Sigmoid()(t / torch.exp(self.log_scale)) - .5) * u + self.initial_condition
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

        return self.anzatc()(u, xs[:, :self.ndims])


def D(y, x):
    """ Differentiation token.
    """
    res = grad(y.sum(), x, retain_graph=True, create_graph=True)[0]
    return res


def V(name, *args, **kwargs):
    """ Token for a trainable variable.
    """
    # if does not exist create ow just take what's been created from the model
    if not hasattr(current_model, name):
        print('creating nn.Parameter')
        setattr(current_model, name, nn.Parameter(*args, **kwargs))
    return getattr(current_model, name)


class Solver():
    def __init__(self, equation, model=CustomModel, criterion=nn.MSELoss(),
                 fake_run=True, **kwargs):
        self.equation = equation
        self.criterion = criterion
        self.model = model(**kwargs)
        self.losses = []
        self.ndims = kwargs.get('ndims')

        # fake run to create all variables and the making an optimizer
        xs = [torch.rand((1, 1)) for _ in range(self.ndims)]
        for x in xs:
            x.requires_grad_()
        u_hat = self.model(*xs)
        _ = self.equation(u_hat, *xs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)


    def fit(self, niters, batch_size):
        for _ in tqdm_notebook(range(niters)):
            # sampling points and passing it through the net
            self.optimizer.zero_grad()
            xs = [torch.rand((batch_size, 1)) for _ in range(self.ndims)]
            for x in xs:
                x.requires_grad_()
            u_hat = self.model(*xs)

            with self.model as current_model:
                loss = (
                        self.criterion(self.equation(u_hat, *xs), torch.zeros_like(xs[0]))
                       )
                loss.backward()
                self.optimizer.step()

            # gather stats
            self.losses.append(loss.detach().numpy())