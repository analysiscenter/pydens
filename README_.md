# PyDEns

**PyDEns** is a framework for solving Partial Differential Equations (PDEs) using neural networks. With **PyDEns** a user can solve PDEs from a large family including [heat-equation](https://en.wikipedia.org/wiki/Heat_equation), [poisson equation](https://en.wikipedia.org/wiki/Poisson%27s_equation) and [wave-equation](https://en.wikipedia.org/wiki/Wave_equation).

## Getting started with **PyDEns**: solving common PDEs
Let's solve poisson equation

$$
\frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} = 5\sin[\pi(x + y)],
\\ \quad (x, y) \in [0, 1]^2,
f(0, y)=f(x, 0)=f(1, y)=f(x, 1)=1.
$$

using simple feed-forward neural network with `tahn`-activations. We only need to set up a **PyDEns**-model for solving the task at hand

```python
from pydens import Solver, NumpySampler
import numpy as np

pde = {'n_dims': 2,
       'form': lambda u, x, y: D(D(u, x), x) + D(D(u, y), y) - 5 * sin(np.pi * (x + y)),
       'boundary_condition': 1}

body = {'layout': 'fa fa fa f',
        'units': [15, 25, 15, 1],
        'activation': [tf.nn.tanh, tf.nn.tanh, tf.nn.tanh]}

config = {'body': body,
          'pde': pde,
          'loss': 'mse'}

us = NumpySampler('u', dim=2)
```

and run the optimization procedure

```python
dg = Solver(config)
dg.fit(batch_size=100, sampler=us, n_iters=1500, bar='notebook')
```
in a fraction of second we've got a mesh-free approximation of the solution on **[0, 1]X[0, 1]**-square:

## Going deeper into **PyDEns**-capabilities
**PyDEns** allows to do much more than just solve common PDEs: it also deals with (i) parametric families of PDEs and (ii) PDEs with trainable coefficients.

### Solving parametric families of PDEs


### Solving PDEs with trainable coefficients

## Installation

### Installation as a `python`-package

With [pip](https://pip.pypa.io/en/stable/):

    pip3 install git+https://github.com/analysiscenter/pydens.git

### Installation as a project repository:

  Do not forget to use the flag ``--recursive`` to make sure that ``BatchFlow`` submodule is also cloned.

        git clone --recursive https://github.com/analysiscenter/pydens.git
