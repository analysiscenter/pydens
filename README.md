# PyDEns

**PyDEns** is a framework for solving Partial Differential Equations (PDEs) using neural networks. With **PyDEns** a user can solve PDEs from a large family including [heat-equation](https://en.wikipedia.org/wiki/Heat_equation), [poisson equation](https://en.wikipedia.org/wiki/Poisson%27s_equation) and [wave-equation](https://en.wikipedia.org/wiki/Wave_equation).

## Getting started with **PyDEns**: solving common PDEs
Let's solve poisson equation

<p align="center">
<img src="./imgs/poisson_eq.png?invert_in_darkmode" align=middle width=621.3306pt height=38.973825pt/>
</p>

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

<p align="center">
<img src="./imgs/poisson_sol.png?invert_in_darkmode" align=middle height=295.973825pt/>
</p>

## Going deeper into **PyDEns**-capabilities
**PyDEns** allows to do much more than just solve common PDEs: it also deals with (i) parametric families of PDEs and (ii) PDEs with trainable coefficients.

### Solving parametric families of PDEs
Consider a *family* of ordinary differential equations

<p align="center">
<img src="./imgs/sinus_eq.png?invert_in_darkmode" align=middle height=40.973825pt/>
</p>

Clearly, the solution is **sin** with a phase parametrized by ϵ:

<p align="center">
<img src="./imgs/sinus_sol_expr.png?invert_in_darkmode" align=middle height=18.973825pt/>
</p>

Solving this problem is just as easy as solving common PDEs:

```python
pde = {'n_dims': 1,
       'form': lambda u, t, e: D(u, t) - P(e) * np.pi * cos(P(e) * np.pi * t),
       'initial_condition': 1}

config = {'body': body,
          'pde': pde,
          'loss': loss}

s = NumpySampler('uniform') & NumpySampler('uniform', low=1, high=5)

dg = Solver(config)
dg.fit(batch_size=1000, sampler=s, n_iters=5000, bar='notebook')
```

Check out the result:

<p align="center">
<img src="./imgs/sinus_sol.gif?invert_in_darkmode" align=middle height=250.973825pt/>
</p>

### Solving PDEs with trainable coefficients

With **PyDEns** things can get even more interesting! Assume that the *initial state of the system is unknown and to be determined*:

<p align="center">
<img src="./imgs/sinus_eq_trainable.png?invert_in_darkmode" align=middle height=40.973825pt/>
</p>

Of course, without additional information, [the problem is undefined](https://en.wikipedia.org/wiki/Initial_value_problem). To make things better, let's fix the state of the system at some other point:

<p align="center">
<img src="./imgs/sinus_eq_middle_fix.png?invert_in_darkmode" align=middle height=18.973825pt/>
</p>

Setting this problem requires a slightly more complex configuring:

```python
pde = {'n_dims': 1,
       'form': lambda u, t: D(u, t) - 2 * np.pi * cos(2 * np.pi * t),
       'initial_condition': lambda: V(3.0, 'initial')}

config = {'pde': pde,
          'track': {'u05': lambda u, t: u - 2,
                    'dt': lambda u, t: D(u, t)},
          'train_steps': {'initial_condition_step': {'scope': 'addendums',
                                                     'loss': {'name': 'mse', 'predictions': 'u05'}},
                          'equation_step': {'scope': '-addendums'}}}

s1 = NumpySampler('uniform')
s2 = ConstantSampler(0.5)
```

## Installation

### Installation as a project repository:

  Do not forget to use the flag ``--recursive`` to make sure that ``BatchFlow`` submodule is also cloned.

        git clone --recursive https://github.com/analysiscenter/pydens.git
