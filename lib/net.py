import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random as jr

from lib.solver import EULER
from lib.solver import PC
from lib.solver import RK2
from lib.solver import Solver
from lib.utils import State

class MLP(eqx.Module):
    """
    Fully Connected MLP
    """

    layers: list

    def __init__(self, layers: list[int], key):
        self.layers = []

        for _in, _out in zip(layers[:-1], layers[1:]):
            key, subkey = jr.split(key, 2)
            self.layers.append(eqx.nn.Linear(_in, _out, key=subkey))

    def __call__(self, u, t):
        """
        assuming x in R^{n x (d - 1)}, t in R
        """
        if jnp.ndim(t) == 0:
            t = jnp.expand_dims(t, -1)

        out = jnp.concatenate([u, t], -1)
        # out = u

        for layer in self.layers[:-1]:
            out = layer(out)
            out = jax.nn.relu(out)

        out = self.layers[-1](out)
        return jax.nn.tanh(out)


class NeuralODE(eqx.Module):
    f: eqx.Module 

    def __init__(self, f: eqx.Module):
        self.f = f

    def __forward(self, y0, t0, t1, solver):
        """
        # forward 1 time-step for a single sample
        y0: R^n
        t0, t1: R
        """
        y1_hat = solver(self.f, y0, t0, t1)
        return y1_hat


    def __call__(self, y0, t0, t1, solver):
        """
        For inference
        """
        y1_hat = solver(self.f, y0, t0, t1)
        return y1_hat


    def __adjoint(self, y1_hat, y1, t0, t1, solver):
        """
        compute dL/dtheta for single step from t0 to t1

        y1, y1_hat : R^n
        t0, t1: R
        """

        dgdy = jax.grad(lambda a, b: jnp.sum((a-b)**2))
        dfdy = jax.jacfwd(lambda y, t: self.f(y, t))
        dfdtheta = jax.jacfwd(lambda f, y, t: f(y, t))


        # define initial state for augmented dynamic
        dgdy1 = dgdy(y1_hat, y1)

        # jax.debug.print("{}", dgdy1)

        zero_grad = jax.tree.map(lambda x: x * .0, self.f)

        s0 = State((
            y1_hat,
            dgdy1,
            zero_grad))

        # dynamic of the adjoint state for a single sample 
        def augmented_adjoint_dynamic(s, t):
            # unpack
            y, a, grad = s

            s0 = self.f(y, t)
            s1 = -a @ dfdy(y, t)
            s2 = jax.tree.map(
                    lambda x: jnp.einsum('j,j...->...', -a, x),
                    dfdtheta(self.f, y, t))


            return State((s0, s1, s2)) # adfdtheta

        s1 = solver(
                augmented_adjoint_dynamic,
                s0, t1, t0)

        y0, dldy0, dgdtheta = list(s1)
        return dldy0, dgdtheta

    def __once(self, y0, y1, t0, t1, solver):
        """
        step forward 1 and then compute the gradient
        """
        y1_hat = self.__forward(y0, t0, t1, solver)
        dldy0, grad = self.__adjoint(y1_hat, y1, t0, t1, solver)

        return y1_hat, grad 

    def __all(self, Y, T, solver):
        """
        Prediction & Gradient Across the time dimesntion
        shape's:
        Y: (seq x N)
        T: (seq)
        
        Ref: 
            - jax.lax.scan : https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html
        """

        zero_grad = jax.tree.map(lambda x: x * .0000, self.f)
        init = (Y[0, :], T[0], zero_grad)

        def fn(carry, x):
            y0, t0, grad = carry 
            y1, t1 = x 

            y1_hat, _grad = self.__once(y0, y1, t0, t1, solver)
            grad = jax.tree.map(lambda a, b: a + b, grad, _grad) 

            return (y1, t1, grad), (y1_hat, t1)

        xs = (Y, T)
        (_, _, grad), (Y_hat, _) = jax.lax.scan(fn, init, xs)

        return Y_hat, grad


    def forward(self, Ys, Ts, solver):
        """
        shape's:
        Ys: (b x seq x N)
        Ts: (b x seq)
        """
        Ys_hat, grad = jax.vmap(self.__all, (0, 0, None))(Ys, Ts, solver)
        # Ys_hat, grad = self.__all(Ys[0, :, :], Ts[0, :], solver)
        # print(graD)

        # Summing gradient from across sample in the batch
        grad = jax.tree.map(
                lambda x: jnp.einsum('j... -> ...', x), grad)

        # convert from f -> NeuralODE gradient
        grad = NeuralODE(grad)
        return Ys_hat, grad
