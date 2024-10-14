import jax
from jax import numpy as jnp
from jax import random as jr

from src.ode.solver import RK4
from src.ode.solver import Solver

"""
Reference from: 
https://docs.kidger.site/diffrax/examples/neural_ode/
"""
def create_dataset(
        key             : jr.PRNGKey,
        n_samples       : int,
        n_time_step     : int,
        tmin            : float,
        tmax            : float,
        f_dynamic       : callable, 
        noise_scale     : float):
    """
    Given dynamic f and a random iv of a system:
    compute the state at timestep t for T = {t_i}_i=1...N
    """

    k1, k2, k3 = jr.split(key, 3)

    # random initial value
    y0 = jr.uniform(
            key=k1,
            shape=(n_samples, 2),
            minval=-1.,
            maxval=1.)

    # random time-step
    Ts = jr.uniform(
            key=k2,
            shape=(n_samples, n_time_step),
            minval=tmin,
            maxval=tmax).sort(axis=-1) # time is sorted in the last dimension

    # solver
    solv = Solver(RK4, h_max=0.0071)
    _, Ys = jax.vmap(solv.saveat, (None, 0, 0))(f_dynamic, y0, Ts)

    # Adding noise
    noise = jr.normal(key=k3, shape=Ys.shape) * noise_scale
    Ys = Ys + noise

    return Ys, Ts 


def batching(key, Ys, Ts, batch_size):
    key, next_key = jr.split(key, 2)
    n_samples = jnp.size(Ys, 0)
    idx = jr.randint(key, shape=(batch_size,), minval=0, maxval=n_samples)

    return next_key, (Ys[idx, ...], Ts[idx])
