import jax
import jax.numpy as jnp


"""
Some toy dynamical system (ODE initial value problem):
    - damped harmonical occilator
    - bunny and wolf population dynamic


signature:
    f(*args, **kwargs) -> s0, f:
    - s0
"""


# Dampen Harmoic Occililator
def damped_harmonic_occiliator_fn():
    """
    Problem:
        ODE: x'' + c/m x' + k/m x = 0
        IV: x(t0) = A, x'(t0) = V

    Params:
        c: dampener coefficient
        k: spring coefficient
        m: mass
        S0: initial position and velocity

    Returns: s0, f
    """
    # fixed params
    c = 0.1
    k = 1.0
    m = 1.0
    S0 = jnp.array([1., -10.0])

    # constructing dynamic function
    A = jnp.array([[0, 1.0], [-k / m, -c / m]])
    F = lambda S, t: A @ S

    return S0, F


# Sample from git's tutorial
# https://github.com/KamenB/neural-ode-tutorial/blob/main/Neural%20Ordinary%20Differential%20Networks.ipynb


def spiral_fn():
    """
    Adapting SpiralFunctionExample
    """
    A = jnp.array([[-0.1, -1.0], [1.0, -0.1]])
    F = lambda S, t: A @ S

    S0 = jnp.array([0.6, .3])

    return S0, F
