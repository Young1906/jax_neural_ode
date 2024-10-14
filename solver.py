"""
!!! Implement some ODE solvers:
    - Euler
    - RK2
    - RK4
    - Predictor-Correct

    Features:
    - Vmappable
    - Saveat Irregular time steps (see saveat)
"""
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from matplotlib import pyplot as plt


"""
f : dynamic function
y : state of the dynamic system
t : current time
h : time-step
"""

def EULER(f, s, t, h):
    return f(s, t) * h 


def RK2(f, s, t, h):
    k1 = f(s, t)
    k2 = f(s + k1 * h, t + h)

    return (k1 + k2) * h * .5


def RK4(f, s, t, h):
    k1 = f(s, t)
    k2 = f(s + .5 * k1 * h, t + .5 * h)
    k3 = f(s + .5 * k2 * h, t + .5 * h)
    k4 = f(s + k3 * h, t + h)

    return (k1 + 2 * k2 + 2 * k3 + k4) * h / 6.


def PC(f, s, t, h):
    """
    """
    half_step = s + f(s, t) * h * .5
    return f(half_step, t + h * .5) * h


class Solver:
    def __init__(self, step_fn: callable, h_max: float):
        self.h_max = h_max
        self.step_fn = step_fn


    def __call__(self,f: callable, s0: jnp.ndarray, tmin: float, tmax: float):
        """
        Args:
        - f: ODE's dynamic function
        - s0: ODE's initial state
        - tmin, tmax: initial and terminal time-step

        Returns -> s(tmax)
        """

        n_step = jnp.astype(
                jnp.ceil(jnp.abs(tmin - tmax)/self.h_max),
                jnp.int32)
        step_size = (tmax - tmin)/n_step

        #  def cond_fn(val):
        #      s, t = val
        #      return t < tmax

        def fn(i, val):
            s, t = val

            # take correction step in the last step
            def true_fn(s, t):
                step_size = tmax - t
                s = s + self.step_fn(f, s, t, step_size)
                t = t + step_size
                return s, t

            def false_fn(s, t):
                s = s + self.step_fn(f, s, t, step_size)
                t = t + step_size
                return s, t

            pred = jnp.abs(tmax - t) < step_size
            return jax.lax.cond(pred, true_fn, false_fn, s, t)

        # initial value
        val = (s0, tmin)
        s1, t1 = jax.lax.fori_loop(0, n_step, fn, val)
        return s1 



    def saveat(self, f: callable, s0, t_seq: jnp.ndarray):
        """
        Save solution at multiple timestep (tseq)
        """
        T = jnp.c_[t_seq[:-1], t_seq[1:]]

        def fn(s0, ts):
            tmin, tmax = ts

            s1 = self.__call__(f, s0, tmin, tmax)
            return s1, s1
        sT, S = jax.lax.scan(fn, s0, T)
        s0 = jnp.expand_dims(s0, 0)
        S = jnp.concatenate([s0, S], axis=0)

        return sT, S
