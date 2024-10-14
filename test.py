"""
Test case for ODE solver
"""
import unittest

import jax
import jax.numpy as jnp
import jax.random as jr

from src.ode.solver import EULER
from src.ode.solver import PC
from src.ode.solver import RK2
from src.ode.solver import RK4
from src.ode.solver import Solver

class TestSolver(unittest.TestCase):
    def setUp(self):
        self.h_max = 1e-2
        self.tmin, self.tmax = 0., 1.
        self.thres = 1e-2
        self.rmse = lambda a, b: jnp.mean((a - b)**2)**.5
        self.f = lambda x, t: -x
        self.s = lambda s0: s0 / jnp.e

    def test_rk4(self):
        solver = Solver(RK4, self.h_max)
        s0 = jnp.e
        s1 = self.s(s0)
        s1_hat = solver(self.f, s0, self.tmin, self.tmax)
        err = self.rmse(s1_hat, s1)
        self.assertLessEqual(err, self.thres)

    def test_rk2(self):
        solver = Solver(RK2, self.h_max)
        s0 = jnp.e
        s1 = self.s(s0)
        s1_hat = solver(self.f, s0, self.tmin, self.tmax)
        err = self.rmse(s1_hat, s1)
        self.assertLessEqual(err, self.thres)

    def test_pc(self):
        solver = Solver(PC, self.h_max)
        s0 = jnp.e
        s1 = self.s(s0)
        s1_hat = solver(self.f, s0, self.tmin, self.tmax)
        err = self.rmse(s1_hat, s1)
        self.assertLessEqual(err, self.thres)

    def test_euler(self):
        solver = Solver(EULER, self.h_max)
        s0 = jnp.e
        s1 = self.s(s0)
        s1_hat = solver(self.f, s0, self.tmin, self.tmax)
        err = self.rmse(s1_hat, s1)
        self.assertLessEqual(err, self.thres)

    def test_high_dimensional(self):
        solver = Solver(RK2, self.h_max)
        s0 = jnp.array([-jnp.e, jnp.e])
        s1 = self.s(s0)
        s1_hat = solver(self.f, s0, self.tmin, self.tmax)
        err = self.rmse(s1, s1_hat) 
        self.assertLessEqual(err, self.thres)

    def test_vmappability(self):
        key = jr.PRNGKey(0)
        s0 = jr.uniform(key=key, shape=(32, 2), minval=-1, maxval=1)
        s1 = self.s(s0) 

        solver = Solver(RK2, self.h_max)
        s1_hat= jax.vmap(solver, (None, 0, None, None))(
                self.f, s0,
                self.tmin, self.tmax)
        mse = jnp.mean((s1_hat - s1)**2)
        self.assertLessEqual(mse, self.thres)


    def test_saveat_vmappablility(self):
        key = jr.PRNGKey(0)
        k1, k2 = jr.split(key, 2)

        s0 = jr.uniform(key=k1, shape=(32, 2), minval=-1, maxval=1)
        t_seq = jr.uniform(key=k2, shape=(32, 16), minval=0,
                           maxval=1).sort(axis=-1)

        tmin, tmax = t_seq.min(axis=-1), t_seq.max(-1)
        tdiff = tmax - tmin
        s1 = s0 * jnp.exp(-tdiff[:, jnp.newaxis])
        solver = Solver(RK2, self.h_max)

        s1_hat, S = jax.vmap(solver.saveat, (None, 0, 0))(
                self.f, s0, t_seq)

        err = self.rmse(s1_hat, s1)
        self.assertLessEqual(err, self.thres)
