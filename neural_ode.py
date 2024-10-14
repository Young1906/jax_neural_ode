from collections.abc import Callable
from functools import partial

import click
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tqdm
import yaml
from jax import random as jr
from matplotlib import pyplot as plt

from src.net import MLP
from src.ode.base import NeuralODEConfig
from src.ode.data import batching
from src.ode.data import create_dataset
from src.ode.dynamics import spiral_fn
from src.ode.solver import EULER
from src.ode.solver import PC
from src.ode.solver import RK2
from src.ode.solver import Solver
from src.ode.utils import State


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

        # leaves, _ = jax.tree_util.tree_flatten(grad)

        # for l in leaves:
        #     jax.debug.print("{}", l)
        

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


def train(
        net:eqx.Module,
        solver: Callable,
        data: list[jnp.ndarray],
        config: NeuralODEConfig,
        key: jr.PRNGKey):
    """
    Train a neural network to learn the true dynamics
    """
    Ys, Ts = data

    # optimizer
    optim = optax.adamax(config.training.learning_rate)
    optim_state = optim.init(net)

    @jax.jit
    def train_step(model, optim_state, bY, bT):
        """
        Args:
        - net: model to be trained
        - optim_state: state of the optimizer
        - dseed: seed for random data generation
        """
        bY_hat, update = model.forward(bY, bT, solver)
        _loss = jnp.sum((bY_hat - bY)**2) / config.dataset.batch_size

        update, optim_state = optim.update(update, optim_state)
        new_model = eqx.apply_updates(model, update)

        return _loss, new_model, optim_state


    losses = []
    pbar = tqdm.trange(config.training.n_iter)

    key, bkey = jr.split(key, 2)

    for i in pbar:
        bkey, (bY, bT) = batching(bkey, Ys, Ts, config.dataset.batch_size)
        loss, net, optim_state = train_step(net, optim_state, bY, bT)
        pbar.set_description(f"Loss = {loss:.3f}")
        losses.append(loss)

    return net, losses





################################################################################
# Main program
################################################################################
@click.command()
@click.option("--config", "-C", type=str, help="path/to/config", required=True)
def main(config: str):
    with open(config, "r") as f:
        config = yaml.safe_load(f)
        config = NeuralODEConfig(**config["params"])

    seed = jr.PRNGKey(config.seed)

    # seed for model and data
    seed, mseed, dseed, bseed = jr.split(seed, 4)

    # True dynamic:
    _, spiral_dynamic = spiral_fn()

    # create dataset
    Ys, Ts = create_dataset(
            dseed,
            config.dataset.n_samples,
            config.dataset.seq_len,
            config.dataset.tmin,
            config.dataset.tmax,
            spiral_dynamic,
            noise_scale=config.dataset.noise_scale)

    # model & neural ode
    solver = Solver(RK2, config.solver.h_max)
    f_theta = MLP(key=mseed, layers=config.model.mlp_layers)
    net = NeuralODE(f_theta) # NeuralODE

    # train
    net, losses = train(net, solver, (Ys, Ts), config, bseed)
    eqx.tree_serialise_leaves("weights/neural_ode.eqx", net)


    # --------------------------------------------------
    # DEBUG CODE
    # --------------------------------------------------
    # seed = jr.PRNGKey(config.seed)
    # seed, fseed, dseed = jr.split(seed, 3)
    # solver = Solver(RK2, .0031)
    # f_theta = MLP(key=fseed, layers=[config.u_dim, 2])
    # net = NeuralODE(f_theta) # NeuralODE

    # y0 = jnp.array([0., 0.])
    # y1 = jnp.array([0., 0.])
    # t0 = 0.
    # t1 = .25

    # y1_hat, grad = net._once(y0, t1, t0, t1, solver)
    # # print(y1_hat, grad)
    # leaves, _ = jax.tree_util.tree_flatten(grad)
    # print(leaves)

    # --------------------------------------------------
    # Visualization 
    # --------------------------------------------------
    plt.plot(losses, lw=1., c="black", marker="x")
    plt.savefig("neural_ode_loss.png")
    plt.show()

    return 0


if __name__ == "__main__":
    main()
