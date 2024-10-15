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

from lib.base import NeuralODEConfig
from lib.data import batching
from lib.data import create_dataset
from lib.dynamics import spiral_fn
from lib.net import MLP
from lib.net import NeuralODE
from lib.solver import EULER
from lib.solver import PC
from lib.solver import RK2
from lib.solver import Solver


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

    # Inference & visualization
    plt.rcParams["figure.figsize"] = [12, 5]
    plt.subplot(121)
    plt.plot(losses, lw=1., c="black", marker="x", label="loss")

    plt.subplot(122)
    Y, T = Ys[1, ...], Ts[1]

    saveat = jnp.linspace(0, 10, 500)
    _, Y_true = solver.saveat(spiral_dynamic, Y[0,:], saveat)
    _, Y_node = solver.saveat(net.f, Y[0,:], saveat)
    _, Y_node1 = solver.saveat(net.f, Y[-1,:], saveat - config.dataset.tmax)
     
    plt.scatter(Y[:, 0], Y[:, 1], c=T, label="noisy samples")
    plt.plot(Y_true[:, 0], Y_true[:, 1], c="green", lw=2, label="ground truth")
    plt.plot(Y_node[:, 0], Y_node[:, 1], c="black", alpha=.5, lw=1, label="Neural ODE")
    plt.plot(Y_node1[:, 0], Y_node1[:, 1], c="red", alpha=.5, lw=1,
             label="Neural ODE (from last time-step)")
    plt.colorbar()
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/neural_ode_rs.png")

    return 0


if __name__ == "__main__":
    main()
