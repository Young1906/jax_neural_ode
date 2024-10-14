import click
import equinox as eqx
import yaml
from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt
from functools import partial

from src.net import MLP
from src.ode.base import NeuralODEConfig
from src.ode.data import create_dataset
from src.ode.dynamics import spiral_fn
from src.ode.neural_ode import NeuralODE
from src.ode.solver import Solver, RK2

@click.command()
@click.option("--config", "-C", type=str, help="path/to/config", required=True)
def main(config: str):
    with open(config, "r") as f:
        config = yaml.safe_load(f)
        config = NeuralODEConfig(**config["params"])

    seed = jr.PRNGKey(config.seed)
    seed, mseed, dseed, bseed = jr.split(seed, 4)

    # ode solver
    solver = Solver(RK2, config.solver.h_max)

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


    # Samples
    Y, T = Ys[1, :, :], Ts[1, :]

    # True solution
    saveat = jnp.linspace(0, 10, 500)
    _, Y_true = solver.saveat(spiral_dynamic, Y[0,:], saveat)


    # NeuralODE solution
    f_theta = MLP(key=mseed, layers=config.model.mlp_layers)
    net = NeuralODE(f_theta) # NeuralODE
    net = eqx.tree_deserialise_leaves("weights/neural_ode.eqx", net)

    _, Y_node = solver.saveat(net.f, Y[0,:], saveat)

    _, Y_node1 = solver.saveat(net.f, Y[-1,:], saveat)
    

    # Visualization
    # plt.rcParams["figure.figsize"] = [10, 10]
    plt.scatter(Y[:, 0], Y[:, 1], c=T, label="noisy samples")
    plt.plot(Y_true[:, 0], Y_true[:, 1], c="green", lw=2, label="ground truth")
    plt.plot(Y_node[:, 0], Y_node[:, 1], c="black", alpha=.5, lw=1, label="Neural ODE")
    plt.plot(Y_node1[:, 0], Y_node1[:, 1], c="red", alpha=.5, lw=1, label="Neural ODE (from last time-step)")
    plt.colorbar()
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/neural_ode_sample.png")
    plt.show()




if __name__ == "__main__":
    main()
