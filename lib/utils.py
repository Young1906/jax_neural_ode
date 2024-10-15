import jax
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import register_pytree_node

class State:
    def __init__(self, pytree):
        """
        grad is a pytree
        """
        self.pytree = pytree
        self.leaves, self.treedef = jax.tree_util.tree_flatten(pytree)

    def __add__(self, other):
        return jax.tree.map(lambda a, b: a + b, self, other)


    def __sub__(self, other):
        leaves = jax.tree.map(lambda a, b: a - b, self.leaves, other.leaves)
        return jax.tree_util.tree_unflatten(self.treedef, leaves)

    def __mul__(self, s):
        """
        scalar multiplication for pytree
        """
        return jax.tree.map(lambda x: x * s, self)

    def __iter__(self):
        for i in self.pytree:
            yield i

def state_flatten(s):
    child = list(s)
    aux = None
    return (child, aux)


def state_unflatten(aux, child):
    return State(child)


register_pytree_node(State, state_flatten, state_unflatten)

if __name__ == "__main__":
    a = State((1, (.5, .1))) 
    b = State((2, (1.5, .3))) 

    print(a + b)
    print(a - b)
