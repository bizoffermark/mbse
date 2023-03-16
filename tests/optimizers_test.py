from mbse.optimizers.cross_entropy_optimizer import CrossEntropyOptimizer
from mbse.optimizers.gradient_based_optimizer import GradientBasedOptimizer
import jax.numpy as jnp

EPS = 5e-2


def loss_function(x, bias):
    return -jnp.sum(jnp.square(x-bias))


opt_cls = CrossEntropyOptimizer
num_steps = 25
action_dim = (10, 2)
for bias in jnp.linspace(-5, 5, 10):
    optimizer = opt_cls(
        action_dim=action_dim,
        num_steps=num_steps,
        lr=0.1,
    )
    func = lambda x: loss_function(x, bias)
    sequence, value = optimizer.optimize(func)
    if jnp.max(jnp.abs(sequence-bias)) > EPS:
        print("Optimizer needs to be fixed")
        print("bias: ", bias)
        print("sequence: ", sequence)
    else:
        print("Optimizer works")
        print("bias: ", bias)
        print("sequence: ", sequence)





