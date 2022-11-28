from mbse.optimizers.cross_entropy_optimizer import CrossEntropyOptimizer
import jax.numpy as jnp

EPS = 5e-2


def loss_function(x, bias):
    return -jnp.square(x-bias)


for bias in jnp.linspace(-5, 5, 10):
    optimizer = CrossEntropyOptimizer(
        func=lambda x: loss_function(x, bias),
        action_dim=(1,),
        num_steps=10,
    )
    sequence = optimizer.optimize()
    if jnp.abs(sequence-bias) > EPS:
        print("Optimizer needs to be fixed")
        print("bias: ", bias)
        print("sequence: ", sequence)




