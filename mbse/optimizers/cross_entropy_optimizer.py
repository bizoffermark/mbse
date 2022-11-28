from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap


class CrossEntropyOptimizer(object):

    def __init__(
            self,
            func,
            action_dim,
            num_samples=500,
            num_elites=50,
            num_steps=100,
            seed=0,
    ):

        self.func = func
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_steps = num_steps
        self.action_dim = action_dim
        self.mean = jnp.zeros(action_dim)
        self.std = jnp.ones(action_dim)
        self.seed = seed

    @partial(jit, static_argnums=0)
    def step(self):
        key = jax.random.PRNGKey(self.seed)
        samples = self.mean + jax.random.multivariate_normal(
            key=key,
            mean=self.mean,
            cov=jnp.diag(jnp.square(self.std)),
            shape=(self.num_samples, ))

        values = vmap(self.func, 0, 0)(samples)

        best_elite_idx = np.argsort(-values)[:self.num_elites]

        elites = samples[best_elite_idx]
        elite_values = values[best_elite_idx]
        return elites, elite_values

    def optimize(self):
        best_value = -jnp.inf
        best_sequence = self.mean
        for i in range(self.num_steps):
            elites, elite_values = self.step()
            self.mean = jnp.mean(elites, axis=0)
            self.std = jnp.std(elites, axis=0)
            best_elite = elite_values[0]
            if best_value <= best_elite:
                best_value = best_elite
                best_sequence = elites[0]

        return best_sequence

    def reset(self):
        self.mean = jnp.zeros(self.action_dim)
        self.std = jnp.ones(self.action_dim)



