from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from mbse.optimizers.dummy_optimizer import DummyOptimizer


class CrossEntropyOptimizer(DummyOptimizer):

    def __init__(
            self,
            num_samples=500,
            num_elites=50,
            seed=0,
            init_var=5,
            *args,
            **kwargs,
    ):
        super(CrossEntropyOptimizer, self).__init__(*args, **kwargs)
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.seed = seed
        self.init_var = init_var

    @partial(jit, static_argnums=(0, 1))
    def step(self, func, mean, std, key):
        mean = mean.reshape(-1, 1).squeeze()
        std = std.reshape(-1, 1).squeeze()
        samples = mean + jax.random.multivariate_normal(
            key=key,
            mean=jnp.zeros_like(mean),
            cov=jnp.diag(jnp.ones_like(mean)),
            shape=(self.num_samples, ))*std
        samples = samples.reshape((self.num_samples,) + self.action_dim)
        samples = self.clip_action(samples)
        values = vmap(func)(samples)

        best_elite_idx = np.argsort(values, axis=0).squeeze()[-self.num_elites:]

        elites = samples[best_elite_idx]
        elite_values = values[best_elite_idx]
        return elites, elite_values

    @partial(jit, static_argnums=(0, 1))
    def optimize(self, func, rng=None):
        best_value = -jnp.inf
        mean = jnp.zeros(self.action_dim)
        std = jnp.ones(self.action_dim)*self.init_var
        best_sequence = mean
        get_best_action = lambda best_val, best_seq, val, seq: [val[0].squeeze(), seq[0]]
        get_curr_best_action = lambda best_val, best_seq, val, seq: [best_val, best_seq]
        if rng is None:
            rng = jax.random.PRNGKey(self.seed)

        def step(carry, ins):
            key = carry[0]
            mu = carry[1]
            sig = carry[2]
            best_val = carry[3]
            best_seq = carry[4]
            key, sample_key = jax.random.split(key, 2)
            elites, elite_values = self.step(func, mu, sig, sample_key)
            mean = jnp.mean(elites, axis=0)
            std = jnp.std(elites, axis=0)
            best_elite = elite_values[0].squeeze()
            bests = jax.lax.cond(best_val <= best_elite,
                                 get_best_action,
                                 get_curr_best_action,
                                 best_val,
                                 best_seq,
                                 elite_values,
                                 elites)
            best_val = bests[0]
            best_seq = bests[-1]

            outs = [best_val, best_seq]
            carry = [key, mean, std, best_val, best_seq]

            return carry, outs
        carry = [rng, mean, std, best_value, best_sequence]
        carry, outs = jax.lax.scan(step, carry, xs=None, length=self.num_steps)
        return outs[1][-1, ...], outs[0][-1, ...]


