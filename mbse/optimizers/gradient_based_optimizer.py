from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from mbse.optimizers.dummy_optimizer import DummyOptimizer
import optax


class GradientBasedOptimizer(DummyOptimizer):

    def __init__(
            self,
            func,
            action_dim,
            num_samples=50,
            num_steps=20,
            seed=0,
            init_var=5,
            lr=1e-3,
            *args,
            **kwargs,
    ):
        super(GradientBasedOptimizer, self).__init__()
        self.func = func
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.action_dim = action_dim
        self.seed = seed
        self.init_var = init_var
        self.optimizer = optax.adam(learning_rate=lr)
        key = jax.random.PRNGKey(self.seed)
        mean = jnp.zeros(self.action_dim)
        std = jnp.ones(self.action_dim) * self.init_var
        samples = mean + jax.random.multivariate_normal(
            key=key,
            mean=jnp.zeros_like(mean),
            cov=jnp.diag(jnp.ones_like(mean)),
            shape=(self.num_samples,)) * std

        optimizer_state = self.optimizer.init(samples)
        self.optimizer_state = optimizer_state
        self.init_samples = samples

    @partial(jit, static_argnums=0)
    def step(self, samples, opt_state):
        def loss_fn(action):
            values = vmap(self.func)(action)
            return -values.mean(), values
        (loss, values), grads = jax.value_and_grad(loss_fn, has_aux=True)(samples)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, samples)
        new_samples = optax.apply_updates(samples, updates)
        return values, new_samples, new_opt_state

    @partial(jit, static_argnums=0)
    def optimize(self):
        best_value = -jnp.inf
        samples = self.init_samples
        opt_state = self.optimizer_state
        best_sequence = jnp.mean(samples, axis=0).reshape(self.action_dim)

        def get_best_seq(val, seq):
            best_elite_idx = np.argsort(val, axis=0).squeeze()[-1]
            elite = seq[best_elite_idx]
            best_val = val[best_elite_idx].squeeze()
            return [best_val, elite]
        get_best_action = lambda best_val, best_seq, val, seq: get_best_seq(val, seq)
        get_curr_best_action = lambda best_val, best_seq, val, seq: [best_val, best_seq]
        for i in range(self.num_steps):
            values, samples, opt_state = self.step(samples, opt_state)
            bests = jax.lax.cond(best_value <= jnp.max(values),
                                 get_best_action,
                                 get_curr_best_action,
                                 best_value,
                                 best_sequence,
                                 values,
                                 samples)
            best_sequence = bests[-1]
            best_value = bests[0]
        return best_sequence