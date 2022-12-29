import jax.numpy as jnp
from typing import Optional


class DummyOptimizer(object):
    def __init__(self,
                 action_dim=(1, ),
                 upper_bound: Optional[jnp.ndarray] = None,
                 num_steps: int = 20,
                 lower_bound: Optional[jnp.ndarray] = None,
                 *args,
                 **kwargs):

        self.num_steps = num_steps
        self.action_dim = action_dim
        if upper_bound is None:
            upper_bound = jnp.ones(self.action_dim)*jnp.inf
        self.upper_bound = upper_bound
        if lower_bound is None:
            self.lower_bound = - upper_bound
        else:
            self.lower_bound = lower_bound

    def optimize(self, func):
        pass

    def clip_action(self, action):
        return jnp.clip(action, a_min=self.lower_bound, a_max=self.upper_bound)
