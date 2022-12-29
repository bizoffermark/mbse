import jax.numpy as jnp
from mbse.utils.replay_buffer import Transition
import numpy as np


class DummyAgent(object):

    def __init__(self):
        pass

    def act(self, obs: jnp.ndarray, rng=None):
        return np.asarray(self.act_in_jax(obs, rng))

    def act_in_jax(self, obs: jnp.ndarray, rng=None):
        NotImplementedError

    def train_step(self,
                   rng,
                   tran: Transition,
                   ):
        NotImplementedError
