import jax.numpy as jnp
from mbse.utils.replay_buffer import Transition
import numpy as np


class DummyAgent(object):

    def __init__(self):
        pass

    def act(self, obs: np.ndarray, rng=None, eval=False):
        return np.asarray(self.act_in_jax(jnp.asarray(obs), rng, eval=eval))

    def act_in_jax(self, obs: jnp.ndarray, rng=None, eval=False):
        NotImplementedError

    def train_step(self,
                   rng,
                   tran: Transition,
                   ):
        NotImplementedError
