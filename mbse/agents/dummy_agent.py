import jax.numpy as jnp
from mbse.utils.replay_buffer import Transition


class DummyAgent(object):

    def __init__(self):
        pass

    def act(self, obs: jnp.ndarray, rng=None):
        NotImplementedError

    def train_step(self,
                   rng,
                   tran: Transition,
                   ):
        NotImplementedError
