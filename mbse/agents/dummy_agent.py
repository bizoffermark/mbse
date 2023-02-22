import jax.numpy as jnp
from mbse.utils.replay_buffer import ReplayBuffer
import numpy as np
from typing import Callable


class DummyAgent(object):

    def __init__(self, use_wandb=True, validate=False, train_steps=1, batch_size=256):
        self.use_wandb = use_wandb
        self.validate = validate
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.act_in_train = lambda obs, rng: self.act(obs, rng, eval=False)
        pass

    def act(self, obs: np.ndarray, rng=None, eval=False):
        return np.asarray(self.act_in_jax(jnp.asarray(obs), rng, eval=eval))

    def act_in_jax(self, obs: jnp.ndarray, rng=None, eval=False):
        NotImplementedError

    def train_step(self,
                   rng,
                   buffer: ReplayBuffer,
                   ):
        NotImplementedError
