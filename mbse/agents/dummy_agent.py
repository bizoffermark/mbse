import jax.numpy as jnp
from mbse.utils.replay_buffer import ReplayBuffer
import numpy as np
from typing import Callable


class DummyAgent(object):

    def __init__(self, use_wandb=True, validate=False, train_steps: int = 1, batch_size: int = 256,
                 num_epochs: int = -1):
        self.use_wandb = use_wandb
        self.validate = validate
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.act_in_train = lambda obs, rng: self.act(obs, rng, eval=False)
        pass

    def act(self, obs: np.ndarray, rng=None, eval: bool = False, eval_idx: int = 0):
        return np.asarray(self.act_in_jax(jnp.asarray(obs), rng, eval=eval, eval_idx=eval_idx))

    def act_in_jax(self, obs: jnp.ndarray, rng=None, eval: bool = False, eval_idx: int = 0):
        NotImplementedError

    def train_step(self,
                   rng,
                   buffer: ReplayBuffer,
                   ):
        NotImplementedError
