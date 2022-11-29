import jax.numpy as jnp
import jax
from typing import Optional
from flax import struct

EPS = 1e-6


@struct.dataclass
class Transition:
    obs: jnp.ndarray
    action: jnp.ndarray
    next_obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray

    @property
    def shape(self):
        return self.obs.shape[:-1]

    @property
    def shape(self):
        return self.obs.shape[:-1]


class Normalizer(object):
    def __init__(self, input_shape):
        self.mean = jnp.zeros(*input_shape)
        self.std = jnp.ones(*input_shape)

    def update(self, x):
        self.mean = jnp.mean(x, axis=0)
        self.std = jnp.std(x, axis=0)

    def normalize(self, x):
        return (x-self.mean)/(self.std+EPS)

    def inverse(self, x):
        return x*self.std + self.mean


class ReplayBuffer(object):
    def __init__(self, obs_shape, action_shape, max_size: int = 1e6):
        self.max_size = max_size
        self.current_ptr = 0
        self.size = 0

        self.obs = jnp.zeros((max_size, *obs_shape))
        self.action = jnp.zeros((max_size, *action_shape))
        self.next_obs = jnp.zeros((max_size, *obs_shape))
        self.reward = jnp.zeros((max_size, 1))
        self.done = jnp.zeros((max_size, 1))

        self.state_normalizer = Normalizer(obs_shape)
        self.action_normalizer = Normalizer(action_shape)
        self.reward_normalizer = Normalizer((1, ))

    def add(self, transition: Transition):
        size = transition.shape[0]
        start = self.current_ptr
        end = self.current_ptr + size
        self.obs = self.obs.at[start:end].set(transition.obs)
        self.action = self.action.at[start:end].set(transition.action)
        self.next_obs = self.next_obs.at[start:end].set(transition.next_obs)
        self.reward = self.reward.at[start:end].set(transition.reward.reshape(-1, 1))
        self.done = self.done.at[start:end].set(transition.done.reshape(-1, 1))
        self.size = min(self.size + size, self.max_size)
        self.state_normalizer.update(self.obs[:self.size])
        # self.action_normalizer.update(self.action[:self.size])
        self.reward_normalizer.update(self.reward[:self.size])
        self.current_ptr = end % self.max_size

    def sample(self, batch_size: int = 256, rng: Optional[jnp.ndarray] = None):
        ind = jax.random.randint(rng if rng is not None else 0, (batch_size,), 0, self.size)
        return Transition(
            self.state_normalizer.normalize(self.obs[ind]),
            self.action_normalizer.normalize(self.action[ind]),
            self.state_normalizer.normalize(self.next_obs[ind]),
            self.reward_normalizer.normalize(self.reward[ind]),
            self.done[ind],
        )
