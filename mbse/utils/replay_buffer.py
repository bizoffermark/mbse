import jax.numpy as jnp
import jax
from flax import struct

EPS = 1e-8


def merge_transitions(tran_a, tran_b):
    obs = jnp.concatenate([tran_a.obs, tran_b.obs], axis=0)
    action = jnp.concatenate([tran_a.action, tran_b.action], axis=0)
    next_obs = jnp.concatenate([tran_a.next_obs, tran_b.next_obs], axis=0)
    reward = jnp.concatenate([tran_a.reward, tran_b.reward], axis=0)
    done = jnp.concatenate([tran_a.done, tran_b.done], axis=0)
    return Transition(
        obs,
        action,
        next_obs,
        reward,
        done,
    )


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
    def __init__(self, obs_shape, action_shape, max_size: int = 1e6, normalize=False):
        self.max_size = max_size
        self.current_ptr = 0
        self.size = 0
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.normalize = normalize
        self.obs, self.action, self.next_obs, self.reward, self.done = None, None, None, None, None
        self.state_normalizer, self.action_normalizer, self.reward_normalizer = None, None, None
        self.reset()

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
        if self.normalize:
            self.state_normalizer.update(self.obs[:self.size])
            # self.action_normalizer.update(self.action[:self.size])
            self.reward_normalizer.update(self.reward[:self.size])
        self.current_ptr = end % self.max_size

    def sample(self, rng, batch_size: int = 256):
        ind = jax.random.randint(rng, (batch_size,), 0, self.size)
        return Transition(
            self.state_normalizer.normalize(self.obs[ind]),
            self.action_normalizer.normalize(self.action[ind]),
            self.state_normalizer.normalize(self.next_obs[ind]),
            self.reward_normalizer.normalize(self.reward[ind]),
            self.done[ind],
        )

    def reset(self):
        self.current_ptr = 0
        self.size = 0
        self.obs = jnp.zeros((self.max_size, *self.obs_shape))
        self.action = jnp.zeros((self.max_size, *self.action_shape))
        self.next_obs = jnp.zeros((self.max_size, *self.obs_shape))
        self.reward = jnp.zeros((self.max_size, 1))
        self.done = jnp.zeros((self.max_size, 1))

        self.state_normalizer = Normalizer(self.obs_shape)
        self.action_normalizer = Normalizer(self.action_shape)
        self.reward_normalizer = Normalizer((1,))
