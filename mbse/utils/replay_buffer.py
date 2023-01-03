import jax.numpy as jnp
import jax
from flax import struct
import numpy as np
from typing import Union
EPS = 1e-8


def merge_transitions(tran_a, tran_b):
    obs = np.concatenate([tran_a.obs, tran_b.obs], axis=0)
    action = np.concatenate([tran_a.action, tran_b.action], axis=0)
    next_obs = np.concatenate([tran_a.next_obs, tran_b.next_obs], axis=0)
    reward = np.concatenate([tran_a.reward, tran_b.reward], axis=0)
    done = np.concatenate([tran_a.done, tran_b.done], axis=0)
    return Transition(
        obs,
        action,
        next_obs,
        reward,
        done,
    )

def transition_to_jax(tran):
    return Transition(
            obs=jnp.asarray(tran.obs),
            action=jnp.asarray(tran.action),
            next_obs=jnp.asarray(tran.next_obs),
            reward=jnp.asarray(tran.reward),
            done=jnp.asarray(tran.done),
    )


@struct.dataclass
class Transition:
    obs: Union[np.ndarray, jnp.ndarray]
    action: Union[np.ndarray, jnp.ndarray]
    next_obs: Union[np.ndarray, jnp.ndarray]
    reward: Union[np.ndarray, jnp.ndarray]
    done: Union[np.ndarray, jnp.ndarray]

    @property
    def shape(self):
        return self.obs.shape[:-1]

    @property
    def shape(self):
        return self.obs.shape[:-1]


class Normalizer(object):
    def __init__(self, input_shape):
        self.mean = np.zeros(*input_shape)
        self.std = np.ones(*input_shape)

    def update(self, x):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

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
        self.obs[start:end] = transition.obs
        self.action[start:end] = transition.action
        self.next_obs[start:end] = transition.next_obs
        self.reward[start:end] = transition.reward.reshape(-1, 1)
        self.done[start:end] = transition.done.reshape(-1, 1)
        # self.obs = self.obs.at[start:end].set(transition.obs)
        # self.action = self.action.at[start:end].set(transition.action)
        # self.next_obs = self.next_obs.at[start:end].set(transition.next_obs)
        # self.reward = self.reward.at[start:end].set(transition.reward.reshape(-1, 1))
        # self.done = self.done.at[start:end].set(transition.done.reshape(-1, 1))
        self.size = min(self.size + size, self.max_size)
        if self.normalize:
            self.state_normalizer.update(self.obs[:self.size])
            # self.action_normalizer.update(self.action[:self.size])
            self.reward_normalizer.update(self.reward[:self.size])
        self.current_ptr = end % self.max_size

    def sample(self, rng, batch_size: int = 256):
        ind = jax.random.randint(rng, (batch_size,), 0, self.size)
        return transition_to_jax(
            Transition(
                self.state_normalizer.normalize(self.obs[ind]),
                self.action_normalizer.normalize(self.action[ind]),
                self.state_normalizer.normalize(self.next_obs[ind]),
                self.reward_normalizer.normalize(self.reward[ind]),
                self.done[ind],
            )
        )

    def reset(self):
        self.current_ptr = 0
        self.size = 0
        self.obs = np.zeros((self.max_size, *self.obs_shape))
        self.action = np.zeros((self.max_size, *self.action_shape))
        self.next_obs = np.zeros((self.max_size, *self.obs_shape))
        self.reward = np.zeros((self.max_size, 1))
        self.done = np.zeros((self.max_size, 1))

        self.state_normalizer = Normalizer(self.obs_shape)
        self.action_normalizer = Normalizer(self.action_shape)
        self.reward_normalizer = Normalizer((1,))
