import jax.numpy as jnp
from jax import jit
import jax
from functools import partial
from mbse.utils.replay_buffer import Transition

EPS = 1e-6


@jit
def gaussian_log_likelihood(x, mu, sig):
    log_sig = jnp.log(sig + EPS)
    log_l = -0.5 * (2 * log_sig + jnp.log(2*jnp.pi)
                     + jnp.square((x - mu)/(sig + EPS)))
    log_l = jnp.sum(log_l, axis=-1)
    return log_l


@jit
def sample_normal_dist(mu, sig, rng):
    return mu + jax.random.normal(rng, mu.shape)*sig


@jit
def rbf_kernel(x, y, bandwidth=None):
  square_sum = lambda x, y: jnp.sum(jnp.square(x-y))
  pairwise = jax.vmap(lambda y: jax.vmap(lambda x: square_sum(x, y), in_axes=0, out_axes=0)(x))(y)
  n_x = x.shape[-2]
  if bandwidth is None:
      bandwidth = jnp.median(pairwise)
  bandwidth = 0.5 * bandwidth / jnp.log(n_x + 1)
  bandwidth = jnp.maximum(jax.lax.stop_gradient(bandwidth), 1e-5)
  k_xy = jnp.exp(-pairwise / bandwidth / 2)
  return k_xy


@partial(jit, static_argnums=(2, 3))
def rollout_actions(action_sequence, initial_state, dynamics_model, reward_model, rng):
    state = initial_state
    states = jnp.zeros_like(initial_state)
    rewards = jnp.zeros([1, 1])
    num_actions = action_sequence.shape[0]
    if rng is not None:
        rng_seq = jax.random.split(rng, num_actions + 1)
    else:
        rng_seq = [None] * (num_actions + 1)
    for i, act in enumerate(action_sequence):
        next_state = dynamics_model.predict(state, act, rng=rng_seq[i])
        reward = reward_model.predict(state, act, next_state)
        states = jnp.concatenate([states, next_state], axis=0)
        # rewards += reward
        rewards = jnp.concatenate([rewards, reward], axis=0)
        state = next_state
    return states, rewards


@partial(jit, static_argnums=(0, 2, 3, 5))
def rollout_policy(policy, initial_state, dynamics_model, reward_model, rng, num_steps=10):
    state = initial_state
    state_shape = (num_steps + 1, ) + initial_state.shape
    states = jnp.zeros(state_shape)
    states = states.at[0].set(initial_state)
    reward_shape = (num_steps, ) + (initial_state.shape[0], )
    rewards = jnp.zeros(reward_shape)
    dones = jnp.zeros(reward_shape, dtype=jnp.int8)
    test_act = policy(state)
    actions_shape = (num_steps, ) + test_act.shape
    actions = jnp.zeros(actions_shape)
    if rng is not None:
        rng_seq = jax.random.split(rng, num_steps + 1)
    else:
        rng_seq = [None] * (num_steps + 1)
    for i in range(num_steps):
        act_rng, obs_rng = jax.random.split(rng_seq[i], 2)
        act = policy(state, act_rng)
        next_state = dynamics_model.predict(state, act, rng=obs_rng)
        reward = reward_model.predict(state, act, next_state)
        states = states.at[i+1].set(next_state)
        actions = actions.at[i].set(act)
        rewards = rewards.at[i].set(reward)
        state = next_state
    next_states = states[1:, ...]
    states = states[:-1, ...]

    def flatten(arr):
        new_arr = arr.reshape(-1, arr.shape[-1])
        return new_arr

    transitions = Transition(
        obs=flatten(states),
        action=flatten(actions),
        reward=flatten(rewards),
        next_obs=flatten(next_states),
        done=flatten(dones),
    )
    return transitions
