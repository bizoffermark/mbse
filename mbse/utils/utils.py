import jax.numpy as jnp
from jax import jit
import jax
from functools import partial
from mbse.utils.replay_buffer import Transition
from mbse.models.dynamics_model import DynamicsModel
from mbse.models.reward_model import RewardModel
from mbse.agents.dummy_agent import DummyAgent
from typing import Optional, Union, Tuple, Callable

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
    batch_size = initial_state.shape[0]
    num_actions = action_sequence.shape[0]
    rewards = jnp.zeros([batch_size, num_actions])
    if rng is not None:
        rng_seq = jax.random.split(rng, num_actions + 1)
    else:
        rng_seq = [None] * (num_actions + 1)
    for i, act in enumerate(action_sequence):
        action = jnp.tile(act, (batch_size, 1))
        next_state = dynamics_model.predict(state, action, rng=rng_seq[i])
        reward = reward_model.predict(state, action, next_state)
        states = jnp.concatenate([states, next_state], axis=0)
        # rewards += reward
        rewards = rewards.at[:, i].set(reward)
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


def sample_trajectories(
    dynamics_model: DynamicsModel,
    reward_model: RewardModel,
    init_state: jnp.ndarray,
    horizon: int,
    key: jax.random.PRNGKey,
    *,
    policy: Optional[Callable] = None,
    actions: Optional[jnp.ndarray] = None,
    observations: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    TODO (yarden): document this thing.
    """
    assert policy is not None or actions is not None, \
      'Please provide policy or actor'
    use_observations = observations is not None
    use_policy = policy is not None
    batch_size = init_state.shape[0]
    def step(carry, ins):
        seed = carry[0]
        #if use_observations:
        #  obs = ins[0]
        #else:
        #  obs = carry[1]
        obs = carry[1]
        if use_policy:
          seed, actor_seed = jax.random.split(seed, 2) \
              if seed is not None else seed, None
          acs = policy(jax.lax.stop_gradient(obs), rng=actor_seed)
        else:
          acs = ins[-1]

        model_seed = None
        reward_seed = None
        if seed is not None:
            seed, model_seed = jax.random.split(seed, 2)
            seed, reward_seed = jax.random.split(seed, 2)
        next_obs = dynamics_model.predict(obs, acs, rng=model_seed)
        reward = reward_model.predict(obs, acs, rng=reward_seed)
        # if use_observations:
        #  carry = seed, obs, hidden
        #else:
        carry = [seed, next_obs]
        outs = [next_obs, reward]
        return carry, outs

    ins = []

  #if use_observations:
  #  observations = jnp.concatenate([init_state[0][:, None], observations], 1)
  #  assert observations.shape[1] == horizon
  #  ins.append(observations)
    if not use_policy:
        assert actions.shape[-2] == horizon, 'action shape must be the same as horizon'
        # actions = jnp.repeat(actions, repeats=batch_size, axis=0)
        # actions_T = jnp.expand_dims(actions_T, 2) \
        #    if len(actions_T.shape) < 3 else actions_T
        ins.append(actions)
    if ins:
    # `jax.lax.scan` scans over the first dimension, transpose the inputs.
        ins = tuple(map(lambda x: x.swapaxes(0, 1), ins))
    else:
        ins = None
    carry = [key, init_state]
    _, outs = jax.lax.scan(step, carry, ins, length=horizon)
    # Transpose back such that batch_dim is the leading dimension.
    next_state = outs[0]
    state = jnp.zeros_like(next_state)
    state = state.at[0, ...].set(init_state)
    state = state.at[1:, ...].set(next_state[:-1, ...])
    rewards = outs[1].reshape(-1, 1)

    def flatten(arr):
        new_arr = arr.reshape(-1, arr.shape[-1])
        return new_arr

    transitions = Transition(
        obs=flatten(state),
        action=flatten(actions),
        reward=rewards,
        next_obs=flatten(next_state),
        done=flatten(jnp.zeros_like(rewards)),
    )
    return transitions
    # outs = jax.tree_map(lambda x: x.swapaxes(0, 1), outs)
    # return tuple(outs)  # noqa