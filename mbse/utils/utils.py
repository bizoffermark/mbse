import jax.numpy as jnp
from jax import jit
import jax
from functools import partial
from mbse.utils.type_aliases import ModelProperties, PolicyProperties
from mbse.utils.replay_buffer import Transition
from typing import Optional, Union, Callable

EPS = 1e-6


@jit
def gaussian_log_likelihood(x, mu, sig):
    log_sig = jnp.log(sig + EPS)
    log_l = -0.5 * (2 * log_sig + jnp.log(2 * jnp.pi)
                    + jnp.square((x - mu) / (sig + EPS)))
    log_l = jnp.sum(log_l, axis=-1)
    return log_l


@jit
def sample_normal_dist(mu, sig, rng):
    return mu + jax.random.normal(rng, mu.shape) * sig


@jit
def rbf_kernel(x, y, bandwidth=None):
    square_sum = lambda x, y: jnp.sum(jnp.square(x - y))
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
    state_shape = (num_steps + 1,) + initial_state.shape
    states = jnp.zeros(state_shape)
    states = states.at[0].set(initial_state)
    reward_shape = (num_steps,) + (initial_state.shape[0],)
    rewards = jnp.zeros(reward_shape)
    dones = jnp.zeros(reward_shape, dtype=jnp.int8)
    test_act = policy(state)
    actions_shape = (num_steps,) + test_act.shape
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
        states = states.at[i + 1].set(next_state)
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


@partial(jax.jit, static_argnums=(0, 3, 5))
def sample_trajectories(
        evaluate_fn: Callable,
        parameters,
        init_state: jnp.ndarray,
        horizon: int,
        key: Optional[jax.random.PRNGKey],
        policy: Optional[Callable] = None,
        actor_params=None,
        actions: Optional[jnp.ndarray] = None,
        model_props: ModelProperties = ModelProperties(),
        sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
        policy_props: PolicyProperties = PolicyProperties(),
        # observations: Optional[jnp.ndarray] = None
) -> Transition:
    """
    TODO (yarden): document this thing.
    """
    assert policy is not None or actions is not None, \
        'Please provide policy or actor'
    # use_observations = observations is not None
    use_policy = policy is not None

    # batch_size = init_state.shape[0]
    def step(carry, ins):
        seed = carry[0]
        # if use_observations:
        #  obs = ins[0]
        # else:
        #  obs = carry[1]
        obs = carry[1]
        if use_policy:
            seed, actor_seed = jax.random.split(seed, 2)
            normalized_obs = (jax.lax.stop_gradient(obs) - policy_props.policy_bias_obs) / \
                             (policy_props.policy_scale_obs + EPS)
            acs = policy(actor_params=actor_params, obs=normalized_obs, rng=actor_seed)
        else:
            acs = ins[-1]
        model_seed = None
        if seed is not None:
            seed, model_seed = jax.random.split(seed, 2)
        next_obs, reward = evaluate_fn(
            parameters=parameters,
            obs=obs,
            action=acs,
            rng=model_seed,
            sampling_idx=sampling_idx,
            model_props=model_props,
        )
        # if use_observations:
        #  carry = seed, obs, hidden
        # else:
        carry = [seed, next_obs]
        outs = [next_obs, reward, acs]
        return carry, outs

    ins = []

    # if use_observations:
    #  observations = jnp.concatenate([init_state[0][:, None], observations], 1)
    #  assert observations.shape[1] == horizon
    #  ins.append(observations)
    if not use_policy:
        # assert actions.shape[-2] == horizon, 'action shape must be the same as horizon'
        # actions = jnp.repeat(actions, repeats=batch_size, axis=0)
        # actions_T = jnp.expand_dims(actions_T, 2) \
        #    if len(actions_T.shape) < 3 else actions_T
        ins.append(actions)
    # if ins:
    # `jax.lax.scan` scans over the first dimension, transpose the inputs.
    #    ins = tuple(map(lambda x: x.swapaxes(0, 1), ins))
    # else:
    #    ins = None
    carry = [key, init_state]
    _, outs = jax.lax.scan(step, carry, ins, length=horizon)
    # Transpose back such that batch_dim is the leading dimension.
    next_state = outs[0]
    state = jnp.zeros_like(next_state)
    state = state.at[0, ...].set(init_state)
    state = state.at[1:, ...].set(next_state[:-1, ...])
    rewards = outs[1].reshape(-1, 1)
    acs = outs[-1]

    def flatten(arr):
        new_arr = arr.reshape(-1, arr.shape[-1])
        return new_arr

    transitions = Transition(
        obs=flatten(state),
        action=flatten(acs),
        reward=rewards,
        next_obs=flatten(next_state),
        done=flatten(jnp.zeros_like(rewards)),
    )
    return transitions
    # outs = jax.tree_map(lambda x: x.swapaxes(0, 1), outs)
    # return tuple(outs)  # noqa


def tree_stack(trees, axis=0):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l, axis=axis) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def get_idx(tree, idx):
    return jax.tree_util.tree_map(lambda x: x[idx], tree)


def convert_to_jax(x):
    if not isinstance(x, jax.Array):
        return jnp.asarray(x)
    else:
        return x

def normalize(x, mu_x, std_x, eps=1e-8):
    return (x - mu_x)/(std_x + eps)

def denormalize(x, mu_x, std_x, eps=1e-8):
    return (std_x + eps)*x + mu_x

def moving_average(x, window_size):
    """Compute the moving average of a sequence using a sliding window."""
    x = jnp.concatenate((jnp.array([0.0]*(window_size-1)),x))
    weights = jnp.repeat(1.0, window_size) / window_size
    return jnp.convolve(x, weights, mode='valid')

def states2ind_states(states):
    thetas = states[:,12]
    theta_dots = states[:,13]
    p_pivots = states[:,14:17]
    v_ees = states[:,17:20]
    p_balls = states[:,20:23]
    mocap_valids = states[:,23]
    states = states[:,0:12]
    return states, thetas, theta_dots, p_pivots, v_ees, p_balls, mocap_valids

import json

def get_data_jax(fpath, moving_win=None):
    ts = []
    states = []
    actions = []
    with open(fpath, "r") as f:
        for line in f.readlines():
            data =json.loads(line)
            ts.append(data[0]["t"])
            states.append(data[1]['state'])
            actions.append(data[2]['input'])

    ts = jnp.array(ts)
    states = jnp.array(states)
    actions = jnp.array(actions)

    #states, thetas, theta_dots, p_pivots, v_ees, p_balls, mocap_valids = states2ind_states(states)
    states = states[:,:,0]
    actions = actions[:,0,0]
    print(states.shape)
    print(actions.shape)
    states, thetas, theta_dots, p_pivots, v_ees, p_balls, mocap_valids = states2ind_states(states)
    #states = [thetas.reshape(thetas.shape[0],-1), theta_dots.reshape(theta_dots.shape[0],-1), p_pivots[:,0], v_ees[:,0]]
    states = [thetas, theta_dots, p_pivots[:,0], v_ees[:,0]]

    # reshape tensors
    states = jnp.concatenate([jnp.expand_dims(state,-1) for state in states],1)
    actions = jnp.expand_dims(actions,-1)
    states_chunk = jnp.array([])
    ts = jnp.expand_dims(ts, -1)
    inds = [0] + jnp.where(mocap_valids < 1.0)[0].tolist() + [len(mocap_valids)]
    idxes = [(inds[i], inds[i+1]) for i in range(len(inds)-1)]
    # TODO: Check if this makes big difference in performance
    xs = jnp.array([])
    ys = jnp.array([])
    us = jnp.array([])

    for i, idx in enumerate(idxes):
        state_chunk = states[idx[0]:idx[1]]
        action_chunk = actions[idx[0]:idx[1]]
        ts_chunk = ts[idx[0]:idx[1]]

        if moving_win is not None:
            state_chunk = state_chunk.at[:,1].set(moving_average(state_chunk[:,1], moving_win))
            state_chunk = state_chunk.at[:,3].set(moving_average(state_chunk[:,3], moving_win))
            action_chunk = action_chunk.at[:,0].set(moving_average(action_chunk[:,0], moving_win))
            
        x = state_chunk[:-1]
        y = state_chunk[1:] - state_chunk[:-1]
        y = y.at[:,0].set(jnp.arctan2(jnp.sin(y[:,0]), jnp.cos(y[:,0]))) # normalize the angle difference
        u = action_chunk[:-1]
        x = jnp.concatenate([x, u], 1)

        print("action_chunk.shape: {}".format(action_chunk.shape))

        if i == 0:
            states_chunk = state_chunk
            #action_chunk = action_chunk
            #tss_chunk
            xs = x
            ys = y
            us = u
        else:
            states_chunk = jnp.append(states_chunk, state_chunk, axis=0)
            #actions_chunk = jnp.append(actions_chunk, action_chunk, axis=0)
            #ts_chunk = jnp.append(ts_chunk, ts_chunk, axis=0)

            xs = jnp.append(xs, x, axis=0)
            ys = jnp.append(ys, y, axis=0)
            us = jnp.append(us, u, axis=0)

        # x = states[:-1]
        # y = states[1:] - states[:-1]
        # u = actions[:-1]
        # x = jnp.concatenate([x, u], 1) 
        # u = actions
    states=states_chunk
    # print((mocap_valids>0).sum())
    print("xs.shape: ", xs.shape)
    print("ys.shape: ", ys.shape)
    print("us.shape: ", us.shape)
    return states, xs, ys, us, ts