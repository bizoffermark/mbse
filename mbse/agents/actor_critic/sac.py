from functools import partial

import numpy as np
from typing import Sequence, Callable, Optional
import optax
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import jit, vmap, random, value_and_grad
from mbse.utils.network_utils import MLP, mse
from mbse.utils.replay_buffer import Transition
from flax import struct
from copy import deepcopy
import gym

EPS = 1e-6


@jit
def sample_normal_dist(mu, sig, rng):
    return mu + jax.random.normal(rng, mu.shape)*sig


@jit
def gaussian_log_likelihood(x, mu, sig):
    log_sig = jnp.log(sig)
    log_l = -0.5 * (2 * log_sig + jnp.log(2*jnp.pi)
                     + jnp.square((x - mu)/(sig + EPS)))
    return jnp.sum(log_l, axis=-1)


# Perform Polyak averaging provided two network parameters and the averaging value tau.
@jit
def soft_update(
    target_params, online_params, tau=0.005
):
    return jax.tree_util.tree_map(
        lambda x, y: (1 - tau) * x + tau * y, target_params, online_params
    )


@jit
def squash_action(action):
    squashed_action = nn.tanh(action)
    # sanity check to clip between -1 and 1
    squashed_action = jnp.clip(squashed_action, -1, 1)
    return squashed_action


@partial(jit, static_argnums=(0, ))
def get_squashed_log_prob(actor_fn, params, obs, rng):
    mu, sig = actor_fn(params, obs)
    u = sample_normal_dist(mu, sig, rng)
    log_l = gaussian_log_likelihood(u, mu, sig)
    a = squash_action(u)
    log_l -= jnp.sum(
        jnp.log((1 - jnp.square(a)) + EPS), axis=-1
    )
    return a, log_l.reshape(-1, 1)


@partial(jit, static_argnums=(0, 1))
def get_soft_td_target(
        critic_fn,
        actor_fn,
        next_obs,
        reward,
        not_done,
        critic_target_params,
        actor_params,
        alpha,
        discount,
        rng,
        reward_scale=1.0,
):

    # Sample action from Pi to simulate expectation
    next_action, next_log_a = get_squashed_log_prob(
        actor_fn=actor_fn,
        params=actor_params,
        obs=next_obs,
        rng=rng,
    )
    # Get predictions for both Q functions and take the min
    target_q1, target_q2 = critic_fn(critic_target_params, obs=next_obs, action=next_action)
    # Soft target update
    target_q = jnp.minimum(target_q1, target_q2)
    target_q = target_q - alpha * next_log_a
    # Td target = r_t + \gamma V_{t+1}
    target_q = reward_scale*reward + not_done * discount * target_q
    return target_q


@partial(jit, static_argnums=(0, ))
def get_action(actor_fn, actor_params, obs, rng=None):
    mu, sig = actor_fn(actor_params, obs)
    if rng is None:
        action = mu
    else:
        action = sample_normal_dist(mu, sig, rng)
    return squash_action(action)


@partial(jit, static_argnums=(0, 1, 3))
def update_actor(
        actor_fn,
        critic_fn,
        actor_params,
        actor_update_fn,
        actor_opt_state,
        critic_params,
        alpha,
        obs,
        rng
):
    def loss(params):
        action, log_l = get_squashed_log_prob(actor_fn, params, obs, rng)
        q1, q2 = critic_fn(critic_params, obs=obs, action=action)
        min_q = jnp.minimum(q1, q2)
        actor_loss = - min_q + alpha*log_l
        return jnp.mean(actor_loss), log_l

    (loss, log_a), grads = value_and_grad(loss, has_aux=True)(actor_params)
    updates, new_actor_opt_state = actor_update_fn(grads, actor_opt_state, params=actor_params)
    new_actor_params = optax.apply_updates(actor_params, updates)
    grad_norm = optax.global_norm(grads)
    return new_actor_params, new_actor_opt_state, loss, log_a, grad_norm


@partial(jit, static_argnums=(0, 1))
def update_critic(
        critic_fn,
        critic_update_fn,
        critic_params,
        critic_opt_state,
        obs,
        action,
        target_q
):
    def loss(params):
        q1, q2 = critic_fn(params, obs, action)
        critic_loss = 0.5 * (mse(q1, target_q) + mse(q2, target_q))
        return jnp.mean(critic_loss)
    loss, grads = value_and_grad(loss)(critic_params)
    updates, new_critic_opt_state = critic_update_fn(grads, critic_opt_state, params=critic_params)
    new_critic_params = optax.apply_updates(critic_params, updates)
    grad_norm = optax.global_norm(grads)
    return new_critic_params, new_critic_opt_state, loss, grad_norm


@partial(jit, static_argnums=(0, 3))
def update_alpha(log_alpha_fn, alpha_params, alpha_opt_state, alpha_update_fn, log_a, target_entropy):
    log_a = jax.lax.stop_gradient(log_a + target_entropy)

    def loss(params):
        # @vmap
        def alpha_loss_fn(lp):
            return -(
                    log_alpha_fn(params) * lp
            ).mean()
        return alpha_loss_fn(log_a)
    loss, grads = value_and_grad(loss)(alpha_params)
    updates, new_alpha_opt_state = alpha_update_fn(grads, alpha_opt_state, params=alpha_params)
    new_alpha_params = optax.apply_updates(alpha_params, updates)
    grad_norm = optax.global_norm(grads)
    return new_alpha_params, new_alpha_opt_state, loss, grad_norm


@struct.dataclass
class SACModelSummary:
    actor_loss: jnp.ndarray
    entropy: jnp.ndarray
    critic_loss: jnp.ndarray
    alpha_loss: jnp.ndarray
    log_alpha: jnp.ndarray
    actor_std: jnp.ndarray
    critic_grad_norm: jnp.ndarray
    actor_grad_norm: jnp.ndarray
    alpha_grad_norm: jnp.ndarray


class Actor(nn.Module):

    features: Sequence[int]
    action_dim: int
    non_linearity: Callable = nn.relu
    sig_min: float = 1e-3
    sig_max: float = 1e8

    @nn.compact
    def __call__(self, obs):
        actor_net = MLP(self.features,
                        2*self.action_dim,
                        self.non_linearity)

        out = actor_net(obs)
        mu, sig = jnp.split(out, 2, axis=-1)
        sig = nn.softplus(sig)
        sig = jnp.clip(sig, 0, self.sig_max) + self.sig_min
        return mu, sig


class Critic(nn.Module):

    features: Sequence[int]
    non_linearity: Callable = nn.relu

    @nn.compact
    def __call__(self, obs, action, train=False):
        critic_1 = MLP(features=self.features,
                         output_dim=1,
                         non_linearity=self.non_linearity)

        critic_2 = MLP(features=self.features,
                            output_dim=1,
                            non_linearity=self.non_linearity)
        obs_action = jnp.concatenate((obs, action), -1)
        value_1 = critic_1(obs_action)
        value_2 = critic_2(obs_action)
        return value_1, value_2


class ConstantModule(nn.Module):

    def setup(self):
        self.const = self.param("log_alpha", nn.ones, (1, ))

    def __call__(self):
        return self.const


class SACAgent(object):

    def __init__(
            self,
            action_space: gym.spaces.box,
            observation_space: gym.spaces.box,
            discount: float = 0.99,
            lr_actor: float = 1e-3,
            weight_decay_actor: float = 1e-5,
            lr_critic: float = 1e-3,
            weight_decay_critic: float = 1e-5,
            lr_alpha: float = 1e-3,
            weight_decay_alpha: float = 0.0,
            actor_features: Sequence[int] = [256, 256],
            critic_features: Sequence[int] = [256, 256, 256, 256],
            target_entropy: Optional[float] = None,
            rng: jax.Array = random.PRNGKey(0),
            q_update_frequency: int = 1,
            scale_reward: float = 1,
            tau: float = 0.05,
    ):
        action_dim = np.prod(action_space.shape)
        sample_obs = observation_space.sample()
        sample_act = action_space.sample()
        self.obs_sample = sample_obs
        self.act_sample = sample_act
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_alpha = lr_alpha
        self.discount = discount
        self.target_entropy = float(-action_dim) if target_entropy is None else target_entropy
        self.actor_optimizer = optax.adamw(learning_rate=lr_actor, weight_decay=weight_decay_actor)
        self.critic_optimizer = optax.adamw(learning_rate=lr_critic, weight_decay=weight_decay_critic)
        self.alpha_optimizer = optax.adamw(learning_rate=lr_alpha, weight_decay=weight_decay_alpha)
        self.actor = Actor(features=actor_features, action_dim=action_dim)
        self.critic = Critic(features=critic_features)
        self.log_alpha = ConstantModule()
        self.critic_features = critic_features

        rng, actor_rng, critic_rng, alpha_rng = random.split(rng, 4)
        actor_params = self.actor.init(actor_rng, sample_obs)
        # print(self.actor.tabulate(actor_rng, sample_obs))
        actor_opt_state = self.actor_optimizer.init(actor_params)
        self.actor_params = actor_params
        self.actor_opt_state = actor_opt_state

        critic_params = self.critic.init(
            critic_rng, sample_obs, sample_act
        )
        target_critic_params = deepcopy(critic_params)
        critic_opt_state = self.critic_optimizer.init(critic_params)
        # print(self.critic.tabulate(critic_rng, sample_obs, sample_act))
        self.critic_params = critic_params
        self.target_critic_params = target_critic_params
        self.critic_opt_state = critic_opt_state

        alpha_params = self.log_alpha.init(alpha_rng)
        # print(self.log_alpha.tabulate(alpha_rng, self.initial_log_alpha))
        alpha_opt_state = self.alpha_optimizer.init(alpha_params)
        self.alpha_params = alpha_params
        self.alpha_opt_state = alpha_opt_state
        self.q_update_frequency = q_update_frequency
        self.scale_reward = scale_reward
        self.tau = tau

    def act(self, obs, rng=None):
        return get_action(self.actor.apply, self.actor_params, obs, rng)

    def train_step(
            self,
            rng,
            tran: Transition,
    ):
        rng, actor_rng, td_rng = random.split(rng, 3)

        alpha = jax.lax.stop_gradient(
            jnp.exp(
                self.log_alpha.apply(self.alpha_params)
            )
        )
        for u in range(self.q_update_frequency):
            td_rng, target_q_rng = random.split(td_rng, 2)
            target_q = jax.lax.stop_gradient(
                get_soft_td_target(
                    critic_fn=self.critic.apply,
                    actor_fn=self.actor.apply,
                    next_obs=tran.next_obs,
                    reward=tran.reward,
                    not_done=1 - tran.done,
                    critic_target_params=self.target_critic_params,
                    actor_params=self.actor_params,
                    alpha=alpha,
                    discount=self.discount,
                    rng=target_q_rng,
                )
            )
            critic_params, critic_opt_state, critic_loss, critic_grad_norm = update_critic(
                critic_fn=self.critic.apply,
                critic_params=self.critic_params,
                critic_opt_state=self.critic_opt_state,
                critic_update_fn=self.critic_optimizer.update,
                obs=tran.obs,
                action=tran.action,
                target_q=target_q,
            )

            self.target_critic_params = soft_update(self.target_critic_params, critic_params, self.tau)
            self.critic_params = critic_params
            self.critic_opt_state = critic_opt_state

        actor_params, actor_opt_state, actor_loss, log_a, actor_grad_norm = update_actor(
            actor_fn=self.actor.apply,
            critic_fn=self.critic.apply,
            actor_params=self.actor_params,
            actor_update_fn=self.actor_optimizer.update,
            critic_params=self.critic_params,
            alpha=alpha,
            actor_opt_state=self.actor_opt_state,
            obs=tran.obs,
            rng=actor_rng,
        )

        alpha_params, alpha_opt_state, alpha_loss, alpha_grad_norm = update_alpha(
            log_alpha_fn=self.log_alpha.apply,
            alpha_params=self.alpha_params,
            alpha_opt_state=self.alpha_opt_state,
            alpha_update_fn=self.alpha_optimizer.update,
            log_a=log_a,
            target_entropy=self.target_entropy
        )

        self.actor_params = actor_params
        self.actor_opt_state = actor_opt_state
        self.alpha_params = alpha_params
        self.alpha_opt_state = alpha_opt_state

        log_alpha = self.log_alpha.apply(self.alpha_params)
        _, std = self.actor.apply(self.actor_params, tran.obs)

        summary = SACModelSummary(
            actor_loss=actor_loss,
            entropy=-log_a.mean(),
            actor_std=std.mean(),
            critic_loss=critic_loss,
            alpha_loss=alpha_loss,
            log_alpha=log_alpha,
            critic_grad_norm=critic_grad_norm,
            actor_grad_norm=actor_grad_norm,
            alpha_grad_norm=alpha_grad_norm,
        )

        return summary











