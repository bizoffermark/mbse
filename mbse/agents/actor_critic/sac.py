from functools import partial

import numpy as np
from typing import Sequence, Callable, Optional
import optax
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import jit, vmap, random, value_and_grad
from mbse.utils.network_utils import MLP, mse
from mbse.utils.replay_buffer import ReplayBuffer, Transition
from flax import struct
from copy import deepcopy
import gym
from mbse.utils.utils import gaussian_log_likelihood, sample_normal_dist
from mbse.agents.dummy_agent import DummyAgent
import wandb

EPS = 1e-6
ZERO = 0.0


# Perform Polyak averaging provided two network parameters and the averaging value tau.
@partial(jit, static_argnums=(2, ))
def soft_update(
    target_params, online_params, tau=0.005
):
    updated_params = jax.tree_util.tree_map(
        lambda old, new: (1 - tau) * old + tau * new, target_params, online_params
    )
    return updated_params


@jit
def squash_action(action):
    squashed_action = nn.tanh(action)
    # sanity check to clip between -1 and 1
    squashed_action = jnp.clip(squashed_action, -0.999, 0.999)
    return squashed_action


@partial(jit, static_argnums=(0, ))
def get_squashed_log_prob(actor_fn, params, obs, rng):
    mu, sig = actor_fn(params, obs)
    u = sample_normal_dist(mu, sig, rng)
    log_l = gaussian_log_likelihood(u, mu, sig)
    a = squash_action(u)
    log_l -= jnp.sum(
        jnp.log((1 - jnp.square(a))), axis=-1
    )
    return a, log_l.reshape(-1, 1)


@partial(jit, static_argnums=(0, 1, 10, 12))
def get_soft_td_target(
        critic_fn,
        actor_fn,
        next_obs,
        obs,
        reward,
        not_done,
        critic_target_params,
        critic_params,
        actor_params,
        alpha,
        discount,
        rng,
        reward_scale=1.0,
):

    # Sample action from Pi to simulate expectation
    # next_action, next_log_a = get_squashed_log_prob(
    #    actor_fn=actor_fn,
    #    params=actor_params,
    #    obs=next_obs,
    #    rng=next_rng,
    # )
    current_action, current_log_a = get_squashed_log_prob(
        actor_fn=actor_fn,
        params=actor_params,
        obs=obs,
        rng=rng,
    )

    # Get predictions for both Q functions and take the min
    _, _, target_v = critic_fn(critic_target_params, obs=next_obs,
                               action=jnp.zeros_like(current_action))
    target_v_term = target_v.mean()
    target_q1, target_q2, _ = critic_fn(critic_params, obs=obs, action=current_action)
    # Soft target update
    target_q = jnp.minimum(target_q1, target_q2)

    entropy_term = - alpha * current_log_a

    target_q_term = target_q.mean()
    target_q = target_q + entropy_term

    # Td target = r_t + \gamma V_{t+1}
    target_v = reward_scale*reward + not_done * discount * target_v
    return target_v, target_q, target_v_term, target_q_term, entropy_term.mean()


@partial(jit, static_argnums=(0, ))
def get_action(actor_fn, actor_params, obs, rng=None, eval=False):
    mu, sig = actor_fn(actor_params, obs)

    def get_mean(mu, sig, rng):
        return mu
    def sample_action(mu, sig, rng):
        return sample_normal_dist(mu, sig, rng)

    action = jax.lax.cond(
        jnp.logical_or(rng is None, eval),
        get_mean,
        sample_action,
        mu,
        sig,
        rng
    )

    # if rng is None or eval:
    #    action = mu
    #else:
    #    action = sample_normal_dist(mu, sig, rng)
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
        q1, q2, _ = critic_fn(critic_params, obs=obs, action=action)
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
        target_q,
        target_v,
):
    def loss(params):
        q1, q2, v = critic_fn(params, obs, action)
        q_loss = jnp.mean(0.5 * (mse(q1, target_v) + mse(q2, target_v)))
        v_loss = 0.5 * jnp.mean(mse(v, target_q))
        critic_loss = q_loss + v_loss
        return critic_loss, (q_loss, v_loss)
    (loss, aux), grads = value_and_grad(loss, has_aux=True)(critic_params)
    q_loss, v_loss = aux
    updates, new_critic_opt_state = critic_update_fn(grads, critic_opt_state, params=critic_params)
    new_critic_params = optax.apply_updates(critic_params, updates)
    grad_norm = optax.global_norm(grads)
    return new_critic_params, new_critic_opt_state, loss, grad_norm, v_loss, q_loss


@partial(jit, static_argnums=(0, 3))
def update_alpha(log_alpha_fn, alpha_params, alpha_opt_state, alpha_update_fn, log_a, target_entropy):
    diff_entropy = jax.lax.stop_gradient(log_a + target_entropy)

    def loss(params):
        def alpha_loss_fn(lp):
            return -(
                    log_alpha_fn(params) * lp
            ).mean()
        return alpha_loss_fn(diff_entropy)
    loss, grads = value_and_grad(loss)(alpha_params)
    updates, new_alpha_opt_state = alpha_update_fn(grads, alpha_opt_state, params=alpha_params)
    new_alpha_params = optax.apply_updates(alpha_params, updates)
    grad_norm = optax.global_norm(grads)
    return new_alpha_params, new_alpha_opt_state, loss, grad_norm


@struct.dataclass
class SACModelSummary:
    actor_loss: jnp.array = ZERO
    entropy: jnp.array = ZERO
    critic_loss: jnp.array = ZERO
    v_loss: jnp.array = ZERO
    q_loss: jnp.array = ZERO
    alpha_loss: jnp.array = ZERO
    log_alpha: jnp.array = ZERO
    actor_std: jnp.array = ZERO
    critic_grad_norm: jnp.array = ZERO
    actor_grad_norm: jnp.array = ZERO
    alpha_grad_norm: jnp.array = ZERO
    target_q_term: jnp.array = ZERO
    target_v_term: jnp.array = ZERO
    entropy_term: jnp.array = ZERO
    max_reward: jnp.array = ZERO
    min_reward: jnp.array = ZERO

    def dict(self):
        return {
                        'actor_loss': self.actor_loss.item(),
                        'entropy': self.entropy.item(),
                        'actor_std': self.actor_std.item(),
                        'critic_loss': self.critic_loss.item(),
                        'v_loss': self.v_loss.item(),
                        'q_loss': self.q_loss.item(),
                        'alpha_loss': self.alpha_loss.item(),
                        'log_alpha': self.log_alpha.item(),
                        'critic_grad_norm': self.critic_grad_norm.item(),
                        'actor_grad_norm': self.actor_grad_norm.item(),
                        'alpha_grad_norm': self.alpha_grad_norm.item(),
                        'target_q_term': self.target_q_term.item(),
                        'target_v_term': self.target_v_term.item(),
                        'entropy_term': self.entropy_term.item(),
                        'max_reward': self.max_reward.item(),
                        'min_reward': self.min_reward.item(),
                    }


class Actor(nn.Module):

    features: Sequence[int]
    action_dim: int
    non_linearity: Callable = nn.relu
    sig_min: float = float(1e-6)
    sig_max: float = float(1e2)

    @nn.compact
    def __call__(self, obs):
        actor_net = MLP(self.features,
                        2*self.action_dim,
                        self.non_linearity)

        out = actor_net(obs)
        mu, sig = jnp.split(out, 2, axis=-1)
        sig = nn.softplus(sig)
        sig = jnp.clip(sig, self.sig_min, self.sig_max)
        return mu, sig


class Critic(nn.Module):

    features: Sequence[int]
    non_linearity: Callable = nn.relu

    @nn.compact
    def __call__(self, obs, action):
        critic_1 = MLP(
            features=self.features,
            output_dim=1,
            non_linearity=self.non_linearity)

        critic_2 = MLP(
            features=self.features,
            output_dim=1,
            non_linearity=self.non_linearity)
        obs_action = jnp.concatenate((obs, action), -1)
        value_1 = critic_1(obs_action)
        value_2 = critic_2(obs_action)
        value = MLP(
            features=self.features,
            output_dim=1,
            non_linearity=self.non_linearity)
        return value_1, value_2, value(obs)


class ConstantModule(nn.Module):
    ent_coef_init: float = 1.0

    def setup(self):
        self.log_ent_coef = self.param("log_ent_coef",
                                       init_fn=lambda key: jnp.full((),
                                                                    jnp.log(self.ent_coef_init)))

    @nn.compact
    def __call__(self):
        return self.log_ent_coef


class SACAgent(DummyAgent):

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
            tau: float = 0.005,
            init_ent_coef: float = 1.0,
            tune_entropy_coef: bool = True,
            *args,
            **kwargs
    ):
        super(SACAgent, self).__init__(*args, **kwargs)
        action_dim = np.prod(action_space.shape)
        sample_obs = observation_space.sample()
        sample_act = action_space.sample()
        self.tune_entropy_coef = tune_entropy_coef
        self.obs_sample = sample_obs
        self.act_sample = sample_act
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_alpha = lr_alpha
        self.discount = discount
        self.target_entropy = -action_dim.astype(np.float32) if target_entropy is None else \
            target_entropy
        action_dim = int(action_dim)
        self.actor_optimizer = optax.adamw(learning_rate=lr_actor, weight_decay=weight_decay_actor)
        self.critic_optimizer = optax.adamw(learning_rate=lr_critic, weight_decay=weight_decay_critic)
        self.alpha_optimizer = optax.adamw(learning_rate=lr_alpha, weight_decay=weight_decay_alpha)
        self.actor = Actor(features=actor_features, action_dim=action_dim)
        self.critic = Critic(features=critic_features)
        self.log_alpha = ConstantModule(init_ent_coef)
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
        self.log_alpha_apply = jax.jit(self.log_alpha.apply)
        self.actor_apply = jax.jit(self.actor.apply)

    def act_in_jax(self, obs, rng=None, eval=False):
        return get_action(
            self.actor.apply,
            self.actor_params,
            obs,
            rng,
            eval
        )

    def _train_step_(self,
                     rng,
                     tran: Transition,
                     alpha_params,
                     alpha_opt_state,
                     actor_params,
                     actor_opt_state,
                     critic_params,
                     target_critic_params,
                     critic_opt_state,
                     ):

        rng, actor_rng, td_rng = random.split(rng, 3)
        alpha = jax.lax.stop_gradient(
            jnp.exp(
                self.log_alpha.apply(
                    alpha_params)
            )
        )
        td_rng, target_q_rng = random.split(td_rng, 2)
        target_v, target_q, target_v_term, target_q_term, entropy_term = \
            jax.lax.stop_gradient(
                get_soft_td_target(
                    critic_fn=self.critic.apply,
                    actor_fn=self.actor.apply,
                    next_obs=tran.next_obs,
                    obs=tran.obs,
                    reward=tran.reward,
                    not_done=1.0 - tran.done,
                    critic_target_params=target_critic_params,
                    critic_params=critic_params,
                    actor_params=actor_params,
                    alpha=alpha,
                    discount=self.discount,
                    rng=target_q_rng,
                    reward_scale=self.scale_reward,
                )
            )
        new_critic_params, new_critic_opt_state, critic_loss, critic_grad_norm, v_loss, q_loss = \
            update_critic(
                critic_fn=self.critic.apply,
                critic_params=critic_params,
                critic_opt_state=critic_opt_state,
                critic_update_fn=self.critic_optimizer.update,
                obs=tran.obs,
                action=tran.action,
                target_q=target_q,
                target_v=target_v,
            )
        new_target_critic_params = soft_update(target_critic_params, new_critic_params, self.tau)

        new_actor_params, new_actor_opt_state, actor_loss, log_a, actor_grad_norm = update_actor(
            actor_fn=self.actor.apply,
            critic_fn=self.critic.apply,
            actor_params=actor_params,
            actor_update_fn=self.actor_optimizer.update,
            critic_params=critic_params,
            alpha=alpha,
            actor_opt_state=actor_opt_state,
            obs=tran.obs,
            rng=actor_rng,
        )

        if self.tune_entropy_coef:
            new_alpha_params, new_alpha_opt_state, alpha_loss, alpha_grad_norm = update_alpha(
                log_alpha_fn=self.log_alpha.apply,
                alpha_params=alpha_params,
                alpha_opt_state=alpha_opt_state,
                alpha_update_fn=self.alpha_optimizer.update,
                log_a=log_a,
                target_entropy=self.target_entropy
            )

        else:
            alpha_loss = jnp.zeros(1)
            alpha_grad_norm = jnp.zeros(1)

        summary = SACModelSummary()
        if self.use_wandb:
            log_alpha = self.log_alpha_apply(new_alpha_params)
            _, std = self.actor_apply(new_actor_params, tran.obs)

            summary = SACModelSummary(
                actor_loss=actor_loss.astype(float),
                entropy=-log_a.mean().astype(float),
                actor_std=std.mean().astype(float),
                critic_loss=critic_loss.astype(float),
                v_loss=v_loss.astype(float),
                q_loss=q_loss.astype(float),
                alpha_loss=alpha_loss.astype(float),
                log_alpha=log_alpha.astype(float),
                critic_grad_norm=critic_grad_norm.astype(float),
                actor_grad_norm=actor_grad_norm.astype(float),
                alpha_grad_norm=alpha_grad_norm.astype(float),
                target_v_term=target_v_term.astype(float),
                target_q_term=target_q_term.astype(float),
                entropy_term=entropy_term.astype(float),
                max_reward=jnp.max(tran.reward).astype(float),
                min_reward=jnp.min(tran.reward).astype(float),
            )

        return (
            new_alpha_params,
            new_alpha_opt_state,
            new_actor_params,
            new_actor_opt_state,
            new_critic_params,
            new_target_critic_params,
            new_critic_opt_state,
            summary,
        )

    def train_step(self,
                   rng,
                   buffer: ReplayBuffer,
                   ):

        # @partial(jit, static_argnums=(0, 2))
        def sample_data(data_buffer, rng, batch_size):
            tran = data_buffer.sample(rng, batch_size=batch_size)
            return tran
        def step(carry, ins):
            rng = carry[0]
            alpha_params = carry[1]
            alpha_opt_state = carry[2]
            actor_params = carry[3]
            actor_opt_state = carry[4]
            critic_params = carry[5]
            target_critic_params = carry[6]
            critic_opt_state = carry[7]
            # if use_observations:
            #  obs = ins[0]
            # else:
            #  obs = carry[1]
            buffer_rng, train_rng, rng = jax.random.split(rng, 3)
            tran = sample_data(buffer, buffer_rng, self.batch_size)

            (
                new_alpha_params,
                new_alpha_opt_state,
                new_actor_params,
                new_actor_opt_state,
                new_critic_params,
                new_target_critic_params,
                new_critic_opt_state,
                summary,
            ) = \
                self._train_step_(
                rng=train_rng,
                tran=tran,
                alpha_params=alpha_params,
                alpha_opt_state=alpha_opt_state,
                actor_params=actor_params,
                actor_opt_state=actor_opt_state,
                critic_params=critic_params,
                target_critic_params=target_critic_params,
                critic_opt_state=critic_opt_state,
            )
            # if use_observations:
            #  carry = seed, obs, hidden
            # else:
            carry = [
                rng,
                new_alpha_params,
                new_alpha_opt_state,
                new_actor_params,
                new_actor_opt_state,
                new_critic_params,
                new_target_critic_params,
                new_critic_opt_state,
                summary,
            ]
            outs = carry[1:]
            return carry, outs
        carry = [
            rng,
            self.alpha_params,
            self.alpha_opt_state,
            self.actor_params,
            self.actor_opt_state,
            self.critic_params,
            self.target_critic_params,
            self.critic_opt_state,
            SACModelSummary(),
        ]
        carry, outs = jax.lax.scan(step, carry, xs=None, length=self.train_steps)
        self.alpha_params = carry[1]
        self.alpha_opt_state = carry[2]
        self.actor_params = carry[3]
        self.actor_opt_state = carry[4]
        self.critic_params = carry[5]
        self.target_critic_params = carry[6]
        self.critic_opt_state = carry[7]
        summary = carry[8]
        if self.use_wandb:
            wandb.log(summary.dict())







    # def train_step(
    #         self,
    #         rng,
    #         tran: Transition,
    #         writer,
    # ):
    #     rng, actor_rng, td_rng = random.split(rng, 3)
    #
    #     alpha = jax.lax.stop_gradient(
    #         jnp.exp(
    #             self.log_alpha.apply(self.alpha_params)
    #         )
    #     )
    #
    #     #for u in range(self.q_update_frequency):
    #     td_rng, target_q_rng = random.split(td_rng, 2)
    #     target_v, target_q, target_v_term, target_q_term, entropy_term = \
    #         jax.lax.stop_gradient(
    #             get_soft_td_target(
    #                 critic_fn=self.critic.apply,
    #                 actor_fn=self.actor.apply,
    #                 next_obs=tran.next_obs,
    #                 obs=tran.obs,
    #                 reward=tran.reward,
    #                 not_done=1.0 - tran.done,
    #                 critic_target_params=self.target_critic_params,
    #                 critic_params=self.critic_params,
    #                 actor_params=self.actor_params,
    #                 alpha=alpha,
    #                 discount=self.discount,
    #                 rng=target_q_rng,
    #                 reward_scale=self.scale_reward,
    #             )
    #     )
    #
    #     critic_params, critic_opt_state, critic_loss, critic_grad_norm, v_loss, q_loss = \
    #         update_critic(
    #             critic_fn=self.critic.apply,
    #             critic_params=self.critic_params,
    #             critic_opt_state=self.critic_opt_state,
    #             critic_update_fn=self.critic_optimizer.update,
    #             obs=tran.obs,
    #             action=tran.action,
    #             target_q=target_q,
    #             target_v=target_v,
    #         )
    #
    #     target_critic_params = soft_update(self.target_critic_params, critic_params, self.tau)
    #
    #     actor_params, actor_opt_state, actor_loss, log_a, actor_grad_norm = update_actor(
    #         actor_fn=self.actor.apply,
    #         critic_fn=self.critic.apply,
    #         actor_params=self.actor_params,
    #         actor_update_fn=self.actor_optimizer.update,
    #         critic_params=self.critic_params,
    #         alpha=alpha,
    #         actor_opt_state=self.actor_opt_state,
    #         obs=tran.obs,
    #         rng=actor_rng,
    #     )
    #
    #     if self.tune_entropy_coef:
    #         alpha_params, alpha_opt_state, alpha_loss, alpha_grad_norm = update_alpha(
    #             log_alpha_fn=self.log_alpha.apply,
    #             alpha_params=self.alpha_params,
    #             alpha_opt_state=self.alpha_opt_state,
    #             alpha_update_fn=self.alpha_optimizer.update,
    #             log_a=log_a,
    #             target_entropy=self.target_entropy
    #         )
    #         self.alpha_params = alpha_params
    #         self.alpha_opt_state = alpha_opt_state
    #     else:
    #         alpha_loss = jnp.zeros(1)
    #         alpha_grad_norm = jnp.zeros(1)
    #
    #     self.actor_params = actor_params
    #     self.actor_opt_state = actor_opt_state
    #     self.critic_params = critic_params
    #     self.critic_opt_state = critic_opt_state
    #     self.target_critic_params = target_critic_params
    #
    #     log_alpha = self.log_alpha.apply(self.alpha_params)
    #     _, std = self.actor.apply(self.actor_params, tran.obs)
    #
    #     summary = SACModelSummary(
    #         actor_loss=actor_loss.item(),
    #         entropy=-log_a.mean().item(),
    #         actor_std=std.mean().item(),
    #         critic_loss=critic_loss.item(),
    #         v_loss=v_loss.item(),
    #         q_loss=q_loss.item(),
    #         alpha_loss=alpha_loss.item(),
    #         log_alpha=log_alpha.item(),
    #         critic_grad_norm=critic_grad_norm.item(),
    #         actor_grad_norm=actor_grad_norm.item(),
    #         alpha_grad_norm=alpha_grad_norm.item(),
    #         target_v_term=target_v_term.item(),
    #         target_q_term=target_q_term.item(),
    #         entropy_term=entropy_term.item(),
    #         max_reward=jnp.max(tran.reward).item(),
    #         min_reward=jnp.min(tran.reward).item(),
    #     )
    #
    #     return summary.dict()











