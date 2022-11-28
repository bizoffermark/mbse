from functools import partial


from typing import Sequence, Callable, Optional
import optax
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import jit, vmap, random, value_and_grad
from mbse.utils.network_utils import MLP, mse
from mbse.utils.replay_buffer import Transition
from flax import struct

EPS = 1e-6


def sample_normal_dist(mu, log_sig, rng):
    return mu + jax.random.normal(rng, mu.shape)*jnp.exp(log_sig)


def gaussian_log_likelihood(x, mu, log_sig):
    log_l = -0.5 * (2 * log_sig + jnp.log(2*jnp.pi)
                     + ((x - mu)/(jnp.exp(log_sig) + EPS))**2)
    return jnp.sum(log_l, axis=-1)

# Perform Polyak averaging provided two network parameters and the averaging value tau.
@jit
def soft_update(
    target_params, online_params, tau=0.005
):
    return jax.tree_util.tree_map(
        lambda x, y: (1 - tau) * x + tau * y, target_params, online_params
    )


@struct.dataclass
class SACModelSummary:
    actor_loss: jnp.ndarray
    entropy: jnp.ndarray
    critic_loss: jnp.ndarray
    alpha_loss: jnp.ndarray
    log_alpha: jnp.ndarray


class Actor(nn.Module):

    features: Sequence[int]
    action_dim: int
    non_linearity: Callable = nn.swish
    log_sig_min: float = -20
    log_sig_max: float = 3

    @nn.compact
    def __call__(self, obs, train=False):
        actor_net = MLP(self.features,
                        2*self.action_dim,
                        self.non_linearity)

        out = actor_net(obs)
        mu, log_sig = jnp.split(out, 2, axis=-1)
        log_sig = nn.softplus(log_sig)
        log_sig = jnp.clip(log_sig, self.log_sig_min, self.log_sig_max)
        return mu, log_sig


class Critic(nn.Module):

    features: Sequence[int]
    non_linearity: Callable = nn.swish

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

    def __call__(self, constant):
        return self.const*constant


class SACAgent(object):

    def __init__(
            self,
            action_dim: int,
            sample_obs: jnp.ndarray,
            sample_act: jnp.ndarray,
            discount: float = 0.99,
            initial_log_alpha: float = 1.0,
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
            q_update_frequency: int = 5,
            scale_reward: float = 1,
    ):
        self.initial_log_alpha = initial_log_alpha
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_alpha = lr_alpha
        self.discount = discount
        self.target_entropy = -action_dim if target_entropy is None else target_entropy
        self.actor_optimizer = optax.adamw(learning_rate=lr_actor, weight_decay=weight_decay_actor)
        self.critic_optimizer = optax.adamw(learning_rate=lr_critic, weight_decay=weight_decay_critic)
        self.alpha_optimizer = optax.adamw(learning_rate=lr_alpha, weight_decay=weight_decay_alpha)
        self.actor = Actor(features=actor_features, action_dim=action_dim)
        self.critic = Critic(features=critic_features)
        self.log_alpha = ConstantModule()
        self.critic_features = critic_features

        rng, actor_rng, critic_rng, alpha_rng = random.split(rng, 4)
        actor_params = target_actor_params = self.actor.init(actor_rng, sample_obs)
        actor_opt_state = self.actor_optimizer.init(actor_params)
        self.actor_params = actor_params
        self.target_actor_params = target_actor_params
        self.actor_opt_state = actor_opt_state

        critic_params = target_critic_params = self.critic.init(
            critic_rng, sample_obs, sample_act
        )
        critic_opt_state = self.critic_optimizer.init(critic_params)
        self.critic_params = critic_params
        self.target_critic_params = target_critic_params
        self.critic_opt_state = critic_opt_state

        alpha_params = self.log_alpha.init(alpha_rng, self.initial_log_alpha)
        alpha_opt_state = self.alpha_optimizer.init(alpha_params)
        self.alpha_params = alpha_params
        self.alpha_opt_state = alpha_opt_state
        self.q_update_frequency = q_update_frequency
        self.scale_reward = scale_reward

    @staticmethod
    def squash_action(action):
        return nn.tanh(action)

    @partial(jit, static_argnums=0)
    def get_action(self, actor_params, obs, rng=None):
        mu, log_sig = self.actor.apply(actor_params, obs)
        action = mu if rng is None else sample_normal_dist(mu, log_sig, rng)
        return self.squash_action(action)

    @partial(jit, static_argnums=0)
    def act(self, obs, rng=None):
        return self.get_action(self.actor_params, obs, rng)

    @partial(jit, static_argnums=0)
    def get_soft_td_target(
            self,
            next_obs,
            reward,
            not_done,
            critic_target_params,
            actor_params,
            alpha_params,
            rng
    ):

        # Sample action from Pi to simulate expectation
        next_action, next_log_a = self.get_squashed_log_prob(
                    obs=next_obs,
                    params=actor_params,
                    rng=rng,
                )

        # next_action = self.get_action(obs=next_obs, actor_params=actor_params, rng=rng)

        log_alpha = self.log_alpha.apply(
                    alpha_params,
                    self.initial_log_alpha,
        )
        alpha = jnp.exp(log_alpha)
        # Get predictions for both Q functions and take the min
        target_q1, target_q2 = self.critic.apply(critic_target_params, next_obs, next_action)
        # Soft target update
        target_q = jnp.minimum(target_q1, target_q2)
        target_q = target_q - alpha * next_log_a
        # Td target = r_t + \gamma V_{t+1}
        target_q = self.scale_reward*reward + not_done * self.discount * target_q
        return target_q

    @partial(jit, static_argnums=0)
    def get_squashed_log_prob(self, params, obs, rng):
        mu, log_sig = self.actor.apply(params, obs)
        u = sample_normal_dist(mu, log_sig, rng)
        log_l = gaussian_log_likelihood(u, mu, log_sig)
        a = self.squash_action(u)
        log_l -= jnp.sum(
            jnp.log((1 - a ** 2) + EPS), axis=-1
        )
        return a, log_l.reshape(-1, 1)

    @partial(jit, static_argnums=0)
    def update_actor(self, actor_params, critic_params, alpha_params, actor_opt_state, obs, rng):
        def loss(params):
            log_alpha = self.log_alpha.apply(alpha_params, self.initial_log_alpha)
            alpha = jnp.exp(log_alpha)
            action, log_l = self.get_squashed_log_prob(params, obs, rng)
            q1, q2 = self.critic.apply(critic_params, obs, action)
            min_q = jnp.minimum(q1, q2)
            actor_loss = - min_q + alpha*log_l
            return jnp.mean(actor_loss), log_l

        (loss, log_a), grads = value_and_grad(loss, has_aux=True)(actor_params)
        updates, new_actor_opt_state = self.actor_optimizer.update(grads, actor_opt_state)
        new_actor_params = optax.apply_updates(actor_params, updates)
        return new_actor_params, new_actor_opt_state, loss, log_a

    @partial(jit, static_argnums=0)
    def update_critic(self, critic_params, critic_opt_state, obs, action, target_q):
        def loss(params):
            q1, q2 = self.critic.apply(params, obs, action)
            critic_loss = 0.5 * (mse(q1, target_q) + mse(q2, target_q))
            return jnp.mean(critic_loss)
        loss, grads = value_and_grad(loss)(critic_params)
        updates, new_critic_opt_state = self.critic_optimizer.update(grads, critic_opt_state)
        new_critic_params = optax.apply_updates(critic_params, updates)
        return new_critic_params, new_critic_opt_state, loss

    @partial(jit, static_argnums=0)
    def update_alpha(self, alpha_params, alpha_opt_state, log_a):
        log_a = jax.lax.stop_gradient(log_a)

        def loss(params):
            @vmap
            def alpha_loss_fn(lp):
                return -(
                        self.log_alpha.apply(params, self.initial_log_alpha) * (lp + self.target_entropy)
                ).mean()

            return jnp.mean(alpha_loss_fn(log_a))

        loss, grads = value_and_grad(loss)(alpha_params)
        updates, new_alpha_opt_state = self.alpha_optimizer.update(grads, alpha_opt_state)
        new_alpha_params = optax.apply_updates(alpha_params, updates)
        return new_alpha_params, new_alpha_opt_state, loss

    def train_step(
            self,
            rng,
            tran: Transition,
    ):
        rng, actor_rng, td_rng = random.split(rng, 3)

        for u in range(self.q_update_frequency):
            target_q = jax.lax.stop_gradient(
                self.get_soft_td_target(
                    next_obs=tran.next_obs,
                    reward=tran.reward,
                    not_done=1 - tran.done,
                    actor_params=self.actor_params,
                    critic_target_params=self.target_critic_params,
                    alpha_params=self.alpha_params,
                    rng=td_rng,
                )
            )

            critic_params, critic_opt_state, critic_loss = self.update_critic(
                critic_params=self.critic_params,
                critic_opt_state=self.critic_opt_state,
                obs=tran.obs,
                action=tran.action,
                target_q=target_q,
            )

            self.target_critic_params = soft_update(self.target_critic_params, critic_params)
            self.critic_params = critic_params
            self.critic_opt_state = critic_opt_state

        actor_params, actor_opt_state, actor_loss, log_a = self.update_actor(
            actor_params=self.actor_params,
            critic_params=self.critic_params,
            alpha_params=self.alpha_params,
            actor_opt_state=self.actor_opt_state,
            obs=tran.obs,
            rng=actor_rng,
        )

        alpha_params, alpha_opt_state, alpha_loss = self.update_alpha(
            alpha_params=self.alpha_params, alpha_opt_state=self.alpha_opt_state, log_a=log_a
        )

        target_actor_params = soft_update(self.target_actor_params, actor_params)
        self.actor_params = actor_params
        self.actor_opt_state = actor_opt_state
        self.alpha_params = alpha_params
        self.alpha_opt_state = alpha_opt_state
        self.target_actor_params = target_actor_params

        log_alpha = self.log_alpha.apply(self.alpha_params, self.initial_log_alpha)
        summary = SACModelSummary(
            actor_loss=actor_loss,
            entropy=-log_a.mean(),
            critic_loss=critic_loss,
            alpha_loss=alpha_loss,
            log_alpha=log_alpha,
        )

        return summary











