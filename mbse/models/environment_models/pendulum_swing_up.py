"""Pendulum Swing-up Environment with full observation."""
import numpy as np
from mbse.models.reward_model import RewardModel
from mbse.models.dynamics_model import DynamicsModel
from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize
import jax.numpy as jnp
import jax
from functools import partial
from typing import Union, Optional, Any
from mbse.utils.type_aliases import ModelProperties
import math
from gym.spaces import Box

class PendulumReward(RewardModel):
    """Get Pendulum Reward."""

    def __init__(self, action_space, ctrl_cost_weight=0.001, sparse=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctrl_cost_weight = ctrl_cost_weight
        self.sparse = sparse
        self.min_action = None
        self.max_action = None
        self.action_space = action_space
        self._init_fn()

    def _init_fn(self):
        self.rescale_action = jax.jit(lambda action: self._rescale_action(action=action,
                                                                          min_action=self.min_action,
                                                                          max_action=self.max_action,
                                                                          low=self.action_space.low,
                                                                          high=self.action_space.high,
                                                                          ))
        self.input_cost = jax.jit(lambda u: self._input_cost(ctrl_cost_weight=self.ctrl_cost_weight, u=u))

        def predict(obs, action, next_obs=None, rng=None):
            return self._predict(
                state_reward_fn=self.state_reward,
                input_cost_fn=self.input_cost,
                action_transform_fn=self.rescale_action,
                obs=obs,
                action=action,
                next_obs=next_obs,
                rng=rng,
            )

        self.predict = jax.jit(predict)

    def set_bounds(self, max_action, min_action=None):
        self.max_action = max_action
        if min_action is None:
            min_action = - max_action
        self.min_action = min_action
        self._init_fn()

    @staticmethod
    @jax.jit
    def state_non_sparse_reward(theta, omega):
        """Get sparse reward."""
        theta = angle_normalize(theta)
        return -(theta ** 2 + 0.1 * omega ** 2)

    @staticmethod
    def _input_cost(ctrl_cost_weight, u):
        return ctrl_cost_weight * (jnp.sum(jnp.square(u), axis=-1))

    @staticmethod
    @jax.jit
    def state_reward(state):
        """Compute reward associated with state dynamics."""
        theta, omega = jnp.arctan2(state[..., 1], state[..., 0]), state[..., 2]
        theta = angle_normalize(theta)
        return -(theta ** 2 + 0.1 * omega ** 2)

    @staticmethod
    def _predict(state_reward_fn, input_cost_fn, action_transform_fn, obs, action, next_obs=None, rng=None):
        action = action_transform_fn(action)
        return state_reward_fn(state=obs) - input_cost_fn(action)

    def evaluate(self,
                 parameters,
                 obs,
                 action,
                 rng,
                 sampling_idx=None,
                 model_props: ModelProperties = ModelProperties()):
        next_state = self.predict(obs=obs, action=action, rng=rng)
        reward = jnp.zeros(next_state.shape[0])
        return next_state, reward

    @staticmethod
    def _rescale_action(action, min_action, max_action, low, high):
        """
        Args:
            action: The action to rescale

        Returns:
            The rescaled action
        """
        if min_action is not None and max_action is not None:
            action = jnp.clip(action, min_action, max_action)
            action = low + (high - low) * (
                    (action - min_action) / (max_action - min_action)
            )
            action = jnp.clip(action, low, high)
        return action


class CustomPendulumEnv(PendulumEnv):
    def __init__(self, ctrl_cost=0.001, render_mode='rgb_array', *args, **kwargs):
        self.state = None
        super(CustomPendulumEnv, self).__init__(render_mode=render_mode, *args, **kwargs)
        self.observation_space.sample = self.sample_obs
        self.standard_ctrl_cost = 0.001
        self.ctrl_cost = ctrl_cost

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
        self.state = np.array([np.pi, 0.0])
        return self._get_obs(), {}

    def sample_obs(self, mask: Optional[Any] = None):
        high = np.array([np.pi, 1.0])
        low = -high
        theta, thetadot = self.np_random.uniform(low=low, high=high)
        obs = np.asarray([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
        return obs

    def step(self, u):
        next_obs, reward, terminate, truncate, output_dict = super().step(u)
        action_reward = (-self.ctrl_cost + self.standard_ctrl_cost) * (u ** 2)
        reward = reward + action_reward
        return next_obs, reward, terminate, truncate, output_dict

    
class PendulumSwingUpEnv(PendulumEnv):
    """Pendulum Swing-up Environment."""

    def __init__(self, reset_noise_scale=0.01, ctrl_cost_weight=0.001, sparse=False, render_model='rgb_array'):
        self.base_mujoco_name = "Pendulum-v1"

        super(PendulumSwingUpEnv, self).__init__(render_mode=render_model)
        self.reset_noise_scale = reset_noise_scale
        self.state = np.zeros(2)
        self.last_u = None
        self._reward_model = PendulumReward(
            action_space=self.action_space,
            ctrl_cost_weight=ctrl_cost_weight,
            sparse=sparse,
        )

    def reset(self, seed=None):
        """Reset to fix initial conditions."""
        x0 = np.array([np.pi, 0])
        self.state = self.np_random.uniform(
            low=x0 - self.reset_noise_scale, high=x0 + self.reset_noise_scale
        )

        self.last_u = None
        info = {}
        return self._get_obs(), info

    def step(self, u):
        """Override step method of pendulum env."""
        reward = self._reward_model.predict(jnp.asarray(self._get_obs()), jnp.asarray(u))
        reward = np.asarray(reward)
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        th, omega = self.state

        omega_dot = (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u)

        new_omega = omega + omega_dot * dt
        new_theta = th + new_omega * dt  # Simplectic integration new_omega.

        new_omega = np.clip(new_omega, -self.max_speed, self.max_speed)

        self.state = np.array([new_theta, new_omega])
        next_obs = self._get_obs()
        return next_obs, reward, False, False, {}

    def reward_model(self):
        """Get reward model."""
        return self._reward_model


class PendulumDynamicsModel(DynamicsModel):
    def __init__(self, env: PendulumEnv, ctrl_cost_weight=0.001, sparse=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env
        reward_model = PendulumReward(
            action_space=self.env.action_space,
            ctrl_cost_weight=ctrl_cost_weight,
            sparse=sparse
        )
        self.reward_model = reward_model
        self.pred_diff = False
        self.obs_dim = 3

    @partial(jax.jit, static_argnums=0)
    def predict(self,
                obs,
                action,
                rng=None,
                parameters=None,
                model_props: ModelProperties = ModelProperties(),
                sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
                ):
        u = jnp.clip(self.rescale_action(action), -self.env.max_torque, self.env.max_torque)[0]
        theta, omega = self._get_reduced_state(obs)

        g = self.env.g
        m = self.env.m
        l = self.env.l
        dt = self.env.dt
        th, omega = self._get_reduced_state(obs)

        omega_dot = (3 * g / (2 * l) * jnp.sin(th) + 3.0 / (m * l ** 2) * u)

        new_omega = omega + omega_dot * dt
        new_theta = theta + new_omega * dt  # Simplectic integration new_omega.

        new_omega = jnp.clip(new_omega, -self.env.max_speed, self.env.max_speed)

        new_state = jnp.asarray([new_theta, new_omega]).T
        next_obs = self._get_obs(new_state)
        next_obs = next_obs.squeeze()
        return next_obs

    def evaluate(self,
                 obs,
                 action,
                 rng=None,
                 parameters=None,
                 sampling_idx=None,
                 model_props: ModelProperties = ModelProperties(),
                 ):
        next_obs = self.predict(obs, action, rng)
        reward = self.reward_model.predict(obs, action, next_obs)
        return next_obs, reward

    @staticmethod
    @jax.jit
    def _get_obs(state):
        theta, thetadot = state[..., 0], state[..., 1]
        return jnp.asarray([jnp.cos(theta), jnp.sin(theta), thetadot], dtype=jnp.float32)

    @staticmethod
    @jax.jit
    def _get_reduced_state(obs):
        cos_theta, sin_theta = obs[..., 0], obs[..., 1]
        theta = jnp.arctan2(sin_theta, cos_theta)
        return theta, obs[..., -1]

    @partial(jax.jit, static_argnums=0)
    def rescale_action(self, action):
        """Rescales the action affinely from  [:attr:`min_action`, :attr:`max_action`] to the action space of the base environment, :attr:`env`.

        Args:
            action: The action to rescale

        Returns:
            The rescaled action
        """
        action = jnp.clip(action, self.env.min_action, self.env.max_action)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low) * (
                (action - self.env.min_action) / (self.env.max_action - self.env.min_action)
        )
        action = jnp.clip(action, low, high)
        return action

