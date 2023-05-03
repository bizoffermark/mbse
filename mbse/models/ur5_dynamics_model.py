"""Pendulum Swing-up Environment with full observation."""
import numpy as np
from mbse.models.reward_model import RewardModel
from mbse.models.dynamics_model import DynamicsModel
from gym.envs.classic_control.pendulum import angle_normalize
import jax.numpy as jnp
import jax
from functools import partial
from typing import Union, Optional, Any
from mbse.utils.type_aliases import ModelProperties
from pyur5.models import EnsembleModel
import math

# class Ur5PendulumEnv(PendulumEnv):
#     def __init__(self, ctrl_cost=0.001, *args, **kwargs):
#         self.state = None
#         super(Ur5PendulumEnv, self).__init__(*args, **kwargs)
#         self.observation_space.sample = self.sample_obs
#         self.standard_ctrl_cost = 0.001
#         self.ctrl_cost = ctrl_cost
#         self.model = EnsembleModel()
#         super().set_bounds(max_action=0.7, min_action=-0.7)
                
#     def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
#         super().reset(seed=seed, options=options)
#         self.state = np.array([0.0, 0.0])
#         return self._get_obs(), {}
    
#     def sample_obs(self, mask: Optional[Any] = None):
#         high = np.array([np.pi, 1.0])
#         low = -high
#         theta, thetadot = self.np_random.uniform(low=low, high=high)
#         obs = np.asarray([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
#         return obs
    
#     def step(self, u):
#         next_obs = self.model.predict(self._get_obs(), u)


class Ur5PendulumDynamicsModel(DynamicsModel):
    def __init__(self, ctrl_cost_weight=0.001, sparse=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.env = env
        # reward_model = PendulumReward(
        #     action_space=self.env.action_space,
        #     ctrl_cost_weight=ctrl_cost_weight,
        #     sparse=sparse
        # )
        # self.max_action = 0.7
        # self.min_action = -0.7
        self.model = EnsembleModel()
        self.min_action = self.model.action_min
        self.max_action = self.model.action_max
                
        # TODO: Maybe think of punishing v_ee to be zero as well
        self.target_state = jnp.array([math.pi, 0]) # target state to be theta = pi/2, theta_dot = 0 
        self.cost_weights = jnp.array([1, 0.1]) # weight for theta and theta_dot
        
        self.reward_model = Ur5PendulumReward(
            action_space = self.model.action_space,
            ctrl_cost_weight = ctrl_cost_weight,
            sparse = sparse,
            min_action = self.min_action,
            max_action = self.max_action,
            target_state = self.target_state,
            cost_weights = self.cost_weights
        )
        
        self.pred_diff = False
        self.obs_dim = 4 #observation

        
    @partial(jax.jit, static_argnums=0)
    def predict(self,
                obs,
                action,
                rng=None,
                parameters=None,
                model_props: ModelProperties = ModelProperties(),
                sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
                ):
        
        u = self.rescale_action(action)
        next_obs, terminate, truncate, output_dict = self.model.step(obs, u)
        
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
        theta, thetadot, p_ee, v_ee = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
        return jnp.asarray([theta, thetadot, p_ee, v_ee], dtype=jnp.float32)


    @partial(jax.jit, static_argnums=0)
    def rescale_action(self, action):
        """Rescales the action affinely from  [:attr:`min_action`, :attr:`max_action`] to the action space of the base environment, :attr:`env`.

        Args:
            action: The action to rescale

        Returns:
            The rescaled action
        """
        action = jnp.clip(action, self.min_action, self.max_action)
        low = self.model.action_space.low
        high = self.model.action_space.high
        action = low + (high - low) * (
                (action - self.min_action) / (self.max_action - self.min_action)
        )
        action = jnp.clip(action, low, high)
        return action

class Ur5PendulumReward(RewardModel):
    """Get Pendulum Reward."""

    def __init__(self, action_space, ctrl_cost_weight=0.001, sparse=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctrl_cost_weight = ctrl_cost_weight
        self.sparse = sparse
        print(kwargs)
        self.min_action = kwargs['min_action']
        self.max_action = kwargs['max_action']
        self.action_space = action_space 
        self._init_fn()
        self.target_state = kwargs['target_state']
        self.cost_weights = kwargs['cost_weights']
        
    def _init_fn(self):
        # rescale actions from [-1, 1] to [min_action, max_action]
        self.rescale_action = jax.jit(lambda action: self._rescale_action(action=action,
                                                                          min_action=self.min_action,
                                                                          max_action=self.max_action,
                                                                          low=self.action_space.low,
                                                                          high=self.action_space.high,
                                                                          ))
        self.input_cost = jax.jit(lambda u: self._input_cost(u=u, ctrl_cost_weight=self.ctrl_cost_weight))

        def predict(obs, action, next_obs=None, rng=None):
            return self._predict(
                state_reward_fn=self.state_reward,
                input_cost_fn=self.input_cost,
                action_transform_fn=self.rescale_action,
                obs=obs,
                action=action,
                target_state=self.target_state,
                cost_weights=self.cost_weights,
                next_obs=next_obs,
                rng=rng
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
    def state_non_sparse_reward(theta, thetadot, p_ee, v_ee, target_state, cost_weights):
        """Get sparse reward."""
        theta = angle_normalize(theta)
        cost = cost_weights[0] * (theta - target_state[0])**2 + cost_weights[1] * (thetadot - target_state[1])**2 
        return -cost 

    @staticmethod
    def _input_cost(u, ctrl_cost_weight):
        # compute the |u|^2 * w 
        return ctrl_cost_weight * (jnp.sum(jnp.square(u), axis=-1))

    @staticmethod
    @jax.jit
    def state_reward(state, target_state, cost_weights):
        """Compute reward associated with state dynamics."""
        theta, thetadot, p_ee, v_ee = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
        theta = angle_normalize(theta)
        cost = cost_weights[0] * (theta - target_state[0])**2 + cost_weights[1] * (thetadot - target_state[1])**2 

        # the reward is to make sure that we are
        return -cost

    @staticmethod
    def _predict(state_reward_fn, input_cost_fn, action_transform_fn, obs, action, target_state, cost_weights, next_obs=None, rng=None):
        action = action_transform_fn(action) # transform the action to the normal range
        return state_reward_fn(state=obs, target_state=target_state, cost_weights=cost_weights) - input_cost_fn(action)

    def evaluate(self,
                 parameters,
                 obs,
                 action,
                 rng,
                 sampling_idx=None,
                 model_props: ModelProperties = ModelProperties()):
        print("Ur5DynamicsReward evaluate called")
        next_state = self.predict(obs=obs, action=action, rng=rng)
        print("next_state is : ", next_state)
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
