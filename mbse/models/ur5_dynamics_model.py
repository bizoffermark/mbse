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

from pyur5.models.ens_model import EnsembleModel
import math


class Ur5PendulumDynamicsModel(DynamicsModel):
    def __init__(self, task_typ='new', use_cos=True, train_horizon=1, n_model=5, ctrl_cost_weight=0.001, sparse=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_cos = use_cos
        self.task_typ = task_typ
        self.model = EnsembleModel(train_horizon=train_horizon, n_model=n_model, use_cos=use_cos, task_typ=task_typ)
        self.max_action = self.model.action_max
        self.min_action = self.model.action_min
        self.n_action = self.model.n_action
        # TODO: Maybe think of punishing v_ee to be zero as well
        self.target_state = self.model.target_state #jnp.array([math.pi, 0]) # target state to be theta = pi/2, theta_dot = 0 
        self.cost_weights = self.model.cost_weights[:1] #jnp.array([1, 0.1]) # weight for theta and theta_dot
        # self.ctrl_cost_weight = self.model.cost_weights[-1]
        
        # self.reward_model = Ur5PendulumReward(
        #     action_space = self.model.action_space,
        #     ctrl_cost_weight = ctrl_cost_weight,
        #     sparse = sparse,
        #     min_action = self.min_action,
        #     max_action = self.max_action,
        #     target_state = self.target_state,
        #     cost_weights = self.cost_weights
        # )
        self.reward_model = Ur5PendulumReward(self.model)
        
        self.pred_diff = False
        self.obs_dim = self.model.x_dim #observation
        self.obs_space = self.model.obs_space

    def set_bounds(self, max_action, min_action=None):
        self.max_action = max_action
        if min_action is None:
            min_action = - max_action
        self.min_action = min_action
        self._init_fn()
        
    @partial(jax.jit, static_argnums=0)
    def predict(self,
                obs,
                action,
                rng=None,
                parameters=None,
                model_props: ModelProperties = ModelProperties(),
                sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
                rescaled: bool = False,
                ):
        '''
            internal function to predict next state
        '''
        u = jnp.clip(action, -1, 1) # action is assumed to be in the range of [-1, 1]
        next_obs, terminate, truncate, output_dict = self.model.step(obs, u)
        
        return next_obs

    def reset(self):
        if self.use_cos:
            obs = jnp.array([1.0, 0.0, 0.0] + [0.0, 0.0] * self.n_action)
        else:
            obs = jnp.array([0.0, 0.0] + [0.0, 0.0] * self.n_action)
        return obs
    def evaluate(self,
                 obs,
                 action,
                 rng=None,
                 parameters=None,
                 sampling_idx=None,
                 model_props: ModelProperties = ModelProperties(),
                 rescaled: bool = False,
                 ):
        # u = self.rescale_action(action) # rescale action to [-0.7, 0.7]
        '''
            function to evaluate next state and reward
        '''
        # action here is assumed to be in the range of [-1, 1]
        # if not rescaled:
        #     action = self.rescale_action(action)
        next_obs = self.predict(obs, action, rng, rescaled=True)
        reward = self.reward_model.predict(obs, action, next_obs)
        return next_obs, reward

    @staticmethod
    @jax.jit
    def _get_obs(state):
        theta, theta_dot, p_ee, v_ee = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
        return jnp.asarray([theta, theta_dot, p_ee, v_ee], dtype=jnp.float32)


    # @partial(jax.jit, static_argnums=0)
    # def rescale_action(self, action):
    #     """Rescales the action affinely from  [:attr:`min_action`, :attr:`max_action`] to the action space of the base environment, :attr:`env`.

    #     Args:
    #         action: The action to rescale

    #     Returns:
    #         The rescaled action
    #     """
    # #     action = jnp.clip(action, self.min_action, self.max_action)
    #     low = self.model.action_space.low
    #     high = self.model.action_space.high
    #     action = low + (high - low) * (
    #             (action - self.min_action) / (self.max_action - self.min_action)
    #     )
    #     action = jnp.clip(action, low, high)
    #     return action

class Ur5PendulumReward(RewardModel):
    """Get Pendulum Reward."""

    def __init__(self, model=None, ctrl_cost_weight=0.001, sparse=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctrl_cost_weight = ctrl_cost_weight
        self.sparse = sparse
        self.model = model
        if model is not None:
            self.min_action = model.action_min#kwargs['min_action']
            self.max_action = model.action_max#kwargs['max_action']
            self.action_space = model.action_space 
            self.target_state = model.target_state#kwargs['target_state']
            self.cost_weights = model.cost_weights#kwargs['cost_weights']
        else:
            self.min_action = kwargs['min_action']
            self.max_action = kwargs['max_action']
            self.action_space = model.action_space 
            self.target_state = kwargs['target_state']
            self.cost_weights = kwargs['cost_weights']
        self._init_fn()
        print("target state: {}".format(self.target_state))
        print("cost weights: {}".format(self.cost_weights))
        print("reward action space: {}".format(self.action_space))
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
            # return self.model._reward_fn(obs, action)
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
    def state_non_sparse_reward(theta, theta_dot, p_ee, v_ee, target_state, cost_weights):
        """Get sparse reward."""
        theta = angle_normalize(theta)
        dtheta = theta - target_state[0]
        dtheta = angle_normalize(dtheta)
        dtheta_dot = theta_dot - target_state[1]
        cost = cost_weights[0] * (dtheta)**2 + cost_weights[1] * (dtheta_dot)**2 
        return -cost 

    @staticmethod
    def _input_cost(u, ctrl_cost_weight):
        # compute the |u|^2 * w 
        return ctrl_cost_weight * (jnp.sum(jnp.square(u), axis=-1))

    @staticmethod
    @jax.jit
    def state_reward(state, target_state, cost_weights):
        # TODO: This is outdated. NEED MODIFICATION!!
        """Compute reward associated with state dynamics."""
        # print("state shape: {}".format(state.shape))
        # if state.shape[0] == 5:
        if state.shape == (5,):
            cos, sin, theta_dot, p_ee, v_ee = state[..., 0], state[..., 1], state[..., 2], state[..., 3], state[..., 4]
            theta = jnp.arctan2(sin, cos)
        else:
            theta, theta_dot, p_ee, v_ee = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
        
        theta = angle_normalize(theta)
        dtheta = theta - target_state[0]
        dtheta = angle_normalize(dtheta) # normalize the angle to [-pi, pi]
        dtheta_dot = theta_dot - target_state[1]
        print("dtheta: {}".format(dtheta))
        print("dtheta_dot: {}".format(dtheta_dot))
        cost = cost_weights[0] * (dtheta)**2 + cost_weights[1] * (dtheta_dot)**2 

        # the reward is to make sure that we are
        return -cost

    @staticmethod
    def _predict(state_reward_fn, input_cost_fn, action_transform_fn, obs, action, target_state, cost_weights, next_obs=None, rng=None, action_rng=[-1, 1]):
        # action = action_transform_fn(action) # transform the action to the normal range
        action = jnp.clip(action, action_rng[0], action_rng[1]) # clip to [-1, 1]
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
        # if min_action is not None and max_action is not None:
        #     action = jnp.clip(action, min_action, max_action)
        #     action = low + (high - low) * (
        #             (action - min_action) / (max_action - min_action)
        #     )
        #     action = jnp.clip(action, low, high)
        return action
