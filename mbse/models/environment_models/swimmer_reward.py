import jax.numpy as jnp
import jax
from mbse.models.reward_model import RewardModel
from functools import partial


class SwimmerRewardModel(RewardModel):
    """Get Pendulum Reward."""

    def __init__(self, ctrl_cost_weight=0.01):
        super().__init__()
        self.ctrl_cost_weight = ctrl_cost_weight

    @partial(jax.jit, static_argnums=0)
    def predict(self, obs, action, next_obs=None, rng=None):
        nose_to_target = obs[..., -2:]
        reward_control = - jnp.square(jnp.linalg.norm(action, axis=-1)) * self.ctrl_cost_weight
        reward = - jnp.square(jnp.linalg.norm(nose_to_target, axis=-1)) + reward_control
        reward = reward.reshape(-1).squeeze()
        return reward
