import jax.numpy as jnp
import jax
from mbse.models.reward_model import RewardModel
from functools import partial


class SwimmerRewardModel(RewardModel):
    """Get Pendulum Reward."""

    def __init__(self, ctrl_cost_weight=0.0):
        super().__init__()
        self.ctrl_cost_weight = ctrl_cost_weight

    @partial(jax.jit, static_argnums=0)
    def predict(self, obs, action, next_obs=None, rng=None):
        nose_to_target = obs[..., -2:]
        reward_dist = -jnp.linalg.norm(nose_to_target, axis=-1)
        reward_ctrl = -jnp.square(action).sum(-1)
        reward = reward_dist + self.ctrl_cost_weight * reward_ctrl
        reward = reward.reshape(-1).squeeze()
        return reward
