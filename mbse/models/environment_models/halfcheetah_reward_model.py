import jax.numpy as jnp
import jax
from mbse.models.reward_model import RewardModel
from functools import partial


class HalfCheetahReward(RewardModel):
    """Get Pendulum Reward."""

    def __init__(self, forward_velocity_weight: float = 1.0, ctrl_cost_weight: float = 0.1):
        super().__init__()
        self.ctrl_cost_weight = ctrl_cost_weight
        self.forward_velocity_weight = forward_velocity_weight

    @partial(jax.jit, static_argnums=0)
    def predict(self, obs, action, next_obs, rng=None):
        reward_ctrl = -self.ctrl_cost_weight * jnp.square(action).sum(axis=-1)
        reward_run = self.forward_velocity_weight * (next_obs[..., 0] - 0.0 * jnp.square(next_obs[..., 2]))
        reward = reward_run + reward_ctrl
        return reward
