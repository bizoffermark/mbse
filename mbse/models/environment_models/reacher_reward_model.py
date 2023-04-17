import jax.numpy as jnp
import jax
from mbse.models.reward_model import RewardModel
from functools import partial

_RUN_SPEED = 10


class ReacherRewardModel(RewardModel):
    """Get Pendulum Reward."""

    def __init__(self):
        super().__init__()

    @partial(jax.jit, static_argnums=0)
    def predict(self, obs, action, next_obs=None, rng=None):
        vec = obs[..., -2:]
        reward_dist = -jnp.linalg.norm(vec, axis=-1)
        reward_ctrl = -jnp.square(action).sum(-1)
        reward = reward_dist + reward_ctrl
        reward = reward.reshape(-1).squeeze()
        return reward
