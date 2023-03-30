import jax.numpy as jnp
import jax
from mbse.models.reward_model import RewardModel
from functools import partial

_RUN_SPEED = 10


class HalfCheetahReward(RewardModel):
    """Get Pendulum Reward."""

    def __init__(self, forward_velocity_weight: float = 1.0, ctrl_cost_weight: float = 0.1):
        super().__init__()
        self.ctrl_cost_weight = ctrl_cost_weight
        self.forward_velocity_weight = forward_velocity_weight

    @partial(jax.jit, static_argnums=0)
    def predict(self, obs, action, next_obs, rng=None):
        # reward_ctrl = -self.ctrl_cost_weight * jnp.square(action).sum(axis=-1)
        # reward_run = self.forward_velocity_weight * (next_obs[..., 0] - 0.0 * jnp.square(next_obs[..., 2]))
        # reward = reward_run + reward_ctrl
        speed = next_obs[..., 0]
        reward = tolerance(speed,
                           bounds=(_RUN_SPEED, float('inf')),
                           margin=_RUN_SPEED,
                           value_at_margin=0)
        return reward


def tolerance(speed, bounds=(0.0, 0.0), margin=0.0,
              value_at_margin = 0.1):
    """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.

  Args:
    x: A scalar or numpy array.
    bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
      the target interval. These can be infinite if the interval is unbounded
      at one or both ends, or they can be equal to one another if the target
      value is exact.
    margin: Float. Parameter that controls how steeply the output decreases as
      `x` moves out-of-bounds.
      * If `margin == 0` then the output will be 0 for all values of `x`
        outside of `bounds`.
      * If `margin > 0` then the output will decrease sigmoidally with
        increasing distance from the nearest bound.
    sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
       'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
    value_at_margin: A float between 0 and 1 specifying the output value when
      the distance from `x` to the nearest bound is equal to `margin`. Ignored
      if `margin == 0`.

  Returns:
    A float or numpy array with values between 0.0 and 1.0.

  Raises:
    ValueError: If `bounds[0] > bounds[1]`.
    ValueError: If `margin` is negative.
  """

    def _sigmoid(x, value_at_1):
        scale = 1 - value_at_1
        scaled_x = x * scale
        return jnp.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

    lower, upper = bounds
    if lower > upper:
        raise ValueError('Lower bound must be <= upper bound.')
    if margin < 0:
        raise ValueError('`margin` must be non-negative.')

    in_bounds = jnp.logical_and(lower <= speed, speed <= upper)
    if margin == 0:
        value = jnp.where(in_bounds, 1.0, 0.0)
    else:
        d = jnp.where(speed < lower, lower - speed, speed - upper) / margin
        value = jnp.where(in_bounds, 1.0, _sigmoid(d, value_at_margin))

    return float(value) if jnp.isscalar(speed) else value
