import numpy as np
from mbse.models.reward_model import RewardModel
from mbse.models.dynamics_model import DynamicsModel
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
from typing import Union, Optional, Any
from flax import struct


class BicycleCarReward(RewardModel):
    def __init__(self, position_cost_weight: float = 10, velocity_cost_weight: float = 1,
                 ctrl_cost_weight=0.01, goal_pos: Union[np.ndarray, jax.Array] = jnp.zeros(3), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctrl_cost_weight = ctrl_cost_weight
        self.position_cost_weight = position_cost_weight
        self.velocity_cost_weight = velocity_cost_weight
        self.goal_pos = goal_pos
        self._init_fn()

    def _init_fn(self):
        def predict(obs, action, next_obs=None, rng=None):
            return self._predict(
                obs=obs,
                action=action,
                goal_pos=self.goal_pos,
                position_cost_weight=self.position_cost_weight,
                velocity_cost_weight=self.velocity_cost_weight,
                ctrl_cost_weight=self.ctrl_cost_weight,
                next_obs=next_obs,
                rng=rng,
            )

        self.predict = jax.jit(predict)

    @staticmethod
    def _predict(obs, action, goal_pos, position_cost_weight, velocity_cost_weight, ctrl_cost_weight,
                 next_obs=None, rng=None):
        obs = obs.reshape(-1, 6)
        action = action.reshape(-1, 2)
        pos = obs[..., 0:3]
        diff = goal_pos - pos
        theta = diff[..., 2]
        sin_diff, cos_diff = jnp.sin(theta), jnp.cos(theta)
        diff.at[..., 2].set(jnp.arctan2(sin_diff, cos_diff))
        pos_cost = position_cost_weight * jnp.sum(jnp.square(diff), axis=-1)
        velocity_cost = velocity_cost_weight * jnp.sum(jnp.square(obs[..., 3:]), axis=-1)
        input_cost = ctrl_cost_weight * jnp.sum(jnp.square(action), axis=-1)
        # reward = (jnp.sum(jnp.abs(diff), axis=-1) < 0.1) * 100 + input_cost
        reward = - (pos_cost + velocity_cost + input_cost)
        return reward


@struct.dataclass
class CarParams:
    m: float = 0.05
    l: float = 0.06
    a: float = 0.25
    b: float = 0.01
    g: float = 9.81
    d_f: float = 0.2
    c_f: float = 1.25
    b_f: float = 2.5
    d_r: float = 0.2
    c_r: float = 1.25
    b_r: float = 2.5
    c_m_1: float = 0.2
    c_m_2: float = 0.05
    c_rr: float = 0.0
    c_d_max: float = 0.1
    c_d_min: float = 0.01
    tv_p: float = 0.0
    q_pos: float = 0.1
    q_v: float = 0.1
    r_u: float = 1
    room_boundary: float = 80.0
    velocity_limit: float = 100.0
    max_steering: float = 0.25
    dt: float = 0.01
    control_freq: int = 5

    def _get_x_com(self):
        x_com = self.l * (self.a + 2) / (3 * (self.a + 1))
        return x_com

    def _get_moment_of_intertia(self):
        # Moment of inertia around origin
        a = self.a
        b = self.b
        m = self.m
        l = self.l
        i_o = m / (6 * (1 + a)) * ((a ** 3 + a ** 2 + a + 1) * (b ** 2) + (l ** 2) * (a + 3))
        x_com = self._get_x_com()
        i_com = i_o - self.m * (x_com ** 2)
        return i_com


class BicycleCarModel(DynamicsModel):

    def __init__(self, reward_model: BicycleCarReward = BicycleCarReward(), params: CarParams = CarParams()):
        super().__init__()
        self.reward_model = reward_model
        self.params = params
        predict = lambda x, u, rng=None: self._predict(x, u, params=self.params,
                                                                    control_freq=self.params.control_freq)
        self.predict = jax.jit(predict)


    def _ode_dyn(self, x: jax.Array, u: jax.Array, params: CarParams):
        theta, v_x, v_y, w = x[..., 2], x[..., 3], x[..., 4], x[..., 5]
        p_x_dot = v_x * jnp.cos(theta) - v_y * jnp.sin(theta)
        p_y_dot = v_x * jnp.sin(theta) + v_y * jnp.cos(theta)
        theta_dot = w
        p_x_dot = jnp.array([p_x_dot, p_y_dot, theta_dot]).T

        accelerations = self._accelerations_dyn(x, u, params).T

        x_dot = jnp.concatenate([p_x_dot, accelerations], axis=-1)
        return x_dot

    @staticmethod
    def _accelerations_dyn(x: jax.Array, u: jax.Array, params: CarParams):
        i_com = params._get_moment_of_intertia()
        theta, v_x, v_y, w = x[..., 2], x[..., 3], x[..., 4], x[..., 5]
        m = params.m
        l = params.l
        d_f = params.d_f * params.g * m
        d_r = params.d_r * params.g * m
        c_f = params.c_f
        c_r = params.c_r
        b_f = params.b_f
        b_r = params.b_r
        c_m_1 = params.c_m_1
        c_m_2 = params.c_m_2
        c_d_max = params.c_d_max
        c_d_min = params.c_d_min
        c_rr = params.c_rr
        a = params.a
        l_r = params._get_x_com()
        l_f = l - l_r
        tv_p = params.tv_p

        c_d = c_d_min + (c_d_max - c_d_min) * a

        delta, d = u[..., 0], u[..., 1]
        delta = jnp.clip(delta, a_min=-params.max_steering,
                         a_max=params.max_steering)
        d = jnp.clip(d, a_min=-1, a_max=1)

        w_tar = delta * v_x / l

        alpha_f = -jnp.arctan(
            (w * l_f + v_y) /
            (v_x + 1e-6)
        ) + delta
        alpha_r = jnp.arctan(
            (w * l_r - v_y) /
            (v_x + 1e-6)
        )
        f_f_y = d_f * jnp.sin(c_f * jnp.arctan(b_f * alpha_f))
        f_r_y = d_r * jnp.sin(c_r * jnp.arctan(b_r * alpha_r))
        f_r_x = (c_m_1 - c_m_2 * v_x) * d - c_rr - c_d * v_x * v_x

        v_x_dot = (f_r_x - f_f_y * jnp.sin(delta) + m * v_y * w) / m
        v_y_dot = (f_r_y + f_f_y * jnp.cos(delta) - m * v_x * w) / m
        w_dot = (f_f_y * l_f * jnp.cos(delta) - f_r_y * l_r + tv_p * (w_tar - w)) / i_com

        acceleration = jnp.asarray([v_x_dot, v_y_dot, w_dot])
        return acceleration

    @staticmethod
    def _ode_kin(x: jax.Array, u: jax.Array, params: CarParams):
        p_x, p_y, theta, v_x = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        m = params.m
        l = params.l
        c_m_1 = params.c_m_1
        c_m_2 = params.c_m_2
        c_d_max = params.c_d_max
        c_d_min = params.c_d_min
        c_rr = params.c_rr
        a = params.a
        l_r = params._get_x_com()

        c_d = c_d_min + (c_d_max - c_d_min) * a

        delta, d = jnp.atleast_1d(u[..., 0]), jnp.atleast_1d(u[..., 1])

        d_0 = (c_rr + c_d * v_x * v_x - c_m_2 * v_x) / (c_m_1 - c_m_2 * v_x)
        d_slow = jnp.maximum(d, d_0)
        d_fast = d
        slow_ind = v_x <= 0.1
        d_applied = d_slow * slow_ind + d_fast * (~slow_ind)
        f_r_x = ((c_m_1 - c_m_2 * v_x) * d_applied - c_rr - c_d * v_x * v_x) / m

        beta = jnp.arctan(l_r * jnp.arctan(delta) / l)
        p_x_dot = v_x * jnp.cos(beta + theta)  # s_dot
        p_y_dot = v_x * jnp.sin(beta + theta)  # d_dot
        w = v_x * jnp.sin(beta) / l_r

        dx_kin = jnp.concatenate([p_x_dot, p_y_dot, w,
                                  f_r_x], axis=0)
        return dx_kin.reshape(-1, 4)

    def _ode(self, x: jax.Array, u: jax.Array, params: CarParams):
        """
        Using kinematic model with blending: https://arxiv.org/pdf/1905.05150.pdf
        Code based on: https://github.com/manish-pra/copg/blob/4a370594ab35f000b7b43b1533bd739f70139e4e/car_racing_simulator/VehicleModel.py#L381
        """
        v_x = jnp.atleast_1d(x[..., 3])
        blend_ratio = (v_x - 0.3) / (0.2)
        l = params.l
        l_r = params._get_x_com()

        lambda_blend = jnp.minimum(
            jnp.maximum(blend_ratio, jnp.zeros_like(blend_ratio)), jnp.ones_like(blend_ratio)
        )

        # if lambda_blend < 1:
        v_x = jnp.atleast_1d(x[..., 3])
        v_y = jnp.atleast_1d(x[..., 4])
        x_kin = jnp.concatenate([jnp.atleast_1d(x[..., 0]),
                                 jnp.atleast_1d(x[..., 1]), jnp.atleast_1d(x[..., 2]), v_x], axis=-1)
        x_kin = x_kin.reshape(-1, 4)
        dxkin = self._ode_kin(x_kin, u, params)
        delta = u[..., 0]
        beta = l_r * jnp.tan(delta) / l
        v_x_state = dxkin[..., 3]
        v_y_state = dxkin[..., 3] * beta
        w = v_x_state * jnp.tan(delta) / l
        dx_kin_full = jnp.asarray([dxkin[..., 0], dxkin[..., 1],
                                   dxkin[..., 2], v_x_state, v_y_state,
                                   w])
        dx_kin_full = dx_kin_full.reshape(-1, 6)
        dxdyn = self._ode_dyn(x=x, u=u, params=params)
        mul = jax.vmap(lambda x, y: x*y)
        return mul(lambda_blend, jnp.atleast_2d(dxdyn)) + mul(1 - lambda_blend, dx_kin_full).reshape(-1, 6)

    @partial(jax.jit, static_argnums=(0, 4))
    def next_step(self, x: jax.Array, u: jax.Array, params: CarParams, control_freq: int):
        carry = [x, u]

        def step(carry, outs):
            x, u = carry[0], carry[1]
            x_next = self.rk_integrator(x, u, params)
            carry = [x_next, u]
            outs = [x_next]
            return carry, outs

        _, outs = jax.lax.scan(step, carry, None, length=control_freq)
        x = outs[0][-1]
        return x

    def rk_integrator(self, x: jax.Array, u: jax.Array, params: CarParams):
        k1 = self._ode(x, u, params)
        k2 = self._ode(x + params.dt / 2. * k1, u, params)
        k3 = self._ode(x + params.dt / 2. * k2, u, params)
        k4 = self._ode(x + params.dt * k3, u, params)
        x_next = x + params.dt * (k1 / 6. + k2 / 3. + k3 / 3. + k4 / 6.)
        x_next.at[0:2].set(jnp.clip(x_next[0:2], a_min=-params.room_boundary, a_max=params.room_boundary))
        theta = x_next[..., 2]
        sin_theta, cos_theta = jnp.sin(theta), jnp.cos(theta)
        x_next = x_next.at[..., 2].set(jnp.arctan2(sin_theta, cos_theta))
        x_next = x_next.at[..., 3:].set(jnp.clip(x_next[..., 3:], a_min=-params.velocity_limit, a_max=params.velocity_limit))
        return x_next

    @partial(jax.jit, static_argnums=(0, 4))
    def _predict(self, x: jax.Array, u: jax.Array, params: CarParams, control_freq: int):
        action = u
        action = action.at[..., 0].set(action[..., 0] * params.max_steering)
        next_state = self.next_step(x, action, params, control_freq)
        return next_state

    def evaluate(self,
                 parameters,
                 obs,
                 action,
                 rng,
                 sampling_idx=None,
                 alpha: Union[jnp.ndarray, float] = 1.0,
                 bias_obs: Union[jnp.ndarray, float] = 0.0,
                 bias_act: Union[jnp.ndarray, float] = 0.0,
                 bias_out: Union[jnp.ndarray, float] = 0.0,
                 scale_obs: Union[jnp.ndarray, float] = 1.0,
                 scale_act: Union[jnp.ndarray, float] = 1.0,
                 scale_out: Union[jnp.ndarray, float] = 1.0):
        next_state = self.predict(x=obs, u=action, rng=rng)
        reward = self.reward_model.predict(obs=obs, action=action, next_obs=next_state)
        return next_state, reward
