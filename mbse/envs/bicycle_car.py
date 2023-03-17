import gym
from gym import spaces
from gym import Env
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gym.wrappers.record_video import RecordVideo


class TrajectoryGraph:
    """A stock trading visualization using matplotlib made to render
      OpenAI gym environments"""

    def __init__(self, title="RC Car Trajectory"):
        # Create a figure on screen and set the title
        # Create top subplot for net worth axis
        self.pos = []
        self.fig, self.pos_axis = plt.subplots()
        self.fig.suptitle(title)
        self.pos_axis.set_xticks([])
        self.pos_axis.set_yticks([])
        self.rectangle_size = 0.05
        self.canvas = self.fig.canvas
        self._bg = None
        self.fr_number = self.pos_axis.annotate(
            "0",
            (0, 1),
            xycoords="axes fraction",
            xytext=(0, 0),
            textcoords="offset points",
            ha="left",
            va="top",
            animated=True,
        )
        (self.ln,) = self.pos_axis.plot(0.0, 0.0, animated=True)
        (self.pn,) = self.pos_axis.plot(0.0, 0.0, 'ro', animated=True)
        (self.gn,) = self.pos_axis.plot(0.0, 0.0, 'o', animated=True, color='gold')
        self.car_pos = self.pos_axis.annotate(f'x_pos: {0:.2f}, y_pos: {0:.2f}'.format(0.0, 0.0),
                                              (0, 1),
                                              xycoords="axes fraction",
                                              xytext=(0, 0.95),
                                              ha="left",
                                              va="top",
                                              animated=True, )

        self._artists = []
        self.add_artist(self.fr_number)
        self.add_artist(self.ln)
        self.add_artist(self.pn)
        self.add_artist(self.gn)
        self.add_artist(self.car_pos)
        self.cid = self.canvas.mpl_connect("draw_event", self.on_draw)
        # Show the graph without blocking the rest of the program
        plt.show(block=False)
        plt.pause(0.001)

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def add_rectangle(self, car_pos):
        rect = patches.Rectangle(xy=(car_pos[0] - self.rectangle_size, car_pos[1] - self.rectangle_size / 2.0),
                                 angle=car_pos[2] * 180.0 / np.pi,
                                 width=self.rectangle_size * 2.0,
                                 height=self.rectangle_size,
                                 alpha=0.5,
                                 rotation_point='center',
                                 edgecolor='red',
                                 facecolor='none',
                                 animated=True,
                                 )
        self.pos_axis.add_patch(rect)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
        cv.flush_events()
        data = np.frombuffer(cv.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(cv.get_width_height()[::-1] + (3,))
        return data

    def render(self, current_step, car_pos, goal_pos, window_size=40):
        self.pos.append(car_pos)
        pos = np.asarray(self.pos)
        self.pos_axis.clear()
        window_start = 0
        step_range = range(window_start, current_step + 1)
        theta, v_x, v_y = car_pos[2], car_pos[3], car_pos[4]
        p_x_dot = v_x * np.cos(theta) - v_y * np.sin(theta)
        p_y_dot = v_x * np.sin(theta) + v_y * np.cos(theta)
        self.pn.set_data(car_pos[0], car_pos[1])
        self.pn.set_marker(marker=[3, 0, car_pos[2] * 180.0/np.pi - 90.0])
        self.gn.set_data(goal_pos[0], goal_pos[1])
        self.fr_number.set_text("frame: {j}".format(j=current_step))
        self.car_pos.set_text("x_pos: {x}, y_pos: {y}".format(x=np.around(car_pos[0], 2), y=np.around(car_pos[1], 2)))
        self.ln.set_data(pos[step_range, 0], pos[step_range, 1])
        min_y = min(min(pos[:, 1]), goal_pos[1])
        y_lim_min = min_y / 1.25 if min_y >= 0 else min_y * 1.25
        max_y = max(max(pos[:, 1]), goal_pos[1])
        y_lim_max = max_y / 1.25 if max_y < 0 else max_y * 1.25
        self.pos_axis.set_ylim(
            y_lim_min,
            y_lim_max)
        min_x = min(min(pos[:, 0]), goal_pos[0])
        x_lim_min = min_x / 1.25 if min_x >= 0 else min_x * 1.25
        max_x = max(max(pos[:, 0]), goal_pos[0])
        x_lim_max = max_x / 1.25 if max_x < 0 else max_x * 1.25
        self.pos_axis.set_xlim(
            x_lim_min,
            x_lim_max)
        self.pos_axis.set_xticks([])
        self.pos_axis.set_yticks([])
        data = self.update()

        return data


class BicycleEnv(Env):

    def __init__(self,
                 room_boundary: float = 80.0,
                 velocity_limit: float = 100.0,
                 max_steering: float = 0.25,
                 dt: float = 0.01,
                 control_freq: int = 5,
                 goal_pos: np.ndarray = np.asarray([20, 20, 0.0]),
                 _np_random: Optional[np.random.Generator] = None,
                 m: float = 0.05,
                 l: float = 0.06,
                 a: float = 0.25,
                 b: float = 0.01,
                 g: float = 9.81,
                 d_f: float = 0.2,
                 c_f: float = 1.25,
                 b_f: float = 2.5,
                 d_r: float = 0.2,
                 c_r: float = 1.25,
                 b_r: float = 2.5,
                 c_m_1: float = 0.2,
                 c_m_2: float = 0.05,
                 c_rr: float = 0.0,
                 c_d_max: float = 0.1,
                 c_d_min: float = 0.01,
                 tv_p: float = 0.0,
                 q_pos: float = 0.1,
                 q_v: float = 0.1,
                 r_u: float = 1,
                 render_mode: str = 'rgb_array',
                 ):
        super(BicycleEnv).__init__()
        self.render_mode = render_mode
        self.room_boundary = room_boundary
        self.velocity_limit = velocity_limit
        high = np.asarray([room_boundary,
                           room_boundary,
                           np.pi,
                           velocity_limit,
                           velocity_limit,
                           velocity_limit]
                          )
        low = -high
        self.observation_space = spaces.Box(
            high=high,
            low=low,
        )
        self.dim_state = (6,)
        self.dim_action = (2,)
        self.max_steering = max_steering
        high = np.asarray([max_steering, 1])
        low = -high
        self.action_space = spaces.Box(
            high=high,
            low=low,
        )
        self._np_random = _np_random
        self.m, self.l, self.a, self.b, self.g, self.tv_p = m, l, a, b, g, tv_p
        self.d_f, self.c_f, self.b_f = d_f, c_f, b_f
        self.d_r, self.c_r, self.b_r = d_r, c_r, b_r
        self.c_m_1, self.c_m_2, self.c_rr = c_m_1, c_m_2, c_rr
        self.c_d_max, self.c_d_min = c_d_max, c_d_min

        self.dt, self.control_freq = dt, control_freq
        self.state = np.zeros(self.dim_state)
        self.goal_pos = goal_pos
        self.q_pos = q_pos
        self.q_v = q_v
        self.r_u = r_u
        self.current_step = 0
        self.visualization = None
        self.window_size = 40

    def _get_x_com(self):
        x_com = self.l * (self.a + 2) / (3 * (self.a + 1))
        return x_com

    def _get_moment_of_intertia(self):
        # Moment of inertia around origin
        a = self.a
        assert (0 < a <= 1), "a must be between 0 and 1."
        b = self.b
        m = self.m
        l = self.l
        i_o = m / (6 * (1 + a)) * ((a ** 3 + a ** 2 + a + 1) * (b ** 2) + (l ** 2) * (a + 3))
        x_com = self._get_x_com()
        i_com = i_o - self.m * (x_com ** 2)
        return i_com

    def _ode_dyn(self, x, u):
        # state = [p_x, p_y, theta, v_x, v_y, w]. Velocities are in local coordinate frame.
        # Inputs: [\delta, d] -> \delta steering angle and d duty cycle of the electric motor.
        theta, v_x, v_y, w = x[2], x[3], x[4], x[5]
        p_x_dot = v_x * np.cos(theta) - v_y * np.sin(theta)
        p_y_dot = v_x * np.sin(theta) + v_y * np.cos(theta)
        theta_dot = w
        p_x_dot = np.array([p_x_dot, p_y_dot, theta_dot])

        accelerations = self._accelerations_dyn(x, u)

        x_dot = np.concatenate([p_x_dot, accelerations], axis=-1)
        return x_dot

    def _accelerations_dyn(self, x, u):
        i_com = self._get_moment_of_intertia()
        theta, v_x, v_y, w = x[2], x[3], x[4], x[5]
        m = self.m
        l = self.l
        d_f = self.d_f * self.g * m
        d_r = self.d_r * self.g * m
        c_f = self.c_f
        c_r = self.c_r
        b_f = self.b_f
        b_r = self.b_r
        c_m_1 = self.c_m_1
        c_m_2 = self.c_m_2
        c_d_max = self.c_d_max
        c_d_min = self.c_d_min
        c_rr = self.c_rr
        a = self.a
        l_r = self._get_x_com()
        l_f = l - l_r
        tv_p = self.tv_p

        c_d = c_d_min + (c_d_max - c_d_min) * a

        delta, d = u[0], u[1]
        delta = np.clip(delta, a_min=-self.max_steering,
                        a_max=self.max_steering)
        d = np.clip(d, a_min=-1, a_max=1)

        w_tar = delta * v_x / l

        alpha_f = -np.arctan(
            (w * l_f + v_y) /
            (v_x + 1e-6)
        ) + delta
        alpha_r = np.arctan(
            (w * l_r - v_y) /
            (v_x + 1e-6)
        )
        f_f_y = d_f * np.sin(c_f * np.arctan(b_f * alpha_f))
        f_r_y = d_r * np.sin(c_r * np.arctan(b_r * alpha_r))
        f_r_x = (c_m_1 - c_m_2 * v_x) * d - c_rr - c_d * v_x * v_x

        v_x_dot = (f_r_x - f_f_y * np.sin(delta) + m * v_y * w) / m
        v_y_dot = (f_r_y + f_f_y * np.cos(delta) - m * v_x * w) / m
        w_dot = (f_f_y * l_f * np.cos(delta) - f_r_y * l_r + tv_p * (w_tar - w)) / i_com

        acceleration = np.array([v_x_dot, v_y_dot, w_dot])
        return acceleration

    def _ode_kin(self, x, u):
        p_x, p_y, theta, v_x = x[0], x[1], x[2], x[3]
        m = self.m
        l = self.l
        c_m_1 = self.c_m_1
        c_m_2 = self.c_m_2
        c_d_max = self.c_d_max
        c_d_min = self.c_d_min
        c_rr = self.c_rr
        a = self.a
        l_r = self._get_x_com()

        c_d = c_d_min + (c_d_max - c_d_min) * a

        delta, d = u[0], u[1]

        d_0 = (c_rr + c_d * v_x * v_x - c_m_2 * v_x) / (c_m_1 - c_m_2 * v_x)
        d_slow = np.maximum(d, d_0)
        d_fast = d
        slow_ind = v_x <= 0.1
        d_applied = d_slow * slow_ind + d_fast * (~slow_ind)
        f_r_x = ((c_m_1 - c_m_2 * v_x) * d_applied - c_rr - c_d * v_x * v_x) / m

        beta = np.arctan(l_r * np.arctan(delta) / l)
        p_x_dot = v_x * np.cos(beta + theta)  # s_dot
        p_y_dot = v_x * np.sin(beta + theta)  # d_dot
        w = v_x * np.sin(beta) / l_r

        dx_kin = np.asarray([p_x_dot, p_y_dot, w, f_r_x])
        return dx_kin

    def _ode(self, x, u):
        """
        Using kinematic model with blending: https://arxiv.org/pdf/1905.05150.pdf
        Code based on: https://github.com/manish-pra/copg/blob/4a370594ab35f000b7b43b1533bd739f70139e4e/car_racing_simulator/VehicleModel.py#L381
        """
        v_x = x[3]
        blend_ratio = (v_x - 0.3) / (0.2)
        l = self.l
        l_r = self._get_x_com()

        lambda_blend = np.min(np.asarray([
            np.max(np.asarray([blend_ratio, 0])), 1])
        )

        if lambda_blend < 1:
            v_x = x[3]
            v_y = x[4]
            # x_kin = np.asarray([x[0], x[1], x[2], np.sqrt(v_x ** 2 + v_y ** 2)])
            x_kin = np.asarray([x[0], x[1], x[2], v_x])
            dxkin = self._ode_kin(x_kin, u)
            delta = u[0]
            # beta = np.arctan(l_r * np.tan(delta) / l)
            beta = l_r * np.tan(delta) / l
            v_x_state = dxkin[3]
            v_y_state = dxkin[3] * beta  # V*sin(beta)
            # w = v_x_state * np.arctan(delta) / l
            w = v_x_state * np.tan(delta) / l
            dx_kin_full = np.asarray([dxkin[0], dxkin[1], dxkin[2], v_x_state, v_y_state, w])

            if lambda_blend == 0:
                return dx_kin_full

        if lambda_blend > 0:
            dxdyn = self._ode_dyn(x=x, u=u)

            if lambda_blend == 1:
                return dxdyn

        return lambda_blend * dxdyn + (1 - lambda_blend) * dx_kin_full

    def next_step(self, x, u):
        for step in range(self.control_freq):
            x = self.rk_integrator(x, u)
        return x

    def rk_integrator(self, x, u):
        k1 = self._ode(x, u)
        k2 = self._ode(x + self.dt / 2. * k1, u)
        k3 = self._ode(x + self.dt / 2. * k2, u)
        k4 = self._ode(x + self.dt * k3, u)
        x_next = x + self.dt * (k1 / 6. + k2 / 3. + k3 / 3. + k4 / 6.)
        x_next[0:2] = np.clip(x_next[0:2], a_min=-self.room_boundary, a_max=self.room_boundary)
        theta = x_next[2]
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        x_next[2] = np.arctan2(sin_theta, cos_theta)
        x_next[3:] = np.clip(x_next[3:], a_min=-self.velocity_limit, a_max=self.velocity_limit)
        return x_next

    def reward(self, x, u):
        pos = x[0:3]
        diff = self.goal_pos - pos
        theta = diff[2]
        sin_diff, cos_diff = np.sin(theta), np.cos(theta)
        diff[2] = np.arctan2(sin_diff, cos_diff)
        pos_cost = self.q_pos * np.sum(np.square(diff))
        velocity_cost = self.q_v * np.sum(np.square(x[3:]))
        input_cost = self.r_u * np.sum(np.square(u))
        reward = - (pos_cost + velocity_cost + input_cost)
        return reward

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed, options=options)
        self.state = np.zeros(self.dim_state)
        self.current_step = 0
        self.visualization = None
        return self.state, {}

    def step(self, action):
        next_state = self.next_step(self.state, action)
        reward = self.reward(self.state, action)
        self.current_step += 1
        if self.render_mode == "human":
            self.render()
        self.state = next_state
        return next_state, reward, False, False, {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        if self.visualization is None:
            self.visualization = TrajectoryGraph()

        return self.visualization.render(self.current_step, self.state, self.goal_pos,
                                         window_size=min(self.current_step, self.window_size))


if __name__ == "__main__":
    def simulate_car(k_p=1, k_d=0.6, horizon=500):
        env = BicycleEnv()
        env = RecordVideo(env, video_folder='./', episode_trigger=lambda x: True)
        x, _ = env.reset()
        goal = env.goal_pos
        x_traj = np.zeros([horizon, 2])
        for h in range(horizon):
            pos_error = goal[0:2] - x[0:2]
            goal_direction = np.arctan2(pos_error[1], pos_error[0])
            goal_dist = np.sqrt(pos_error[0] ** 2 + pos_error[1] ** 2)
            velocity = np.sqrt(x[3] ** 2 + x[4] ** 2)
            s = np.clip(0.01 * goal_direction, a_min=-0.3, a_max=0.3)
            d = np.clip(k_p * goal_dist - k_d * velocity, a_min=-1, a_max=1)
            u = np.asarray([s, d])
            x, reward, _, _, _ = env.step(u)
            # env.render()
        return x_traj


    x_traj = simulate_car()
    check = True
