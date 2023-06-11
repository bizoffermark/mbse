import gym
from gym.utils import seeding
import numpy as np
from dm_control.rl.control import Environment
from typing import Optional
from gym.envs.classic_control.pendulum import angle_normalize
import math
from jax.lax import cond
import matplotlib.pyplot as plt


class SimPendulumEnv(gym.Env):
    def __init__(self, action_dim: int = 1, render_mode: str = 'rgb_array'):
        super().__init__()
        self.action_min = -1.0
        self.action_max = 1.0
        self.action_dim = action_dim
        # state = [theta, theta_dot, x, x_dot, x_ddot] 
        self.obs_dim = int(2 + 3 * action_dim)

        # self.obs_prev = np.zeros(self.obs_dim)

        self.render_mode = render_mode

        # --- Pendulum parameters ---
        self.dt = 1/30 # 30 Hz
        self.dx = 0.05 # 0.01 m
        self.L = 0.3 # 0.3 m
        self.g = 9.81 # 9.81 m/s^2
        # ---------------------------

        self.target_state = np.array([math.pi, 0.0])
        self.cost_weights = np.array([10, 1, 0.01])
        self.n_horizon = 50

    def step(self, action):
        obs = self.obs

        theta_ddot = self.solve_dynamics(obs)
        obs[1] = obs[1] + self.dt * theta_ddot # theta_dot_{t+1} = theta_dot_t + dt * theta_ddot_t
        obs[0] = angle_normalize(obs[0] + self.dt * obs[1]) # theta_{t+1} = theta_t + dt * theta_dot_t

        obs[2] = obs[2] + self.dx * action[0] # x_{t+1} = x_t + dx * action
        obs[3] = (obs[3] - self.obs[3])/self.dt # x_dot_{t+1} = (x_{t+1} - x_t)/dt
        obs[4] = (obs[3] - self.obs[3])/self.dt # x_ddot_{t+1} = (x_dot_{t+1} - x_dot_t)/dt

        self.obs = obs

        reward = self._reward_fn(obs, action)
        truncate = False
        terminate = False
        return obs, reward, terminate, truncate, {}

    def _reward_fn(self, state, action, t=0):
        # print("reward function called")
        
        return -self._cost_fn(state, action, t)
    
    def _cost_fn(self, state, action, t): 
        # print("cost function called")
        # print("state shape: ", state.shape)
        # print("action shape: ", action.shape)
        # print("state: ", state)
        # print("action: ", action)
        assert state.shape == (self.obs_dim,) and action.shape == (self.action_dim,)

        theta = state[0]
        theta_dot = state[1]
        
        qs = self.cost_weights
        target_state = self.target_state
        
        theta_star = target_state[0]
        theta_dot_star = target_state[1]

        q_theta = qs[0]
        q_theta_dot = qs[1]
        q_u = qs[2]

        dtheta = theta - theta_star
        dtheta_dot = theta_dot - theta_dot_star
        dtheta = angle_normalize(dtheta)
        
        def running_cost(dtheta, dtheta_dot, action):
            return q_theta * np.sum(dtheta ** 2) + \
                + q_theta_dot * np.sum(dtheta_dot**2) \
                + q_u * np.sum(action ** 2)
        
        def terminal_cost(dtheta, dtheta_dot, action):
            return q_theta * np.sum(dtheta ** 2) + \
                + q_theta_dot * np.sum(dtheta_dot**2)

        return cond(t == self.n_horizon, terminal_cost, running_cost, dtheta, dtheta_dot, action)

    def solve_dynamics(self, obs):
        theta = obs[0]
        x_ddot = obs[4]
        theta_ddot = -1/self.L * (x_ddot*np.cos(theta) + self.g * np.sin(theta))
        return theta_ddot
    
    @property
    def action_space(self):
        return gym.spaces.Box(self.action_min, self.action_max, (self.action_dim, ),dtype=np.float32)

    @property
    def observation_space(self):
        return gym.spaces.Box(-np.inf, np.inf, (self.obs_dim, ), dtype=np.float32)


    def reset(self, seed: [Optional] = None):
        # if seed is not None:
        #     self._np_random, seed = seeding.np_random(seed)
        obs = np.zeros(self.obs_dim)
        self.obs = obs
        self.obs_prev = obs

        return obs, {}
    
    def reset_random(self, seed: [Optional] = None):
        obs = np.zeros(self.obs_dim)
        obs[0] = np.pi/2#np.random.uniform(low=-np.pi, high=np.pi) # theta in [-pi, pi]
        self.obs = obs
        self.obs_prev = obs

        return obs, {}
    
    def seed(self, seed=None):
        self._env.task.random.seed(seed)


if __name__ == "__main__":
    env = SimPendulumEnv()
    obs, _ = env.reset_random()
    reward = env._reward_fn(obs, np.array([0.0]))
    
    n_horizon = 200

    obss = []
    rewards = []
    actions = []
    theta_ddots = []

    for i in range(n_horizon):
        action = np.array([0.0])
        theta_ddot = env.solve_dynamics(obs)
        obss.append(obs)
        actions.append(action)
        rewards.append(reward)
        theta_ddots.append(theta_ddot)
        obs, reward, terminate, truncate, _ = env.step(action)
    theta = [obs[0] for obs in obss]
    theta_dot = [obs[1] for obs in obss]
    
    x = [obs[2] for obs in obss]
    x_dot = [obs[3] for obs in obss]
    x_ddot = [obs[4] for obs in obss]
    datas = [theta, theta_dot, theta_ddots, x, x_dot, x_ddot]
    headers = ['theta', 'theta_dot', 'theta_ddot', 'x', 'x_dot', 'x_ddot']

    fig, axs = plt.subplots(len(headers), 1, figsize=(30, 30))
    for i in range(len(headers)):
        axs[i].plot(datas[i])
        axs[i].set_title(headers[i])

    fig_path = "pendulum.png"
    fig.savefig(fig_path)

