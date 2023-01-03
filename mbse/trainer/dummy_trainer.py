import os

import jax
import jax.numpy as jnp
from mbse.utils.replay_buffer import Transition, ReplayBuffer
from mbse.agents.dummy_agent import DummyAgent
import wandb
import cloudpickle
from copy import deepcopy
import numpy as np
from mbse.utils.vec_env import VecEnv
from gym import Env


class DummyTrainer(object):
    def __init__(self,
                 env: VecEnv,
                 agent: DummyAgent,
                 buffer_size: int = int(1e6),
                 max_train_steps: int = int(1e6),
                 batch_size: int = 256,
                 train_freq: int = 100,
                 train_steps: int = 100,
                 eval_freq: int = 1000,
                 seed: int = 0,
                 exploration_steps: int = int(1e4),
                 rollout_steps: int = 200,
                 eval_episodes: int = 100,
                 agent_name: str = "DummyAgent",
                 use_wandb: bool = True,
                 ):
        self.env = env
        self.num_envs = max(env.num_envs, 1)
        self.buffer = ReplayBuffer(
            obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            max_size=buffer_size
        )
        self.buffer_size = buffer_size
        self.agent = agent
        self.eval_episodes = eval_episodes
        self.agent_name = agent_name
        self.rng = jax.random.PRNGKey(seed)
        self.use_wandb = use_wandb

        self.max_train_steps = max_train_steps
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.train_steps = int(train_steps*self.num_envs)
        self.eval_freq = eval_freq
        self.exploration_steps = exploration_steps
        self.rollout_steps = rollout_steps
        self.test_env = deepcopy(self.env.envs[0])

    def train(self):
        pass

    def save_agent(self, step=0, agent_name=None):
        if self.use_wandb:
            prefix = str(step)
            name = self.agent_name if agent_name is None else agent_name
            name = name + "_" + prefix
            save_dir = os.path.join(wandb.run.dir, name)
            with open(save_dir, 'wb') as outp:
                cloudpickle.dump(self.agent, outp)

    def step_env(self, obs, policy, num_steps, rng):
        rng, reset_rng = jax.random.split(rng, 2)
        num_points = int(num_steps*self.num_envs)
        obs_shape = (num_points,) + self.env.observation_space.shape
        action_space = (num_points,) + self.env.action_space.shape
        obs_vec = np.zeros(obs_shape)
        action_vec = np.zeros(action_space)
        reward_vec = np.zeros((num_points,))
        next_obs_vec = np.zeros(obs_shape)
        done_vec = np.zeros((num_points,))
        next_rng = rng
        last_obs = obs
        last_done = False
        for step in range(num_steps):
            next_rng, actor_rng = jax.random.split(next_rng, 2)
            action = policy(obs, actor_rng)
            next_obs, reward, terminate, truncate, info = self.env.step(action)

            obs_vec[step*self.num_envs: (step+1)*self.num_envs] = obs
            action_vec[step*self.num_envs: (step+1)*self.num_envs] = action
            reward_vec[step*self.num_envs: (step+1)*self.num_envs] = reward
            next_obs_vec[step*self.num_envs: (step+1)*self.num_envs] = next_obs
            done_vec[step*self.num_envs: (step+1)*self.num_envs] = terminate
            # obs_vec = obs_vec.at[step].set(jnp.asarray(obs))
            # action_vec = action_vec.at[step].set(jnp.asarray(action))
            # reward_vec = reward_vec.at[step].set(jnp.asarray(reward))
            # next_obs_vec = next_obs_vec.at[step].set(jnp.asarray(next_obs))
            # done_vec = done_vec.at[step].set(jnp.asarray(terminate))

            # for idx, done in enumerate(dones):
            #     if done:
            #         reset_rng, next_reset_rng = jax.random.split(reset_rng, 2)
            #         reset_seed = jax.random.randint(
            #             reset_rng,
            #             (1,),
            #             minval=0,
            #             maxval=num_steps).item()
            #         obs[idx], _ = self.env.reset(seed=reset_seed)
            obs = np.concatenate([x['last_observation'].reshape(1, -1) for x in info], axis=0)
            dones = np.concatenate([x['last_done'].reshape(1, -1) for x in info], axis=0)

            last_obs = obs
            last_done = dones
        transitions = Transition(
            obs=obs_vec,
            action=action_vec,
            reward=reward_vec,
            next_obs=next_obs_vec,
            done=done_vec,
        )
        return transitions, last_obs, last_done

    def rollout_policy(self, num_steps, policy, rng):
        rng, reset_rng = jax.random.split(rng, 2)
        reset_seed = jax.random.randint(
            reset_rng,
            (1, ),
            minval=0,
            maxval=num_steps).item()
        obs, _ = self.env.reset(seed=reset_seed)
        num_points = int(num_steps * self.num_envs)
        obs_shape = (num_points,) + self.env.observation_space.shape
        action_space = (num_points,) + self.env.action_space.shape
        obs_vec = np.zeros(obs_shape)
        action_vec = np.zeros(action_space)
        reward_vec = np.zeros((num_points,))
        next_obs_vec = np.zeros(obs_shape)
        done_vec = np.zeros((num_points,))
        next_rng = rng
        for step in range(num_steps):
            next_rng, actor_rng = jax.random.split(next_rng, 2)
            action = policy(obs, actor_rng)
            next_obs, reward, terminate, truncate, info = self.env.step(action)

            obs_vec[step * self.num_envs: (step + 1) * self.num_envs] = obs
            action_vec[step * self.num_envs: (step + 1) * self.num_envs] = action
            reward_vec[step * self.num_envs: (step + 1) * self.num_envs] = reward
            next_obs_vec[step * self.num_envs: (step + 1) * self.num_envs] = next_obs
            done_vec[step * self.num_envs: (step + 1) * self.num_envs] = terminate
            # obs_vec = obs_vec.at[step].set(jnp.asarray(obs))
            # action_vec = action_vec.at[step].set(jnp.asarray(action))
            # reward_vec = reward_vec.at[step].set(jnp.asarray(reward))
            # next_obs_vec = next_obs_vec.at[step].set(jnp.asarray(next_obs))
            # done_vec = done_vec.at[step].set(jnp.asarray(terminate))
            obs = np.concatenate([x['last_observation'].reshape(1, -1) for x in info], axis=0)
            # for idx, done in enumerate(dones):
            #    if done:
            #        reset_rng, next_reset_rng = jax.random.split(reset_rng, 2)
            #        reset_seed = jax.random.randint(
            #            reset_rng,
            #            (1,),
            #            minval=0,
            #            maxval=num_steps).item()
            #        obs[idx], _ = self.env.reset(seed=reset_seed)

        transitions = Transition(
            obs=obs_vec,
            action=action_vec,
            reward=reward_vec,
            next_obs=next_obs_vec,
            done=done_vec,
        )
        return transitions

    def eval_policy(self) -> float:
        avg_reward = 0.0
        for e in range(self.eval_episodes):
            obs, _ = self.test_env.reset(seed=e)
            done = False
            while not done:
                action = self.agent.act(obs)
                next_obs, reward, terminate, truncate, info = self.test_env.step(action)
                done = terminate or truncate
                avg_reward += reward
                obs = next_obs
                if done:
                    obs, _ = self.test_env.reset(seed=e)
        avg_reward /= self.eval_episodes
        return avg_reward
