import os

import jax
import jax.numpy as jnp
from mbse.utils.replay_buffer import Transition, ReplayBuffer
from mbse.agents.dummy_agent import DummyAgent
from gym import Env
import wandb
import cloudpickle


class DummyTrainer(object):
    def __init__(self,
                 env: Env,
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
        self.train_steps = train_steps
        self.eval_freq = eval_freq
        self.exploration_steps = exploration_steps
        self.rollout_steps = rollout_steps

    @staticmethod
    def step_env(env, obs, action):
        next_obs, reward, terminate, truncate, info = env.step(action)
        done = terminate or truncate
        tran = Transition(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
        )
        return tran

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

    def rollout_policy(self, num_steps, policy, rng):
        obs, _ = self.env.reset()
        obs_shape = (num_steps, ) + self.env.observation_space.shape
        action_space = (num_steps,) + self.env.action_space.shape

        obs_vec = jnp.zeros(obs_shape)
        action_vec = jnp.zeros(action_space)
        reward_vec = jnp.zeros((num_steps, ))
        next_obs_vec = jnp.zeros(obs_shape)
        done_vec = jnp.zeros((num_steps, ))
        next_rng = rng
        for step in range(num_steps):
            next_rng, actor_rng = jax.random.split(next_rng, 2)
            action = policy(obs, actor_rng)
            next_obs, reward, terminate, truncate, info = self.env.step(action)
            done = terminate or truncate
            obs_vec = obs_vec.at[step].set(jnp.asarray(obs))
            action_vec = action_vec.at[step].set(jnp.asarray(action))
            reward_vec = reward_vec.at[step].set(jnp.asarray(reward))
            next_obs_vec = next_obs_vec.at[step].set(jnp.asarray(next_obs))
            done_vec = done_vec.at[step].set(jnp.asarray(done))
            obs = next_obs
            if done:
                obs, _ = self.env.reset()

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
            obs, _ = self.env.reset(seed=e)
            done = False
            i = 0
            while not done:
                action = self.agent.act(obs)
                next_obs, reward, terminate, truncate, info = self.env.step(action)
                done = terminate or truncate
                avg_reward += reward
                obs = next_obs
                if done:
                    obs, _ = self.env.reset(seed=e)
                i+=1
        avg_reward /= self.eval_episodes
        return avg_reward
