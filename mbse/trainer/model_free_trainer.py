import os

import jax
import jax.numpy as jnp
from jax import jit, vmap, random
from mbse.utils.replay_buffer import Transition, ReplayBuffer
from mbse.agents.actor_critic.sac import SACAgent
from gym import Env
import wandb
import copy
import cloudpickle
from tqdm import tqdm
# from flax.training import train_state, checkpoints


class ModelFreeTrainer(object):
    def __init__(self,
                 env: Env,
                 agent: SACAgent,
                 buffer_size: int = 1e6,
                 max_train_steps: int = 1e6,
                 batch_size: int = 256,
                 train_freq: int = 100,
                 train_steps: int = 100,
                 eval_freq: int = 1000,
                 seed: int = 0,
                 exploration_steps: int = 1e4,
                 rollout_steps: int = 200,
                 eval_episodes: int = 100,
                 agent_name: str = "SACAgent",
                 use_wandb: bool = True,
                 ):
        self.env = env
        self.buffer = ReplayBuffer(
            obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            max_size=buffer_size
        )

        self.agent = agent
        self.max_train_steps = max_train_steps
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.train_steps = train_steps
        self.eval_freq = eval_freq
        self.exploration_steps = exploration_steps
        self.rollout_steps = rollout_steps
        self.eval_episodes = eval_episodes
        self.agent_name = agent_name
        self.rng = jax.random.PRNGKey(seed)
        self.use_wandb = use_wandb

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
        # obs, _ = self.env.reset()
        if self.use_wandb:
            wandb.define_metric('env_steps')
            wandb.define_metric('train_steps')
        average_reward = self.eval_policy()
        best_performance = average_reward
        reward_log = {
            'env_steps': 0,
            'average_reward': average_reward
        }
        train_steps = 0
        self.save_agent(0)
        if self.use_wandb:
            wandb.log(reward_log)
        policy = lambda x, y: self.env.action_space.sample()
        transitions = self.rollout_policy(self.exploration_steps, policy, self.rng)
        self.buffer.add(transition=transitions)
        #for i in range(self.exploration_steps):
        #    action = self.env.action_space.sample()
        #    tran = self.step_env(self.env, obs, action)
        #    self.buffer.add(tran)
        #    obs = tran.next_obs
        #    if tran.done:
        #        obs, _ = self.env.reset()

        obs, _ = self.env.reset()
        rng_keys = random.split(self.rng, self.max_train_steps+1)
        self.rng = rng_keys[0]
        rng_keys = rng_keys[1:]
        for step in tqdm(range(self.max_train_steps)):
            actor_rng, train_rng = random.split(rng_keys[step], 2)
            policy = self.agent.act
            transitions = self.rollout_policy(self.rollout_steps, policy, actor_rng)
            self.buffer.add(transitions)

            if step % self.train_freq == 0:
                for _ in range(self.train_steps):
                    train_rng, agent_rng, buffer_rng = random.split(train_rng, 3)
                    batch = self.buffer.sample(self.batch_size, buffer_rng)
                    summary = self.agent.train_step(
                        agent_rng,
                        batch,
                    )
                    train_steps += 1
                    train_log = {
                        'train_steps': train_steps,
                        'env_steps': step,
                        'actor_loss': summary.actor_loss,
                        'entropy': summary.entropy,
                        'critic_loss': summary.critic_loss,
                        'alpha_loss': summary.alpha_loss,
                        'log_alpha': summary.log_alpha,
                    }
                    if self.use_wandb:
                        wandb.log(train_log)

            # Evaluate episode
            if (step + 1) % self.eval_freq == 0:
                eval_reward = self.eval_policy()
                reward_log = {
                    'env_steps': step,
                    'average_reward': eval_reward
                }
                if self.use_wandb:
                    wandb.log(reward_log)
                if eval_reward > best_performance:
                    best_performance = eval_reward
                    self.save_agent(step)

            step += 1
        self.save_agent(step, agent_name="final_agent")

    def save_agent(self, step=0, agent_name=None):
        if self.use_wandb:
            prefix = str(step)
            name = self.agent_name if agent_name is None else agent_name
            name = name + "_" + prefix
            save_dir = os.path.join(wandb.run.dir, name)
            #strain_state
            #state = train_state.TrainState.create(apply_fn=self.agent.actor.apply,
            #                                      params=self.agent.actor_params,
            #                                      tx=self.agent.actor_optimizer)
            #checkpoints.save_checkpoint(ckpt_dir=save_dir, target=state, step=step)
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
            tran = self.step_env(self.env, obs, action)
            #transitions.append(tran)
            obs_vec = obs_vec.at[step].set(tran.obs)
            action_vec = action_vec.at[step].set(tran.action)
            reward_vec = reward_vec.at[step].set(tran.reward)
            next_obs_vec = next_obs_vec.at[step].set(tran.next_obs)
            done_vec = done_vec.at[step].set(tran.done)
            # self.buffer.add(tran)
            obs = tran.next_obs
            if tran.done:
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
        for _ in range(self.eval_episodes):
            obs, _ = self.env.reset()
            done = False
            while not done:
                action = self.agent.act(obs)
                tran = self.step_env(self.env, obs, action)
                avg_reward += tran.reward
                done = tran.done
                obs = tran.next_obs
                if tran.done:
                    obs, _ = self.env.reset()
        avg_reward /= self.eval_episodes
        return avg_reward














