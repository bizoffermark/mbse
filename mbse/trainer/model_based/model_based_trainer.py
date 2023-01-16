import jax.random

from mbse.trainer.dummy_trainer import DummyTrainer
from mbse.agents.model_based.model_based_agent import ModelBasedAgent
import wandb
from jax import random
import numpy as np
from tqdm import tqdm
from mbse.utils.replay_buffer import ReplayBuffer
from mbse.utils.replay_buffer import Transition


class ModelBasedTrainer(DummyTrainer):
    def __init__(self,
                 agent_name: str = "OffPolicyAgent",
                 *args,
                 **kwargs
                 ):

        super(ModelBasedTrainer, self).__init__(agent_name=agent_name, *args, **kwargs)
        assert isinstance(self.agent, ModelBasedAgent), "Only Model based agents are allowed"
        self.validation_buffer = ReplayBuffer(
            obs_shape=self.env.observation_space.shape,
            action_shape=self.env.action_space.shape,
            max_size=self.buffer_size
        )

    def train(self):
        if self.use_wandb:
            wandb.define_metric('env_steps')
            wandb.define_metric('train_steps')
        self.rng, eval_rng = random.split(self.rng, 2)
        eval_rng, curr_eval = random.split(eval_rng, 2)
        average_reward = self.eval_policy(rng=curr_eval)
        best_performance = average_reward
        reward_log = {
            'env_steps': 0,
            'average_reward': average_reward
        }
        train_steps = 0
        self.save_agent(0)
        if self.use_wandb:
            wandb.log(reward_log)
        policy = lambda x, y: np.concatenate([self.env.action_space.sample().reshape(1, -1)
                                              for s in range(self.num_envs)], axis=0)
        self.rng, train_rng, val_rng = jax.random.split(self.rng, 3)
        transitions = self.rollout_policy(self.exploration_steps, policy, train_rng)
        self.buffer.add(transition=transitions)
        val_transitions = self.rollout_policy(self.exploration_steps*10, policy, val_rng)
        self.validation_buffer.add(val_transitions)
        rng_keys = random.split(self.rng, self.max_train_steps + 1)
        self.rng = rng_keys[0]
        rng_keys = rng_keys[1:]
        learning_steps = int(self.max_train_steps/(self.rollout_steps*self.num_envs))
        training_steps = int(self.train_steps*self.num_envs)
        rng_key, reset_rng = random.split(rng_keys[0], 2)
        rng_keys = rng_keys.at[0].set(rng_key)
        reset_seed = random.randint(
            reset_rng,
            (1,),
            minval=0,
            maxval=int(learning_steps*self.rollout_steps)).item()
        obs, _ = self.env.reset(seed=reset_seed)
        for step in tqdm(range(learning_steps)):
            actor_rng, train_rng = random.split(rng_keys[step], 2)
            policy = self.agent.act
            actor_rng, val_rng = random.split(actor_rng, 2)
            transitions, obs, done = self.step_env(obs, policy, self.rollout_steps, actor_rng)
            self.buffer.add(transitions)
            if step % self.train_freq == 0:
                for _ in range(training_steps):
                    train_rng, agent_rng, buffer_rng = random.split(train_rng, 3)
                    batch = self.buffer.sample(buffer_rng, self.batch_size)
                    buffer_rng, val_rng = random.split(buffer_rng)
                    val_batch = self.validation_buffer.sample(val_rng, self.batch_size)
                    summary = self.agent.train_step(
                        agent_rng,
                        batch,
                        val_batch
                    )
                    train_steps += 1
                    train_log = summary
                    train_log['train_steps'] = train_steps
                    train_log['env_steps'] = step
                    if self.use_wandb:
                        wandb.log(train_log)

            # Evaluate episode
            if train_steps % self.eval_freq == 0:
                eval_rng, curr_eval = random.split(eval_rng, 2)
                eval_reward = self.eval_policy(rng=curr_eval)
                reward_log = {
                    'env_steps': train_steps,
                    'average_reward': eval_reward
                }
                if self.use_wandb:
                    wandb.log(reward_log)
                if eval_reward > best_performance:
                    best_performance = eval_reward
                    self.save_agent(step)

            step += 1
        self.save_agent(step, agent_name="final_agent")

    def eval_policy(self, rng=None) -> float:
        avg_reward = 0.0
        observations = []
        next_observations = []
        rewards = []
        actions = []
        dones = []
        for e in range(self.eval_episodes):
            obs, _ = self.test_env.reset(seed=e)
            done = False
            steps = 0
            pbar = tqdm(total=1000)
            while not done:
                action = self.agent.act(obs, rng=rng, eval=True)
                next_obs, reward, terminate, truncate, info = self.test_env.step(action)
                observations.append(obs)
                next_observations.append(next_obs)
                actions.append(action)
                dones.append(terminate)
                rewards.append(reward)
                done = terminate or truncate
                avg_reward += reward
                obs = next_obs
                steps += 1
                pbar.update(1)
                # print(steps)
                if done:
                    obs, _ = self.test_env.reset(seed=e)
        avg_reward /= self.eval_episodes
        pbar.close()
        observations = np.asarray(observations)
        actions = np.asarray(actions)
        next_observations = np.asarray(next_observations)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)
        transitions = Transition(
            obs=observations,
            action=actions,
            reward=rewards,
            next_obs=next_observations,
            done=dones,
        )
        self.validation_buffer.add(transitions)
        return avg_reward