import jax.random

from mbse.trainer.dummy_trainer import DummyTrainer
from mbse.agents.model_based.model_based_agent import ModelBasedAgent
import wandb
from jax import random
import numpy as np
from tqdm import tqdm


class ModelBasedTrainer(DummyTrainer):
    def __init__(self,
                 agent_name: str = "ModelBasedAgent",
                 *args,
                 **kwargs
                 ):

        super(ModelBasedTrainer, self).__init__(agent_name=agent_name, *args, **kwargs)
        assert isinstance(self.agent, ModelBasedAgent), "Only Model based agents are allowed"

    def train(self):
        if self.use_wandb:
            wandb.define_metric('env_steps')
        self.rng, eval_rng = random.split(self.rng, 2)
        eval_rng, curr_eval = random.split(eval_rng, 2)

        self.agent.set_transforms(
            bias_obs=self.buffer.state_normalizer.mean,
            bias_act=self.buffer.action_normalizer.mean,
            bias_out=self.buffer.next_state_normalizer.mean,
            scale_obs=self.buffer.state_normalizer.std,
            scale_act=self.buffer.action_normalizer.std,
            scale_out=self.buffer.next_state_normalizer.std,

        )
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
        transitions = self.rollout_policy(self.exploration_steps, policy, self.rng)
        self.buffer.add(transitions)
        rng_keys = random.split(self.rng, self.max_train_steps + 1)
        self.rng = rng_keys[0]
        rng_keys = rng_keys[1:]
        learning_steps = int(self.max_train_steps/(self.rollout_steps*self.num_envs))
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
            self.agent.set_transforms(
                bias_obs=self.buffer.state_normalizer.mean,
                bias_act=self.buffer.action_normalizer.mean,
                bias_out=self.buffer.next_state_normalizer.mean,
                scale_obs=self.buffer.state_normalizer.std,
                scale_act=self.buffer.action_normalizer.std,
                scale_out=self.buffer.next_state_normalizer.std,

            )
            policy = self.agent.act
            actor_rng, val_rng = random.split(actor_rng, 2)
            transitions, obs, done = self.step_env(obs, policy, self.rollout_steps, actor_rng)
            self.buffer.add(transitions)
            if step % self.train_freq == 0:
                train_rng, agent_rng = random.split(train_rng, 2)
                self.agent.train_step(
                    rng=agent_rng,
                    buffer=self.buffer,
                )
                train_steps += self.train_steps

            # Evaluate episode
            if train_steps % self.eval_freq == 0:
                eval_rng, curr_eval = random.split(eval_rng, 2)
                eval_reward = self.eval_policy(rng=curr_eval, step=train_steps)
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