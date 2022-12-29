from mbse.agents.dummy_agent import DummyAgent
from mbse.agents.model_based.model_based_agent import ModelBasedAgent
from mbse.utils.replay_buffer import merge_transitions, ReplayBuffer
from mbse.trainer.dummy_trainer import DummyTrainer
import wandb
from jax import random
from tqdm import tqdm
from mbse.utils.utils import rollout_policy


class ModelBasedTrainer(DummyTrainer):

    def __init__(
            self,
            model_based_agent: ModelBasedAgent,
            model_free_agent: DummyAgent,
            agent_name: str = "MBTrainer",
            model_rollout_steps: int = 4,
            model_num_rollouts: int = int(1e4),
            model_update_steps: int = 100,
            policy_update_steps: int = 100,
            model_batch_size: int = 256,
            buffer_ratio: int = 0.5,
            clear_sim_buffer: bool = True,
            *args,
            **kwargs,
            ):
        
        super(ModelBasedTrainer, self).__init__(agent_name=agent_name, agent=model_free_agent, *args, **kwargs)
        self.model_based_agent = model_based_agent
        self.model_rollout_steps = model_rollout_steps
        self.model_num_rollouts = model_num_rollouts
        self.model_update_steps = model_update_steps
        self.policy_update_steps = policy_update_steps
        self.model_batch_size = model_batch_size
        self.buffer_ratio = buffer_ratio
        self.simulated_buffer = ReplayBuffer(
            obs_shape=self.env.observation_space.shape,
            action_shape=self.env.action_space.shape,
            max_size=self.buffer_size,
        )
        self.real_batch_size = int(self.batch_size*self.buffer_ratio)
        self.simulated_batch_size = self.batch_size - self.real_batch_size
        self.clear_sim_buffer = clear_sim_buffer

    def train(self):
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
        rng_keys = random.split(self.rng, self.max_train_steps+1)
        self.rng = rng_keys[0]
        rng_keys = rng_keys[1:]
        for step in tqdm(range(self.max_train_steps)):
            actor_rng, train_rng = random.split(rng_keys[step], 2)
            policy = self.agent.act
            transitions = self.rollout_policy(self.rollout_steps, policy, actor_rng)
            self.buffer.add(transitions)

            if step % self.train_freq == 0:
                if self.clear_sim_buffer:
                    self.simulated_buffer.reset()
                for _ in range(self.train_steps):
                    train_rng, agent_rng, buffer_rng = random.split(train_rng, 3)
                    model_buffer_rng, agent_buffer_rng, rollout_buffer_rng = random.split(buffer_rng, 3)
                    agent_rng, model_based_agent_rng = random.split(agent_rng, 2)
                    for _ in range(self.model_update_steps):
                        model_based_agent_rng, train_mb_agent_rng = random.split(model_based_agent_rng, 2)
                        model_buffer_rng, buffer_sample_rng = random.split(model_buffer_rng, 2)
                        batch = self.buffer.sample(buffer_sample_rng, self.model_batch_size)
                        model_summary = self.model_based_agent.train_step(train_mb_agent_rng, tran=batch)
                        if self.use_wandb:
                            wandb.log(model_summary)
                    batch = self.buffer.sample(rollout_buffer_rng, self.model_num_rollouts)
                    init_state = batch.obs
                    agent_rng, rollout_rng = random.split(agent_rng, 2)
                    model_transitions = rollout_policy(
                        policy=self.agent.act_in_jax,
                        initial_state=init_state,
                        dynamics_model=self.model_based_agent.dynamics_model,
                        reward_model=self.model_based_agent.reward_model,
                        rng=rollout_rng,
                        num_steps=self.model_rollout_steps)
                    self.simulated_buffer.add(model_transitions)
                    for _ in range(self.policy_update_steps):
                        agent_buffer_rng, sim_buffer_sample_rng, real_buffer_sample_rng = random.split(
                            agent_buffer_rng,
                            3)
                        policy_batch_sim = self.simulated_buffer.sample(
                            sim_buffer_sample_rng,
                            self.simulated_batch_size)
                        policy_batch_real = self.buffer.sample(real_buffer_sample_rng, self.real_batch_size)
                        policy_batch = merge_transitions(policy_batch_sim, policy_batch_real)
                        agent_rng, train_agent_rng = random.split(agent_rng, 2)
                        policy_summary = self.agent.train_step(
                            train_agent_rng,
                            policy_batch,
                        )
                        if self.use_wandb:
                            wandb.log(policy_summary)

                    train_steps += 1
                    train_log = {
                        'train_steps': train_steps,
                        'env_steps': step,
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






