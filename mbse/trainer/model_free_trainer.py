from jax import random
import wandb
from tqdm import tqdm
from mbse.trainer.dummy_trainer import DummyTrainer


class ModelFreeTrainer(DummyTrainer):
    def __init__(self,
                 agent_name: str = "ModelFreeAgent",
                 *args,
                 **kwargs
                 ):
        
        super(ModelFreeTrainer, self).__init__(agent_name=agent_name, *args, **kwargs)

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
        policy = lambda x, y: self.env.action_space.sample()
        transitions = self.rollout_policy(self.exploration_steps, policy, self.rng)
        self.buffer.add(transition=transitions)
        rng_keys = random.split(self.rng, self.max_train_steps+1)
        self.rng = rng_keys[0]
        rng_keys = rng_keys[1:]
        learning_steps = int(self.max_train_steps/(self.rollout_steps*self.num_envs))
        for step in tqdm(range(learning_steps)):
            actor_rng, train_rng = random.split(rng_keys[step], 2)
            policy = self.agent.act
            transitions = self.rollout_policy(self.rollout_steps, policy, actor_rng)
            self.buffer.add(transitions)

            if step % self.train_freq == 0:
                for _ in range(self.train_steps):
                    train_rng, agent_rng, buffer_rng = random.split(train_rng, 3)
                    batch = self.buffer.sample(buffer_rng, self.batch_size)
                    summary = self.agent.train_step(
                        agent_rng,
                        batch,
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
                eval_reward = self.eval_policy(rng=eval_rng)
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










