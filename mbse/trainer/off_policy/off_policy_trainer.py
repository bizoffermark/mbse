from jax import random
import wandb
from tqdm import tqdm
from mbse.trainer.dummy_trainer import DummyTrainer
from mbse.agents.actor_critic.sac import SACAgent


class OffPolicyTrainer(DummyTrainer):
    def __init__(self,
                 agent_name: str = "OffPolicyAgent",
                 *args,
                 **kwargs
                 ):

        super(OffPolicyTrainer, self).__init__(agent_name=agent_name, *args, **kwargs)
        assert isinstance(self.agent, SACAgent), "Only Off policy agents are allowed"

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
        rng_keys = random.split(self.rng, self.max_train_steps + 1)
        self.rng = rng_keys[0]
        rng_keys = rng_keys[1:]
        learning_steps = int(self.max_train_steps/self.rollout_steps)
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
            transitions, obs, done = self.step_env(obs, policy, self.rollout_steps, actor_rng)
            self.buffer.add(transitions)
            if done:
                reset_rng, next_reset_rng = random.split(reset_rng, 2)
                reset_seed = random.randint(
                    next_reset_rng,
                    (1,),
                    minval=0,
                    maxval=int(learning_steps * self.rollout_steps)).item()
                obs, _ = self.env.reset(seed=reset_seed)
            # transitions = self.rollout_policy(self.rollout_steps, policy, actor_rng)
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
                eval_reward = self.eval_policy()
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










