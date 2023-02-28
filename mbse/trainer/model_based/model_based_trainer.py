import jax.random
from mbse.utils.replay_buffer import ReplayBuffer, Transition
from mbse.trainer.dummy_trainer import DummyTrainer
from mbse.agents.model_based.model_based_agent import ModelBasedAgent
import wandb
from jax import random
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


class ModelBasedTrainer(DummyTrainer):
    def __init__(self,
                 agent_name: str = "ModelBasedAgent",
                 validation_buffer_size: int = 0,
                 validation_batch_size: int = 1024,
                 uniform_exploration: bool = False,
                 *args,
                 **kwargs
                 ):

        video_prefix = "Uniform" if uniform_exploration else "Learning"
        super(ModelBasedTrainer, self).__init__(agent_name=agent_name, video_prefix=video_prefix, *args, **kwargs)
        assert isinstance(self.agent, ModelBasedAgent), "Only Model based agents are allowed"
        self.validation_buffer = None
        self.validation_batch_size = validation_batch_size
        self.collect_validation_data(validation_buffer_size)
        self.uniform_exploration = uniform_exploration

    def collect_validation_data(self, validation_buffer_size: int = 0):
        if validation_buffer_size > 0:
            self.validation_buffer = ReplayBuffer(
                obs_shape=self.buffer.obs_shape,
                action_shape=self.buffer.action_shape,
                normalize=False,
                action_normalize=False,
                learn_deltas=False
            )
            num_points = int(validation_buffer_size * self.num_envs)
            obs_shape = (num_points,) + self.env.observation_space.shape
            action_space = (num_points,) + self.env.action_space.shape
            obs_vec = np.zeros(obs_shape)
            action_vec = np.zeros(action_space)
            reward_vec = np.zeros((num_points,))
            next_obs_vec = np.zeros(obs_shape)
            done_vec = np.zeros((num_points,))
            for step in range(validation_buffer_size):
                obs = self.env.observation_space.sample()
                action = self.env.action_space.sample()
                obs_vec[step * self.num_envs: (step + 1) * self.num_envs] = obs
                action_vec[step * self.num_envs: (step + 1) * self.num_envs] = action

            transitions = Transition(
                obs=obs_vec,
                action=action_vec,
                reward=reward_vec,
                next_obs=next_obs_vec,
                done=done_vec,
            )

            #policy = lambda x, y: np.concatenate(
            #    [self.env.action_space.sample().reshape(1, -1)
            #     for s in range(self.num_envs)], axis=0)
            #self.rng, val_rng = random.split(self.rng, 2)
            #transitions = self.rollout_policy(validation_buffer_size,
            #                                  policy,
            #                                  val_rng
            #                                  )
            self.validation_buffer.add(transitions)

    def validate_model(self, rng):
        model_log = {}
        if self.validation_buffer is not None:
            val_tran = self.validation_buffer.sample(
                rng=rng,
                batch_size=self.validation_batch_size,
            )
            mean_pred, std_pred = self.agent.predict_next_state(val_tran)
            eps_uncertainty = jnp.sum(jnp.std(mean_pred, axis=0), axis=-1)
            mean_eps_uncertainty = jnp.mean(eps_uncertainty)
            max_eps_uncertainty = jnp.max(eps_uncertainty)
            std_eps_uncertainty = jnp.std(eps_uncertainty)
            std_pred = jnp.mean(jnp.sum(jnp.sqrt(jnp.mean(jnp.square(std_pred), axis=0)), axis=-1))
            # y_true = val_tran.next_obs
            # mse = jnp.mean(jnp.sum(jnp.square(y_true - mean_pred), axis=-1))
            model_log = {
                # 'validation_mse': mse.astype(float).item(),
                'validation_al_std': std_pred.astype(float).item(),
                'validation_eps_std_mean': mean_eps_uncertainty.astype(float).item(),
                'validation_eps_std_max': max_eps_uncertainty.astype(float).item(),
                'validation_eps_std_std': std_eps_uncertainty.astype(float).item(),
            }
        return model_log

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
        curr_eval, eval_val_rng = jax.random.split(curr_eval, 2)
        model_log = self.validate_model(eval_val_rng)
        average_reward = self.eval_policy(rng=curr_eval)
        best_performance = average_reward
        initial_log = {
            'env_steps': 0,
            'train_steps': 0,
            'average_reward': average_reward
        }
        train_steps = 0
        self.save_agent(0)
        if self.use_wandb:
            wandb.define_metric("env_steps")
            wandb.define_metric("train_steps")
            initial_log.update(model_log)
            wandb.log(initial_log)

        exploration_policy = lambda x, y: np.concatenate([self.env.action_space.sample().reshape(1, -1)
                                                          for s in range(self.num_envs)], axis=0)
        policy = exploration_policy
        self.rng, explore_rng = random.split(self.rng, 2)
        if self.exploration_steps > 0:
            transitions = self.rollout_policy(self.exploration_steps, policy, explore_rng)
            self.buffer.add(transitions)
        rng_keys = random.split(self.rng, self.max_train_steps + 1)
        self.rng = rng_keys[0]
        rng_keys = rng_keys[1:]
        learning_steps = int(self.max_train_steps / (self.rollout_steps * self.num_envs))
        rng_key, reset_rng = random.split(rng_keys[0], 2)
        rng_keys = rng_keys.at[0].set(rng_key)
        reset_seed = random.randint(
            reset_rng,
            (1,),
            minval=0,
            maxval=int(learning_steps * self.rollout_steps)).item()
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
            policy = self.agent.act_in_train if not self.uniform_exploration else exploration_policy
            actor_rng, val_rng = random.split(actor_rng, 2)
            transitions, obs, done = self.step_env(obs, policy, self.rollout_steps, actor_rng)
            self.buffer.add(transitions)
            reward_log = {}
            train_step_log = {}
            model_log = {}
            env_step_log = {
                'env_steps': step * self.rollout_steps * self.num_envs
            }
            if step % self.train_freq == 0 and self.buffer.size >= self.batch_size:
                train_rng, agent_rng = random.split(train_rng, 2)
                self.agent.train_step(
                    rng=agent_rng,
                    buffer=self.buffer,
                )
                train_steps += self.train_steps
                train_step_log = {
                    'train_steps': train_steps
                }
                eval_rng, eval_val_rng = jax.random.split(eval_rng, 2)
                model_log = self.validate_model(eval_val_rng)
            # Evaluate episode
            if train_steps % self.eval_freq == 0 and train_steps > 0:
                eval_rng, curr_eval = random.split(eval_rng, 2)
                eval_reward = self.eval_policy(rng=curr_eval, step=train_steps)
                reward_log = {
                    'average_reward': eval_reward
                }
                if eval_reward > best_performance:
                    best_performance = eval_reward
                    self.save_agent(step)
            if self.use_wandb:
                train_log = env_step_log
                train_log.update(train_step_log)
                train_log.update(reward_log)
                train_log.update(model_log)
                scaler_dict = {
                    'bias_obs': np.mean(self.buffer.state_normalizer.mean).astype(float).item(),
                    'bias_act': np.mean(self.buffer.action_normalizer.mean).astype(float).item(),
                    'bias_out': np.mean(self.buffer.next_state_normalizer.mean).astype(float).item(),
                    'scale_obs': np.mean(self.buffer.state_normalizer.std).astype(float).item(),
                    'scale_act': np.mean(self.buffer.action_normalizer.std).astype(float).item(),
                    'scale_out': np.mean(self.buffer.next_state_normalizer.std).astype(float).item(),
                }
                train_log.update(scaler_dict)
                wandb.log(train_log)
            # step += 1
        self.save_agent(step, agent_name="final_agent")
