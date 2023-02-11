import jax

from mbse.models.dynamics_model import DynamicsModel, ModelSummary
from mbse.optimizers.dummy_optimizer import DummyOptimizer
import gym
from mbse.utils.utils import rollout_actions, sample_trajectories
from mbse.utils.replay_buffer import ReplayBuffer
from mbse.agents.dummy_agent import DummyAgent
import numpy as np
import jax.numpy as jnp
from functools import partial
import wandb
from typing import Union, Callable


class ModelBasedAgent(DummyAgent):

    def __init__(
            self,
            action_space: gym.spaces.box,
            observation_space: gym.spaces.box,
            dynamics_model: DynamicsModel,
            policy_optimizer: DummyOptimizer,
            discount: float = 0.99,
            n_particles: int = 10,
            *args,
            **kwargs,
    ):
        super(ModelBasedAgent, self).__init__(*args, **kwargs)
        self.action_space = action_space
        self.observation_space = observation_space
        self.dynamics_model = dynamics_model
        self.policy_optimzer = policy_optimizer
        self.discount = discount
        self.n_particles = n_particles
        # self.optimize = lambda rewards, key: self.policy_optimzer.optimize(
        #        rewards,
        #        key
        #    )

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 2, 3))
    def optimize(eval_fn,
                 optimize_fn,
                 n_particles,
                 horizon,
                 params,
                 init_state,
                 key,
                 optimizer_key,
                 bias_obs,
                 bias_act,
                 bias_out,
                 scale_obs,
                 scale_act,
                 scale_out,
                 ):
        eval_func = lambda seq, x, k: sample_trajectories(
            evaluate_fn=eval_fn,
            parameters=params,
            init_state=x,
            horizon=horizon,
            key=k,
            actions=seq,
            bias_obs=bias_obs,
            bias_act=bias_act,
            bias_out=bias_out,
            scale_obs=scale_obs,
            scale_act=scale_act,
            scale_out=scale_out,
        )

        def sum_rewards(seq):
            seq = jnp.repeat(jnp.expand_dims(seq, 0), n_particles, 0)
            transition = eval_func(seq, init_state, key)
            return transition.reward.mean()

        action_seq, reward = optimize_fn(
            sum_rewards,
            optimizer_key
        )
        return action_seq, reward

    def act_in_jax(self, obs, rng, eval=False):
        def optimize(init_state, key, optimizer_key):

            action_seq, reward = self.optimize(
                eval_fn=self.dynamics_model.evaluate,
                optimize_fn=self.policy_optimzer.optimize,
                params=self.dynamics_model.model_params,
                horizon=self.policy_optimzer.action_dim[-2],
                init_state=init_state,
                key=key,
                optimizer_key=optimizer_key,
                n_particles=self.n_particles,
                bias_obs=self.dynamics_model.bias_obs,
                bias_act=self.dynamics_model.bias_act,
                bias_out=self.dynamics_model.bias_out,
                scale_obs=self.dynamics_model.scale_obs,
                scale_act=self.dynamics_model.scale_act,
                scale_out=self.dynamics_model.scale_out,
            )
            return action_seq, reward

        if eval:
            obs = jnp.repeat(jnp.expand_dims(obs, 0), self.n_particles, 0)
            rollout_rng, optimizer_rng = jax.random.split(rng, 2)
            action_sequence, best_reward = optimize(obs, rollout_rng, optimizer_rng)
            action = action_sequence[0, ...]
        else:
            n_envs = obs.shape[0]
            obs = jnp.repeat(jnp.expand_dims(obs, 1), self.n_particles, 1)
            #eval_func = lambda seq: rollout_actions(seq,
            #                                        initial_state=obs,
            #                                        dynamics_model=self.dynamics_model,
            #                                        reward_model=self.reward_model,
            #                                        rng=rng)
            rollout_rng, optimizer_rng = jax.random.split(rng, 2)
            rollout_rng = jax.random.split(rollout_rng, n_envs)
            optimizer_rng = jax.random.split(optimizer_rng, n_envs)

            action_sequence, best_reward = jax.vmap(optimize)(
                obs,
                rollout_rng,
                optimizer_rng
            )
            action = action_sequence[:, 0, ...]
        return action

    def train_step(self,
                   rng,
                   buffer: ReplayBuffer,
                   ):
        # @partial(jax.jit, static_argnums=(0, 2, 3))
        def sample_data(data_buffer, rng, batch_size, validate=False):
            val_tran = None
            if validate:
                rng, val_rng = jax.random.split(rng, 2)
                val_tran = data_buffer.sample(val_rng, batch_size=batch_size)
            tran = data_buffer.sample(rng, batch_size=batch_size)
            return tran, val_tran

        def step(carry, ins):
            rng = carry[0]
            model_params = carry[1]
            model_opt_state = carry[2]
            buffer_rng, train_rng, rng = jax.random.split(rng, 3)
            tran, val_tran = sample_data(
                buffer,
                buffer_rng,
                self.batch_size,
                self.validate
            )

            (
                new_model_params,
                new_model_opt_state,
                summary,
            ) = \
                self.dynamics_model._train_step(
                    tran=tran,
                    model_params=model_params,
                    model_opt_state=model_opt_state,
                    val=val_tran,
                )
            carry = [
                rng,
                new_model_params,
                new_model_opt_state,
                summary,
            ]
            outs = carry[1:]
            return carry, outs

        carry = [
            rng,
            self.dynamics_model.model_params,
            self.dynamics_model.model_opt_state,
            ModelSummary(),
        ]
        carry, outs = jax.lax.scan(step, carry, xs=None, length=self.train_steps)
        self.dynamics_model.update_model(model_params=carry[1], model_opt_state=carry[2])
        summary = carry[3]
        if self.use_wandb:
            wandb.log(summary.dict())

    def set_transforms(self,
                       bias_obs: Union[jnp.ndarray, float] = 0.0,
                       bias_act: Union[jnp.ndarray, float] = 0.0,
                       bias_out: Union[jnp.ndarray, float] = 0.0,
                       scale_obs: Union[jnp.ndarray, float] = 1.0,
                       scale_act: Union[jnp.ndarray, float] = 1.0,
                       scale_out: Union[jnp.ndarray, float] = 1.0,
                       ):
        self.dynamics_model.set_transforms(
            bias_obs=bias_obs,
            bias_act=bias_act,
            bias_out=bias_out,
            scale_obs=scale_obs,
            scale_act=scale_act,
            scale_out=scale_out,
        )

