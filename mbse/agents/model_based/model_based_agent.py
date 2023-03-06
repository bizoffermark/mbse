import jax

from mbse.models.dynamics_model import DynamicsModel, ModelSummary
from mbse.optimizers.dummy_optimizer import DummyOptimizer
import gym
from mbse.utils.utils import sample_trajectories
from mbse.utils.replay_buffer import ReplayBuffer, Transition
from mbse.agents.dummy_agent import DummyAgent
import jax.numpy as jnp
import wandb
from typing import Union


class ModelBasedAgent(DummyAgent):

    def __init__(
            self,
            action_space: gym.spaces.box,
            observation_space: gym.spaces.box,
            dynamics_model: DynamicsModel,
            policy_optimizer: DummyOptimizer,
            discount: float = 0.99,
            n_particles: int = 10,
            reset_model: bool = False,
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
        self.reset_model = reset_model
        self._init_fn()

        # self.optimize = lambda rewards, key: self.policy_optimzer.optimize(
        #        rewards,
        #        key
        #    )

    def _init_fn(self):
        def _optimize(
                params,
                init_state,
                key,
                optimizer_key,
                bias_obs,
                bias_act,
                bias_out,
                scale_obs,
                scale_act,
                scale_out):
            return self._optimize(
                eval_fn=self.dynamics_model.evaluate,
                optimize_fn=self.policy_optimzer.optimize,
                n_particles=self.n_particles,
                horizon=self.policy_optimzer.action_dim[-2],
                params=params,
                init_state=init_state,
                key=key,
                optimizer_key=optimizer_key,
                bias_obs=bias_obs,
                bias_act=bias_act,
                bias_out=bias_out,
                scale_obs=scale_obs,
                scale_act=scale_act,
                scale_out=scale_out,
            )

        self.optimize = jax.jit(_optimize)
        self.optimize_for_eval = self.optimize

        def step(carry, ins):
            rng = carry[0]
            model_params = carry[1]
            model_opt_state = carry[2]
            idx = carry[5]
            transition = carry[6]
            val_transition = carry[7]
            train_rng, rng = jax.random.split(rng, 2)
            tran = transition.get_idx(idx)
            val_tran = val_transition.get_idx(idx)

            (
                new_model_params,
                new_model_opt_state,
                alpha,
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
                alpha,
                summary,
                idx + 1,
                transition,
                val_transition,
            ]
            outs = [summary]
            return carry, outs

        self.step = step

    @staticmethod
    def _optimize(eval_fn,
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

        if eval:
            def optimize_for_eval(init_state, key, optimizer_key):
                action_seq, reward = self.optimize_for_eval(
                    params=self.dynamics_model.model_params,
                    init_state=init_state,
                    key=key,
                    optimizer_key=optimizer_key,
                    bias_obs=self.dynamics_model.bias_obs,
                    bias_act=self.dynamics_model.bias_act,
                    bias_out=self.dynamics_model.bias_out,
                    scale_obs=self.dynamics_model.scale_obs,
                    scale_act=self.dynamics_model.scale_act,
                    scale_out=self.dynamics_model.scale_out,
                )
                return action_seq, reward
            obs = jnp.repeat(jnp.expand_dims(obs, 0), self.n_particles, 0)
            rollout_rng, optimizer_rng = jax.random.split(rng, 2)
            action_sequence, best_reward = optimize_for_eval(obs, rollout_rng, optimizer_rng)
            action = action_sequence[0, ...]
        else:
            def optimize(init_state, key, optimizer_key):

                action_seq, reward = self.optimize(
                    params=self.dynamics_model.model_params,
                    init_state=init_state,
                    key=key,
                    optimizer_key=optimizer_key,
                    bias_obs=self.dynamics_model.bias_obs,
                    bias_act=self.dynamics_model.bias_act,
                    bias_out=self.dynamics_model.bias_out,
                    scale_obs=self.dynamics_model.scale_obs,
                    scale_act=self.dynamics_model.scale_act,
                    scale_out=self.dynamics_model.scale_out,
                )
                return action_seq, reward
            n_envs = obs.shape[0]
            obs = jnp.repeat(jnp.expand_dims(obs, 1), self.n_particles, 1)
            # eval_func = lambda seq: rollout_actions(seq,
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
        action = action[..., :self.action_space.shape[0]]
        return action

    def train_step(self,
                   rng,
                   buffer: ReplayBuffer,
                   ):
        # @partial(jax.jit, static_argnums=(0, 2, 3))
        transitions = buffer.sample(rng, batch_size=int(self.batch_size * self.train_steps))
        transitions = transitions.reshape(self.train_steps, self.batch_size)
        val_transitions = None
        if self.validate:
            rng, val_rng = jax.random.split(rng, 2)
            val_transitions = buffer.sample(val_rng, batch_size=int(self.batch_size * self.train_steps))
            val_transitions = val_transitions.reshape(self.train_steps, self.batch_size)
            alpha = 0.0
        if self.reset_model:
            carry = [
                rng,
                self.dynamics_model.init_model_params,
                self.dynamics_model.init_model_opt_state,
                alpha,
                ModelSummary(),
                0,
                transitions,
                val_transitions,
            ]
        else:
            carry = [
                rng,
                self.dynamics_model.model_params,
                self.dynamics_model.model_opt_state,
                alpha,
                ModelSummary(),
                0,
                transitions,
                val_transitions,
            ]
        carry, outs = jax.lax.scan(self.step, carry, xs=None, length=self.train_steps)
        self.dynamics_model.update_model(model_params=carry[1], model_opt_state=carry[2], alpha=carry[3])
        summary = outs[0].dict()
        if self.use_wandb:
            for log_dict in summary:
                wandb.log(log_dict)

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

    def predict_next_state(self,
                           tran: Transition,
                           ):
        return self.dynamics_model.predict_raw(
            parameters=self.dynamics_model.model_params,
            tran=tran,
            bias_obs=self.dynamics_model.bias_obs,
            bias_act=self.dynamics_model.bias_act,
            bias_out=self.dynamics_model.bias_out,
            scale_obs=self.dynamics_model.scale_obs,
            scale_act=self.dynamics_model.scale_act,
            scale_out=self.dynamics_model.scale_out,
        )
