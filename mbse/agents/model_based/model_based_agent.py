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


Model_list = list[DynamicsModel]


class ModelBasedAgent(DummyAgent):

    def __init__(
            self,
            action_space: gym.spaces.box,
            observation_space: gym.spaces.box,
            dynamics_model: Union[DynamicsModel, Model_list],
            policy_optimizer: DummyOptimizer,
            discount: float = 0.99,
            n_particles: int = 10,
            reset_model: bool = False,
            calibrate_model: bool = True,
            *args,
            **kwargs,
    ):
        super(ModelBasedAgent, self).__init__(*args, **kwargs)
        self.action_space = action_space
        self.observation_space = observation_space
        if isinstance(dynamics_model, DynamicsModel):
            self.dynamics_model_list = [dynamics_model]
            self.num_dynamics_models = 1
        else:
            self.dynamics_model_list = dynamics_model
            self.num_dynamics_models = len(dynamics_model)
        self.policy_optimizer = policy_optimizer
        self.discount = discount
        self.n_particles = n_particles
        self.reset_model = reset_model
        self.calibrate_model = calibrate_model
        self._init_fn()

    def _init_fn(self):

        self.optimize_for_eval_fns = []
        for i, dynamics_model in enumerate(self.dynamics_model_list):
            def _optimize(
                    params,
                    init_state,
                    key,
                    optimizer_key,
                    alpha,
                    bias_obs,
                    bias_act,
                    bias_out,
                    scale_obs,
                    scale_act,
                    scale_out):
                return self._optimize(
                    eval_fn=dynamics_model.evaluate,
                    optimize_fn=self.policy_optimizer.optimize,
                    n_particles=self.n_particles,
                    horizon=self.policy_optimizer.action_dim[-2],
                    params=params,
                    init_state=init_state,
                    key=key,
                    optimizer_key=optimizer_key,
                    alpha=alpha,
                    bias_obs=bias_obs,
                    bias_act=bias_act,
                    bias_out=bias_out,
                    scale_obs=scale_obs,
                    scale_act=scale_act,
                    scale_out=scale_out,
                )

            optimize = jax.jit(_optimize)
            if i == 0:
                self.optimize = optimize
            self.optimize_for_eval_fns.append(optimize)

        # self.optimize_for_eval = self.optimize

        def step(carry, ins):
            rng = carry[0]
            model_params = carry[2]
            model_opt_state = carry[3]
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
                alpha,
                new_model_params,
                new_model_opt_state,
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
                  alpha,
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
            alpha=alpha,
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

    def act_in_jax(self, obs, rng, eval=False, eval_idx: int = 0):

        if eval:
            def optimize_for_eval(init_state, key, optimizer_key):
                optimize_fn = self.optimize_for_eval_fns[eval_idx]
                action_seq, reward = optimize_fn(
                    params=self.dynamics_model.model_params,
                    init_state=init_state,
                    key=key,
                    optimizer_key=optimizer_key,
                    alpha=self.dynamics_model.alpha,
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
                    alpha=self.dynamics_model.alpha,
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
        max_train_steps_per_iter = 1000
        train_steps = min(max_train_steps_per_iter, self.train_steps)
        train_loops = int(train_steps/max_train_steps_per_iter)
        if self.reset_model:
            model_params = self.dynamics_model.init_model_params
            model_opt_state = self.dynamics_model.init_model_opt_state
        else:
            model_params = self.dynamics_model.model_params
            model_opt_state = self.dynamics_model.model_opt_state
        alpha = jnp.ones(self.observation_space.shape)
        for i in range(train_loops):
            train_rng, rng = jax.random.split(rng, 2)
            transitions = buffer.sample(train_rng, batch_size=int(self.batch_size * train_steps))
            transitions = transitions.reshape(train_steps, self.batch_size)
            val_transitions = None
            if self.validate:
                train_rng, val_rng = jax.random.split(train_rng, 2)
                val_transitions = buffer.sample(val_rng, batch_size=int(self.batch_size * train_steps))
                val_transitions = val_transitions.reshape(train_steps, self.batch_size)
            carry = [
                train_rng,
                alpha,
                model_params,
                model_opt_state,
                ModelSummary(),
                0,
                transitions,
                val_transitions
            ]
            carry, outs = jax.lax.scan(self.step, carry, xs=None, length=train_steps)
            model_params = carry[2]
            model_opt_state = carry[3]
            alpha = carry[1]
        if self.calibrate_model:
            alpha = carry[1]
        self.update_models(model_params=carry[2], model_opt_state=carry[3], alpha=alpha)
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
        for i in range(len(self.dynamics_model_list)):
            self.dynamics_model_list[i].set_transforms(
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
            alpha=self.dynamics_model.alpha,
            bias_obs=self.dynamics_model.bias_obs,
            bias_act=self.dynamics_model.bias_act,
            bias_out=self.dynamics_model.bias_out,
            scale_obs=self.dynamics_model.scale_obs,
            scale_act=self.dynamics_model.scale_act,
            scale_out=self.dynamics_model.scale_out,
        )

    def update_models(self, model_params, model_opt_state, alpha: float = 1.0):
        for i in range(len(self.dynamics_model_list)):
            self.dynamics_model_list[i].update_model(
                model_params=model_params,
                model_opt_state=model_opt_state,
                alpha=alpha,
            )

    @property
    def dynamics_model(self):
        return self.dynamics_model_list[0]