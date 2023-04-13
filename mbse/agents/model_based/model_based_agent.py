import jax
import math

from mbse.models.dynamics_model import DynamicsModel
from mbse.optimizers.cem_trajectory_optimizer import CemTO
from mbse.optimizers.trajax_trajectory_optimizer import TraJaxTO
from mbse.optimizers.sac_based_optimizer import SACOptimizer
import gym
from mbse.utils.replay_buffer import ReplayBuffer, Transition
from mbse.utils.utils import get_idx
from mbse.agents.dummy_agent import DummyAgent
import jax.numpy as jnp
import wandb
from typing import Union, Optional, Dict, Any
from mbse.models.hucrl_model import HUCRLModel

Model_list = list[DynamicsModel]


class ModelBasedAgent(DummyAgent):

    def __init__(
            self,
            action_space: gym.spaces.box,
            observation_space: gym.spaces.box,
            dynamics_model: Union[DynamicsModel, Model_list],
            policy_optimizer_name: str = "CemTO",
            horizon: int = 100,
            n_particles: int = 10,
            reset_model: bool = False,
            calibrate_model: bool = True,
            init_function: bool = True,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            start_optimizer_update: int = 0,
            log_full_training: bool = False,
            log_agent_training: bool = False,
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
        assert policy_optimizer_name in ["CemTO", "TraJaxTO", "SacOpt"], "Optimizer must be CEM or TraJax"

        optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        action_dim = self.action_space.shape
        if isinstance(self.dynamics_model, HUCRLModel):
            action_dim = (self.action_space.shape[0] + self.observation_space.shape[0], )
        if policy_optimizer_name == "CemTO":
            self.policy_optimizer = CemTO(
                dynamics_model_list=self.dynamics_model_list,
                horizon=horizon,
                action_dim=action_dim,
                n_particles=n_particles,
                **optimizer_kwargs,
            )
        elif policy_optimizer_name == "TraJaxTO":
            self.policy_optimizer = TraJaxTO(
                dynamics_model_list=self.dynamics_model_list,
                horizon=horizon,
                action_dim=action_dim,
                n_particles=n_particles,
                **optimizer_kwargs,
            )
        elif policy_optimizer_name == "SacOpt":
            self.policy_optimizer = SACOptimizer(
                dynamics_model_list=self.dynamics_model_list,
                horizon=horizon,
                action_dim=action_dim,
                **optimizer_kwargs,
            )
        self.n_particles = n_particles
        self.reset_model = reset_model
        self.calibrate_model = calibrate_model
        self.start_optimizer_update = start_optimizer_update
        self.update_steps = 0
        self.log_agent_training = log_agent_training
        self.log_full_training = log_full_training
        if init_function:
            self._init_fn()

    def _init_fn(self):
        def step(carry, ins):
            rng = carry[0]
            model_params = carry[2]
            model_opt_state = carry[3]
            tran = ins[0]
            val_tran = ins[1]
            train_rng, rng = jax.random.split(rng, 2)

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
            ]
            outs = [summary]
            return carry, outs

        self.step = jax.jit(step)

    def act_in_jax(self, obs, rng, eval=False, eval_idx: int = 0):
        if isinstance(self.policy_optimizer, SACOptimizer):
            if eval:
                action = self.policy_optimizer.get_action_for_eval(obs=obs, rng=rng, agent_idx=eval_idx)
            else:
                action = self.policy_optimizer.get_action_for_exploration(obs=obs, rng=rng)
            action = action[..., :self.action_space.shape[0]]
        else:
            if eval:
                def optimize_for_eval(init_state, key, optimizer_key):
                    optimize_fn = self.policy_optimizer.optimize_for_eval_fns[eval_idx]
                    action_seq, reward = optimize_fn(
                        dynamics_params=self.dynamics_model.model_params,
                        obs=init_state,
                        key=key,
                        optimizer_key=optimizer_key,
                        model_props=self.dynamics_model.model_props,
                    )
                    return action_seq, reward

                dim_state = obs.shape[-1]
                obs = obs.reshape(-1, dim_state)
                n_envs = obs.shape[0]
                rollout_rng, optimizer_rng = jax.random.split(rng, 2)
                rollout_rng = jax.random.split(rollout_rng, n_envs)
                optimizer_rng = jax.random.split(optimizer_rng, n_envs)
                action_sequence, best_reward = jax.vmap(optimize_for_eval, in_axes=(0, 0, 0))(
                    obs,
                    rollout_rng,
                    optimizer_rng
                )
                action = action_sequence[:, 0, ...]
                if action.shape[0] == 1:
                    action = action.squeeze(0)
            else:
                def optimize(init_state, key, optimizer_key):

                    action_seq, reward = self.policy_optimizer.optimize_for_exploration(
                        dynamics_params=self.dynamics_model.model_params,
                        obs=init_state,
                        key=key,
                        optimizer_key=optimizer_key,
                        model_props=self.dynamics_model.model_props,
                    )
                    return action_seq, reward

                n_envs = obs.shape[0]
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
                   validate: bool = True,
                   log_results: bool = True,
                   ):
        max_train_steps_per_iter = 8000
        if self.num_epochs > 0:
            total_train_steps = math.ceil(buffer.size * self.num_epochs / self.batch_size)
        else:
            total_train_steps = self.train_steps
        self.update_steps += 1
        total_train_steps = min(total_train_steps, self.max_train_steps)
        train_loops = math.ceil(total_train_steps / max_train_steps_per_iter)
        train_steps = min(max_train_steps_per_iter, total_train_steps)
        train_loops = max(train_loops, 1)
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
            if validate:
                train_rng, val_rng = jax.random.split(train_rng, 2)
                val_transitions = buffer.sample(val_rng, batch_size=int(self.batch_size * train_steps))
                val_transitions = val_transitions.reshape(train_steps, self.batch_size)
            carry = [
                train_rng,
                alpha,
                model_params,
                model_opt_state,
            ]
            ins = [transitions, val_transitions]
            carry, outs = jax.lax.scan(self.step, carry, ins, length=train_steps)
            model_params = carry[2]
            model_opt_state = carry[3]
            alpha = carry[1]
            summary = outs[0]
            if log_results:
                if self.log_full_training:
                    for i in range(total_train_steps):
                        wandb.log(get_idx(summary, i).dict())
                else:
                    for i in range(0, total_train_steps, 10):
                        wandb.log(get_idx(summary, i).dict())
        if self.calibrate_model:
            alpha = carry[1]
        self.update_models(model_params=model_params, model_opt_state=model_opt_state, alpha=alpha)
        if (buffer.size > self.policy_optimizer.transitions_per_update and self.update_optimizer) and \
                isinstance(self.policy_optimizer, SACOptimizer):
            train_rng = carry[0]
            policy_train_rng, train_rng = jax.random.split(train_rng, 2)
            policy_agent_train_summary = self.policy_optimizer.train(
                rng=policy_train_rng,
                buffer=buffer,
                dynamics_params=model_params,
                model_props=self.dynamics_model.model_props,
            )
            if log_results and self.log_agent_training:
                for j in range(self.policy_optimizer.train_steps_per_model_update):
                    for i in range(len(self.policy_optimizer.agent_list)):
                        summary = policy_agent_train_summary[j][i]
                        summary_dict = summary.dict()
                        summary_relabeled_dict = {}
                        for key, value in summary_dict.items():
                            summary_relabeled_dict[key + '_agent_' + str(i)] = value
                        wandb.log(
                            summary_relabeled_dict
                        )
        return total_train_steps

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
            model_props=self.dynamics_model.model_props,
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

    @property
    def update_optimizer(self):
        return self.update_steps >= self.start_optimizer_update
