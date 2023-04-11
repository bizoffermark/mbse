import copy
import time

import jax.random

from mbse.agents.actor_critic.sac import SACAgent, SACTrainingState
from gym.spaces import Box
from mbse.utils.replay_buffer import Transition, ReplayBuffer, JaxReplayBuffer
from typing import Callable, Union, Optional
import jax.numpy as jnp
from mbse.utils.utils import sample_trajectories, get_idx, tree_stack
import functools
from mbse.optimizers.dummy_policy_optimizer import DummyPolicyOptimizer
from mbse.models.active_learning_model import ActiveLearningHUCRLModel, ActiveLearningPETSModel
import flax.struct

EPS = 1e-6

@functools.partial(
    jax.jit, static_argnums=(0, 2, 4)
)
def _simulate_dynamics(horizon: int,
                       obs: jax.Array,
                       policy: Callable,
                       actor_params,
                       evaluate_fn: Callable,
                       dynamics_params=None,
                       key=None,
                       alpha: Union[jnp.ndarray, float] = 1.0,
                       bias_obs: Union[jnp.ndarray, float] = 0.0,
                       bias_act: Union[jnp.ndarray, float] = 0.0,
                       bias_out: Union[jnp.ndarray, float] = 0.0,
                       scale_obs: Union[jnp.ndarray, float] = 1.0,
                       scale_act: Union[jnp.ndarray, float] = 1.0,
                       scale_out: Union[jnp.ndarray, float] = 1.0,
                       sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
                       policy_bias_obs: Union[jnp.ndarray, float] = 0.0,
                       policy_scale_obs: Union[jnp.ndarray, float] = 1.0,
                       ):
    def sample_trajectories_for_state(state: jax.Array, sample_key):
        return sample_trajectories(
            evaluate_fn=evaluate_fn,
            parameters=dynamics_params,
            init_state=state,
            horizon=horizon,
            key=sample_key,
            policy=policy,
            actor_params=actor_params,
            alpha=alpha,
            bias_obs=bias_obs,
            bias_act=bias_act,
            bias_out=bias_out,
            scale_obs=scale_obs,
            scale_act=scale_act,
            scale_out=scale_out,
            sampling_idx=sampling_idx,
            policy_bias_obs=policy_bias_obs,
            policy_scale_obs=policy_scale_obs,
        )

    key = jax.random.split(key, obs.shape[0])
    transitions = jax.vmap(sample_trajectories_for_state)(obs, key)

    def flatten(arr):
        new_arr = arr.reshape(-1, arr.shape[-1])
        return new_arr

    transitions = Transition(
        obs=flatten(transitions.obs),
        action=flatten(transitions.action),
        reward=flatten(transitions.reward),
        next_obs=flatten(transitions.next_obs),
        done=flatten(transitions.done),
    )
    return transitions


@flax.struct.dataclass
class SacOptimizerState:
    agent_train_state: SACTrainingState
    obs_bias: jnp.array
    obs_scale: jnp.array


class SACOptimizer(DummyPolicyOptimizer):
    def __init__(self,
                 action_dim: tuple,
                 dynamics_model_list: list,
                 horizon: int = 20,
                 n_particles: int = 10,
                 transitions_per_update: int = 10,
                 simulated_buffer_size: int = 1000000,
                 train_steps_per_model_update: int = 20,
                 sim_transitions_ratio: float = 0.5,
                 normalize: bool = True,
                 action_normalize: bool = False,
                 sac_kwargs: Optional[dict] = None,
                 reset_actor_params: bool = False,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        assert isinstance(dynamics_model_list, list)
        self.dynamics_model_list = dynamics_model_list
        obs_dim = self.dynamics_model.obs_dim
        self.active_exploration_agent = False
        if isinstance(self.dynamics_model, ActiveLearningPETSModel) or isinstance(
                self.dynamics_model, ActiveLearningHUCRLModel):
            self.dynamics_model_list.append(dynamics_model_list[0])
            self.active_exploration_agent = True
        dummy_obs_space = Box(low=-1, high=1, shape=(obs_dim,))
        dummy_act_space = Box(low=-1, high=1, shape=action_dim)
        if sac_kwargs is not None:
            self.agent_list = [SACAgent(
                action_space=dummy_act_space,
                observation_space=dummy_obs_space,
                **sac_kwargs,
            ) for model in self.dynamics_model_list]
        else:
            self.agent_list = [SACAgent(
                action_space=dummy_act_space,
                observation_space=dummy_obs_space,
            ) for model in self.dynamics_model_list]

        init_optimizer_state = [SacOptimizerState(
            agent_train_state=agent.training_state,
            obs_bias=jnp.zeros((obs_dim,)),
            obs_scale=jnp.ones((obs_dim,)),
        ) for agent in self.agent_list]
        self.init_optimizer_state = tree_stack(init_optimizer_state)
        self.optimizer_state = copy.deepcopy(self.init_optimizer_state)
        self.horizon = horizon
        self.n_particles = n_particles
        self.transitions_per_update = transitions_per_update
        self.simulated_buffer_size = simulated_buffer_size
        self.normalize = normalize
        self.action_normalize = action_normalize
        self.obs_dim = (obs_dim,)
        self.action_dim = action_dim
        self.train_steps_per_model_update = train_steps_per_model_update
        self.sim_transitions_ratio = sim_transitions_ratio
        self.reset_actor_params = reset_actor_params
        self._init_fn()

    def get_action_for_eval(self, obs: jax.Array, rng, agent_idx: int):
        policy = self.agent_list[0].get_eval_action
        agent_state = get_idx(self.optimizer_state, agent_idx)
        normalized_obs = (obs - agent_state.obs_bias) / (agent_state.obs_scale + EPS)
        action = policy(
            actor_params=agent_state.agent_train_state.actor_params,
            obs=normalized_obs,
            rng=rng,
        )
        return action

    def get_action(self, obs: jax.Array, rng):
        return self.get_action_for_eval(obs=obs, rng=rng, agent_idx=0)

    def get_action_for_exploration(self, obs: jax.Array, rng, *args, **kwargs):
        if self.active_exploration_agent:
            policy = self.agent_list[0].get_action
            agent_state = get_idx(self.optimizer_state, -1)
            normalized_obs = (obs - agent_state.obs_bias) / (agent_state.obs_scale + EPS)
            action = policy(
                actor_params=agent_state.agent_train_state.actor_params,
                obs=normalized_obs,
                rng=rng,
            )
        else:
            policy = self.agent_list[0].get_action
            agent_state = get_idx(self.optimizer_state, 0)
            normalized_obs = (obs - agent_state.obs_bias) / (agent_state.obs_scale + EPS)
            action = policy(
                actor_params=agent_state.agent_train_state.actor_params,
                obs=normalized_obs,
                rng=rng,
            )
        return action

    def _init_fn(self):

        def train_agent_step(
                train_rng,
                train_state,
                sim_transitions,
        ):
            return self.train_agent_step(
                train_rng=train_rng,
                train_state=train_state,
                sim_transitions=sim_transitions,
                agent_train_fn=self.agent_list[0].step,
                agent_train_steps=self.agent_list[0].train_steps,
            )

        self.train_step = jax.jit(jax.vmap(train_agent_step))

    @staticmethod
    def train_agent_step(train_rng,
                         train_state,
                         sim_transitions,
                         agent_train_fn,
                         agent_train_steps,
                         ):
        carry = [
            train_rng,
            train_state
        ]
        ins = [sim_transitions]
        carry, outs = jax.lax.scan(agent_train_fn, carry, ins, length=agent_train_steps)
        next_train_state = carry[1]
        summary = outs[-1]
        return next_train_state, summary

    def train(self,
              rng,
              buffer: ReplayBuffer,
              dynamics_params: Optional = None,
              alpha: Union[jnp.ndarray, float] = 1.0,
              bias_obs: Union[jnp.ndarray, float] = 0.0,
              bias_act: Union[jnp.ndarray, float] = 0.0,
              bias_out: Union[jnp.ndarray, float] = 0.0,
              scale_obs: Union[jnp.ndarray, float] = 1.0,
              scale_act: Union[jnp.ndarray, float] = 1.0,
              scale_out: Union[jnp.ndarray, float] = 1.0,
              sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
              ):
        sim_buffer_kwargs = {
            'obs_shape': self.obs_dim,
            'action_shape': self.action_dim,
            'max_size': self.simulated_buffer_size,
            'normalize': self.normalize,
            'action_normalize': self.action_normalize,
        }

        simulation_buffers = [JaxReplayBuffer(
            learn_deltas=False,
            **sim_buffer_kwargs,
        ) for _ in self.agent_list]
        true_obs = jnp.asarray(buffer.obs[:buffer.size])
        train_steps = self.agent_list[0].train_steps
        batch_size = self.agent_list[0].batch_size
        full_optimizer_state = self.init_optimizer_state
        if not self.reset_actor_params:
            full_optimizer_state = self.optimizer_state
        agent_summary = []
        policy = self.agent_list[0].get_action
        for i in range(self.train_steps_per_model_update):
            actor_obs_bias = []
            actor_obs_scale = []
            transitions_list = []
            for j in range(len(self.agent_list)):
                sim_buffer = simulation_buffers[j]
                agent = self.agent_list[j]
                evaluate_fn = self.dynamics_model_list[j].evaluate
                optimizer_state = get_idx(full_optimizer_state, idx=j)
                if self.is_active_exploration_agent(idx=j):
                    evaluate_fn = self.dynamics_model_list[j].evaluate_for_exploration

                batch_sim_buffer = int(self.sim_transitions_ratio * self.transitions_per_update) * \
                                   (sim_buffer.size > 0)
                batch_true_buffer = int(self.transitions_per_update - batch_sim_buffer)
                buffer_rng, rng = jax.random.split(rng, 2)
                if batch_sim_buffer > 0:
                    true_buffer_rng, sim_buffer_rng = jax.random.split(buffer_rng, 2)
                    ind = jax.random.randint(true_buffer_rng, (batch_true_buffer,), 0, true_obs.shape[0])
                    true_obs_sample = true_obs[ind]
                    sim_trans = sim_buffer.sample(sim_buffer_rng, batch_size=batch_sim_buffer)
                    sim_obs_sample = sim_trans.obs * sim_buffer.state_normalizer.std + \
                                     sim_buffer.state_normalizer.mean
                    obs = jnp.concatenate([true_obs_sample, sim_obs_sample], axis=0)
                else:
                    ind = jax.random.randint(buffer_rng, (batch_true_buffer,), 0, true_obs.shape[0])
                    obs = true_obs[ind]
                simulation_key, rng = jax.random.split(rng, 2)
                simulated_transitions = _simulate_dynamics(
                    obs=obs,
                    policy=policy,
                    actor_params=optimizer_state.agent_train_state.actor_params,
                    evaluate_fn=evaluate_fn,
                    dynamics_params=dynamics_params,
                    key=simulation_key,
                    alpha=alpha,
                    bias_obs=bias_obs,
                    bias_act=bias_act,
                    bias_out=bias_out,
                    scale_obs=scale_obs,
                    scale_act=scale_act,
                    scale_out=scale_out,
                    sampling_idx=sampling_idx,
                    horizon=self.horizon,
                    policy_bias_obs=optimizer_state.obs_bias,
                    policy_scale_obs=optimizer_state.obs_scale,
                )
                simulation_buffers[j].add(transition=simulated_transitions)
                actor_obs_bias.append(simulation_buffers[j].state_normalizer.mean)
                actor_obs_scale.append(simulation_buffers[j].state_normalizer.std)
                sim_buffer_rng, rng = jax.random.split(rng, 2)
                sim_transitions = simulation_buffers[j].sample(sim_buffer_rng,
                                                               batch_size=int(train_steps * batch_size)
                                                               )
                sim_transitions = sim_transitions.reshape(train_steps, batch_size)
                transitions_list.append(sim_transitions)
            sim_transitions = tree_stack(transitions_list)
            train_rng = jax.random.split(rng, len(self.agent_list) + 1)
            rng = train_rng[0]
            train_rng = train_rng[1:]
            agent_train_state, summary = self.train_step(
                train_rng=train_rng,
                train_state=full_optimizer_state.agent_train_state,
                sim_transitions=sim_transitions,
            )
            obs_bias = tree_stack(actor_obs_bias)
            obs_scale = tree_stack(actor_obs_scale)
            full_optimizer_state = SacOptimizerState(
                agent_train_state=agent_train_state,
                obs_bias=obs_bias,
                obs_scale=obs_scale,
            )
            agent_summary.append([get_idx(summary, i) for i in range(len(self.agent_list))])
        self.optimizer_state = full_optimizer_state
        return agent_summary
    @property
    def dynamics_model(self):
        return self.dynamics_model_list[0]

    def is_active_exploration_agent(self, idx):
        return idx == len(self.agent_list) - 1 and self.active_exploration_agent
