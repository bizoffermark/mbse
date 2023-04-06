import jax.random

from mbse.agents.actor_critic.sac import SACAgent, SACModelSummary
from gym.spaces import Box
from mbse.utils.replay_buffer import Transition, ReplayBuffer, JaxReplayBuffer
from typing import Callable, Union, Optional
import jax.numpy as jnp
from mbse.utils.utils import sample_trajectories
import functools
from mbse.optimizers.dummy_policy_optimizer import DummyPolicyOptimizer
from mbse.models.active_learning_model import ActiveLearningHUCRLModel, ActiveLearningPETSModel
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

        self.init_agent_params = {
            'alpha_params': [agent.alpha_params for agent in self.agent_list],
            'actor_params': [agent.actor_params for agent in self.agent_list],
            'critic_params': [agent.critic_params for agent in self.agent_list],
            'target_critic_params': [agent.target_critic_params for agent in self.agent_list],
        }

        self.init_agent_opt_state = {
            'alpha_opt_state': [agent.alpha_opt_state for agent in self.agent_list],
            'actor_opt_state': [agent.actor_opt_state for agent in self.agent_list],
            'critic_opt_state': [agent.critic_opt_state for agent in self.agent_list],
        }

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
        actor_bias_obs = jnp.zeros(obs_dim)
        actor_scale_obs = jnp.ones_like(actor_bias_obs)
        self.actor_normalizers = {
            'actor_bias_obs': [actor_bias_obs for agent in self.agent_list],
            'actor_scale_obs': [actor_scale_obs for agent in self.agent_list],
        }
        self._init_fn()

    def get_action_for_eval(self, obs: jax.Array, rng, agent_idx: int):
        policy = self.agent_list[agent_idx].get_eval_action
        actor_params = self.agent_list[agent_idx].actor_params
        actor_bias_obs = self.actor_normalizers['actor_bias_obs'][agent_idx]
        actor_scale_obs = self.actor_normalizers['actor_scale_obs'][agent_idx]
        normalized_obs = (obs - actor_bias_obs)/(actor_scale_obs + EPS)
        action = policy(
            actor_params=actor_params,
            obs=normalized_obs,
            rng=rng,
        )
        return action

    def get_action(self, obs: jax.Array, rng):
        return self.get_action_for_eval(obs=obs, rng=rng, agent_idx=0)

    def get_action_for_exploration(self, obs: jax.Array, rng, *args, **kwargs):
        if self.active_exploration_agent:
            policy = self.agent_list[-1].get_action
            actor_params = self.agent_list[-1].actor_params
            actor_bias_obs = self.actor_normalizers['actor_bias_obs'][-1]
            actor_scale_obs = self.actor_normalizers['actor_scale_obs'][-1]
            normalized_obs = (obs - actor_bias_obs) / (actor_scale_obs + EPS)
            action = policy(
                actor_params=actor_params,
                obs=normalized_obs,
                rng=rng,
            )
        else:
            policy = self.agent_list[0].get_action
            actor_params = self.agent_list[0].actor_params
            actor_bias_obs = self.actor_normalizers['actor_bias_obs'][0]
            actor_scale_obs = self.actor_normalizers['actor_scale_obs'][0]
            normalized_obs = (obs - actor_bias_obs) / (actor_scale_obs + EPS)
            action = policy(
                actor_params=actor_params,
                obs=normalized_obs,
                rng=rng,
            )
        return action

    def _init_fn(self):
        sim_buffer_kwargs = {
            'obs_shape': self.obs_dim,
            'action_shape': self.action_dim,
            'max_size': self.simulated_buffer_size,
            'normalize': self.normalize,
            'action_normalize': self.action_normalize,
        }

        def train_agent_fn(
                model_idx: int,
                rng,
                true_obs: jax.Array,
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
            return self._train_single_agent(
                rng=rng,
                true_obs=true_obs,
                evaluate_fn=self.dynamics_model_list[model_idx].evaluate,
                policy=self.agent_list[model_idx].get_action,
                agent_train_fn=self.agent_list[model_idx].step,
                sim_buffer_kwargs=sim_buffer_kwargs,
                init_alpha_params=self.init_agent_params['alpha_params'][model_idx],
                init_alpha_opt_state=self.init_agent_opt_state['alpha_opt_state'][model_idx],
                init_actor_params=self.init_agent_params['actor_params'][model_idx],
                init_actor_opt_state=self.init_agent_opt_state['actor_opt_state'][model_idx],
                init_critic_params=self.init_agent_params['critic_params'][model_idx],
                init_target_critic_params=self.init_agent_params['target_critic_params'][model_idx],
                init_critic_opt_state=self.init_agent_opt_state['critic_opt_state'][model_idx],
                sim_transition_ratio=self.sim_transitions_ratio,
                transitions_per_update=self.transitions_per_update,
                horizon=self.horizon,
                train_steps_per_model_update=self.train_steps_per_model_update,
                agent_batch_size=self.agent_list[model_idx].batch_size,
                agent_train_steps=self.agent_list[model_idx].train_steps,
                dynamics_params=dynamics_params,
                alpha=alpha,
                bias_obs=bias_obs,
                bias_act=bias_act,
                bias_out=bias_out,
                scale_obs=scale_obs,
                scale_act=scale_act,
                scale_out=scale_out,
                sampling_idx=sampling_idx,
            )

        self.train_agent_fns = []
        if self.active_exploration_agent:
            for i in range(len(self.dynamics_model_list) - 1):
                self.train_agent_fns.append(functools.partial(
                    train_agent_fn, model_idx=i
                ))

            def train_agent_active_exploration(
                    rng,
                    true_obs: jax.Array,
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
                return self._train_single_agent(
                    rng=rng,
                    true_obs=true_obs,
                    evaluate_fn=self.dynamics_model_list[-1].evaluate_for_exploration,
                    policy=self.agent_list[-1].get_action,
                    agent_train_fn=self.agent_list[-1].step,
                    sim_buffer_kwargs=sim_buffer_kwargs,
                    init_alpha_params=self.init_agent_params['alpha_params'][-1],
                    init_alpha_opt_state=self.init_agent_opt_state['alpha_opt_state'][-1],
                    init_actor_params=self.init_agent_params['actor_params'][-1],
                    init_actor_opt_state=self.init_agent_opt_state['actor_opt_state'][-1],
                    init_critic_params=self.init_agent_params['critic_params'][-1],
                    init_target_critic_params=self.init_agent_params['target_critic_params'][-1],
                    init_critic_opt_state=self.init_agent_opt_state['critic_opt_state'][-1],
                    sim_transition_ratio=self.sim_transitions_ratio,
                    transitions_per_update=self.transitions_per_update,
                    horizon=self.horizon,
                    train_steps_per_model_update=self.train_steps_per_model_update,
                    agent_batch_size=self.agent_list[-1].batch_size,
                    agent_train_steps=self.agent_list[-1].train_steps,
                    dynamics_params=dynamics_params,
                    alpha=alpha,
                    bias_obs=bias_obs,
                    bias_act=bias_act,
                    bias_out=bias_out,
                    scale_obs=scale_obs,
                    scale_act=scale_act,
                    scale_out=scale_out,
                    sampling_idx=sampling_idx,
                )

            self.train_agent_fns.append(jax.jit(train_agent_active_exploration))
        else:
            for i in range(len(self.dynamics_model_list)):
                self.train_agent_fns.append(functools.partial(
                    train_agent_fn, model_idx=i
                ))

    @staticmethod
    def _train_single_agent(rng,
                            true_obs: jax.Array,
                            evaluate_fn,
                            policy,
                            agent_train_fn,
                            sim_buffer_kwargs,
                            init_alpha_params,
                            init_alpha_opt_state,
                            init_actor_params,
                            init_actor_opt_state,
                            init_critic_params,
                            init_target_critic_params,
                            init_critic_opt_state,
                            sim_transition_ratio,
                            transitions_per_update,
                            horizon,
                            train_steps_per_model_update,
                            agent_batch_size,
                            agent_train_steps,
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
        simulation_buffer = JaxReplayBuffer(
            learn_deltas=False,
            **sim_buffer_kwargs,
        )
        alpha_params = init_alpha_params
        alpha_opt_state = init_alpha_opt_state
        actor_params = init_actor_params
        actor_opt_state = init_actor_opt_state
        critic_params = init_critic_params
        target_critic_params = init_target_critic_params
        critic_opt_state = init_critic_opt_state
        summaries = []
        batch_sim_buffer = int(sim_transition_ratio * transitions_per_update) * (simulation_buffer.size > 0)
        batch_true_buffer = int(transitions_per_update - batch_sim_buffer)
        for i in range(train_steps_per_model_update):
            buffer_rng, rng = jax.random.split(rng, 2)
            if batch_sim_buffer > 0:
                true_buffer_rng, sim_buffer_rng = jax.random.split(buffer_rng, 2)
                ind = jax.random.randint(true_buffer_rng, (batch_true_buffer,), 0, true_obs.shape[0])
                true_obs_sample = true_obs[ind]
                sim_obs = simulation_buffer.obs[simulation_buffer.size]
                ind = jax.random.randint(sim_buffer_rng, (batch_sim_buffer,), 0, sim_obs.shape[0])
                sim_obs_sample = sim_obs[ind]
                obs = jnp.concatenate([true_obs_sample, sim_obs_sample], axis=0)
            else:
                ind = jax.random.randint(buffer_rng, (batch_true_buffer,), 0, true_obs.shape[0])
                obs = true_obs[ind]
            simulation_key, rng = jax.random.split(rng, 2)
            simulated_transitions = _simulate_dynamics(
                obs=obs,
                policy=policy,
                actor_params=actor_params,
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
                horizon=horizon,
            )
            simulation_buffer.add(transition=simulated_transitions)
            batch_sim_buffer = int(sim_transition_ratio * transitions_per_update) * (simulation_buffer.size > 0)
            batch_true_buffer = int(transitions_per_update - batch_sim_buffer)
            train_rng, rng = jax.random.split(rng, 2)
            sim_buffer_rng, rng = jax.random.split(rng, 2)
            train_rng, rng = jax.random.split(rng, 2)
            sim_buffer_rng, rng = jax.random.split(rng, 2)
            if simulation_buffer.size > agent_batch_size:
                sim_transitions = simulation_buffer.sample(sim_buffer_rng,
                                                           batch_size=int(agent_batch_size * agent_train_steps)
                                                           )
                sim_transitions = sim_transitions.reshape(agent_train_steps, agent_batch_size)

                carry = [
                    train_rng,
                    alpha_params,
                    alpha_opt_state,
                    actor_params,
                    actor_opt_state,
                    critic_params,
                    target_critic_params,
                    critic_opt_state,
                    SACModelSummary(),
                    0,
                    sim_transitions,
                ]
                carry, outs = jax.lax.scan(agent_train_fn, carry, xs=None, length=agent_train_steps)
                alpha_params = carry[1]
                alpha_opt_state = carry[2]
                actor_params = carry[3]
                actor_opt_state = carry[4]
                critic_params = carry[5]
                target_critic_params = carry[6]
                critic_opt_state = carry[7]
                summary = carry[8]
                summaries.append(summary)
        actor_bias_obs = simulation_buffer.state_normalizer.mean
        actor_scale_obs = simulation_buffer.state_normalizer.std
        return alpha_params, alpha_opt_state, actor_params, actor_opt_state, critic_params, target_critic_params, \
            critic_opt_state, summaries, actor_bias_obs, actor_scale_obs

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
        agent_training_summary = []
        for i in range(len(self.agent_list)):
            agent_rng, rng = jax.random.split(rng, 2)
            true_obs = buffer.obs[:buffer.size]
            train_agent_fn = self.train_agent_fns[i]
            (alpha_params, alpha_opt_state, actor_params, actor_opt_state, critic_params, target_critic_params,
             critic_opt_state, summaries, actor_bias_obs, actor_scale_obs) = train_agent_fn(
                rng=rng,
                true_obs=true_obs,
                dynamics_params=dynamics_params,
                alpha=alpha,
                bias_obs=bias_obs,
                bias_act=bias_act,
                bias_out=bias_out,
                scale_obs=scale_obs,
                scale_act=scale_act,
                scale_out=scale_out,
                sampling_idx=sampling_idx,
            )
            self.agent_list[i].alpha_params = alpha_params
            self.agent_list[i].alpha_opt_state = alpha_opt_state
            self.agent_list[i].actor_params = actor_params
            self.agent_list[i].actor_opt_state = actor_opt_state
            self.agent_list[i].critic_params = critic_params
            self.agent_list[i].target_critic_params = target_critic_params
            self.agent_list[i].critic_opt_state = critic_opt_state
            self.actor_normalizers['actor_bias_obs'][i] = actor_bias_obs
            self.actor_normalizers['actor_scale_obs'][i] = actor_scale_obs
            agent_training_summary.append(summaries)

        return agent_training_summary

    @property
    def dynamics_model(self):
        return self.dynamics_model_list[0]
