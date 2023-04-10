import time

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


def tree_stack(trees, axis=0):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l, axis=axis) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def tree_unstack(tree):
    """Takes a tree and turns it into a list of trees. Inverse of tree_stack.
    For example, given a tree ((a, b), c), where a, b, and c all have first
    dimension k, will make k trees
    [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]
    Useful for turning the output of a vmapped function into normal objects.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(l) for l in new_leaves]
    return new_trees


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
        self.reset_actor_params = reset_actor_params
        self._init_fn()

    def get_action_for_eval(self, obs: jax.Array, rng, agent_idx: int):
        policy = self.agent_list[0].get_eval_action
        actor_params = self.agent_list[agent_idx].actor_params
        actor_bias_obs = self.actor_normalizers['actor_bias_obs'][agent_idx]
        actor_scale_obs = self.actor_normalizers['actor_scale_obs'][agent_idx]
        normalized_obs = (obs - actor_bias_obs) / (actor_scale_obs + EPS)
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
            policy = self.agent_list[0].get_action
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

        def train_agent_step(
                train_rng,
                alpha_params,
                alpha_opt_state,
                actor_params,
                actor_opt_state,
                critic_params,
                target_critic_params,
                critic_opt_state,
                sim_transitions,
        ):
            return self.train_agent_step(
                train_rng=train_rng,
                alpha_params=alpha_params,
                alpha_opt_state=alpha_opt_state,
                actor_params=actor_params,
                actor_opt_state=actor_opt_state,
                critic_params=critic_params,
                target_critic_params=target_critic_params,
                critic_opt_state=critic_opt_state,
                sim_transitions=sim_transitions,
                agent_train_fn=self.agent_list[0].step,
                agent_train_steps=self.agent_list[0].train_steps,
            )

        self.train_step = jax.jit(jax.vmap(train_agent_step))

    @staticmethod
    def train_agent_step(train_rng,
                         alpha_params,
                         alpha_opt_state,
                         actor_params,
                         actor_opt_state,
                         critic_params,
                         target_critic_params,
                         critic_opt_state,
                         sim_transitions,
                         agent_train_fn,
                         agent_train_steps,
                         ):
        carry = [
            train_rng,
            alpha_params,
            alpha_opt_state,
            actor_params,
            actor_opt_state,
            critic_params,
            target_critic_params,
            critic_opt_state,
        ]
        ins = [sim_transitions]
        carry, outs = jax.lax.scan(agent_train_fn, carry, ins, length=agent_train_steps)
        alpha_params = carry[1]
        alpha_opt_state = carry[2]
        actor_params = carry[3]
        actor_opt_state = carry[4]
        critic_params = carry[5]
        target_critic_params = carry[6]
        critic_opt_state = carry[7]
        summary = outs[-1]
        return alpha_params, alpha_opt_state, actor_params, actor_opt_state, critic_params, target_critic_params, \
            critic_opt_state, summary

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
        ) for agent in self.agent_list]
        true_obs = buffer.obs[:buffer.size]
        train_steps = self.agent_list[0].train_steps
        batch_size = self.agent_list[0].batch_size
        alpha_params_list = []
        alpha_opt_state_list = []
        actor_params_list = []
        actor_opt_state_list = []
        critic_params_list = []
        target_critic_params_list = []
        critic_opt_state_list = []
        actor_obs_bias = []
        actor_obs_scale = []
        for j in range(len(self.agent_list)):
            alpha_params = self.init_agent_params['alpha_params'][j]
            alpha_opt_state = self.init_agent_opt_state['alpha_opt_state'][j]
            actor_params = self.init_agent_params['actor_params'][j]
            actor_opt_state = self.init_agent_opt_state['actor_opt_state'][j]
            critic_params = self.init_agent_params['critic_params'][j]
            target_critic_params = self.init_agent_params['target_critic_params'][j]
            critic_opt_state = self.init_agent_opt_state['critic_opt_state'][j]
            if not self.reset_actor_params and not self.is_active_exploration_agent(idx=j):
                alpha_params = self.agent_list[j].alpha_params
                actor_params = self.agent_list[j].actor_params
                critic_params = self.agent_list[j].critic_params
                target_critic_params = self.agent_list[j].target_critic_params
                alpha_opt_state = self.agent_list[j].alpha_opt_state
                actor_opt_state = self.agent_list[j].actor_opt_state
                critic_opt_state = self.agent_list[j].critic_opt_state
            alpha_params_list.append(alpha_params)
            alpha_opt_state_list.append(alpha_opt_state)
            actor_params_list.append(actor_params)
            actor_opt_state_list.append(actor_opt_state)
            critic_params_list.append(critic_params)
            target_critic_params_list.append(target_critic_params)
            critic_opt_state_list.append(critic_opt_state)
            actor_obs_bias.append(self.actor_normalizers['actor_bias_obs'][j])
            actor_obs_scale.append(self.actor_normalizers['actor_scale_obs'][j])
        agent_summary = []
        for i in range(self.train_steps_per_model_update):
            actor_obs_bias = []
            actor_obs_scale = []
            transitions_list = []
            for j in range(len(self.agent_list)):
                sim_buffer = simulation_buffers[j]
                agent = self.agent_list[j]
                evaluate_fn = self.dynamics_model_list[j].evaluate
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
                    policy=agent.get_action,
                    actor_params=actor_params_list[j],
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
                    policy_bias_obs=sim_buffer.state_normalizer.mean,
                    policy_scale_obs=sim_buffer.state_normalizer.std,
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
            alpha_params = tree_stack(alpha_params_list, axis=0)
            alpha_opt_state = tree_stack(alpha_opt_state_list, axis=0)
            actor_params = tree_stack(actor_params_list, axis=0)
            actor_opt_state = tree_stack(actor_opt_state_list, axis=0)
            critic_params = tree_stack(critic_params_list, axis=0)
            target_critic_params = tree_stack(target_critic_params_list, axis=0)
            critic_opt_state = tree_stack(critic_opt_state_list, axis=0)
            sim_transitions = tree_stack(transitions_list)
            train_rng = jax.random.split(rng, len(self.agent_list) + 1)
            rng = train_rng[0]
            train_rng = train_rng[1:]
            alpha_params, alpha_opt_state, actor_params, actor_opt_state, critic_params, target_critic_params, \
                critic_opt_state, summary = self.train_step(
                    train_rng=train_rng,
                    alpha_params=alpha_params,
                    alpha_opt_state=alpha_opt_state,
                    actor_params=actor_params,
                    actor_opt_state=actor_opt_state,
                    critic_params=critic_params,
                    target_critic_params=target_critic_params,
                    critic_opt_state=critic_opt_state,
                    sim_transitions=sim_transitions,
                )
            alpha_params_list = tree_unstack(alpha_params)
            alpha_opt_state_list = tree_unstack(alpha_opt_state)
            actor_params_list = tree_unstack(actor_params)
            actor_opt_state_list = tree_unstack(actor_opt_state)
            critic_params_list = tree_unstack(critic_params)
            target_critic_params_list = tree_unstack(target_critic_params)
            critic_opt_state_list = tree_unstack(critic_opt_state)
            agent_summary.append(tree_unstack(summary))
        for i in range(len(self.agent_list)):
            self.agent_list[i].alpha_params = alpha_params_list[i]
            self.agent_list[i].alpha_opt_state = alpha_opt_state_list[i]
            self.agent_list[i].actor_params = actor_params_list[i]
            self.agent_list[i].actor_opt_state = actor_opt_state_list[i]
            self.agent_list[i].critic_params = critic_params_list[i]
            self.agent_list[i].target_critic_params = target_critic_params_list[i]
            self.agent_list[i].critic_opt_state = critic_opt_state_list[i]
            self.actor_normalizers['actor_bias_obs'][i] = actor_obs_bias[i]
            self.actor_normalizers['actor_scale_obs'][i] = actor_obs_scale[i]
        return agent_summary

    @property
    def dynamics_model(self):
        return self.dynamics_model_list[0]

    def is_active_exploration_agent(self, idx):
        return idx == len(self.agent_list) - 1 and self.active_exploration_agent
