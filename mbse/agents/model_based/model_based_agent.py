import jax

from mbse.models.dynamics_model import DynamicsModel
from mbse.optimizers.dummy_optimizer import DummyOptimizer
import gym
from mbse.utils.utils import rollout_actions, sample_trajectories
from mbse.utils.replay_buffer import Transition
from mbse.agents.dummy_agent import DummyAgent
import numpy as np
import jax.numpy as jnp


class ModelBasedAgent(DummyAgent):

    def __init__(
            self,
            action_space: gym.spaces.box,
            observation_space: gym.spaces.box,
            dynamics_model: DynamicsModel,
            policy_optimizer: DummyOptimizer,
            reward_model,
            discount: float = 0.99,
            n_particles: int = 10,
    ):
        super(ModelBasedAgent, self).__init__()
        self.action_space = action_space
        self.observation_space = observation_space
        self.dynamics_model = dynamics_model
        self.policy_optimzer = policy_optimizer
        self.discount = discount
        self.reward_model = reward_model
        self.n_particles = n_particles

    def act_in_jax(self, obs, rng, eval=False):

        def optimize(init_state, key, optimizer_key):
            eval_func = lambda seq, x, k: sample_trajectories(
                dynamics_model=self.dynamics_model,
                reward_model=self.reward_model,
                init_state=x,
                horizon=self.policy_optimzer.action_dim[-2],
                key=k,
                actions=seq,
            )

            @jax.jit
            def sum_rewards(seq):
                seq = jnp.repeat(jnp.expand_dims(seq, 0), self.n_particles, 0)
                transition = eval_func(seq, init_state, key)
                return transition.reward.mean()

            action_seq, reward = self.policy_optimzer.optimize(
                sum_rewards,
                optimizer_key
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
                   tran: Transition,
                   val: Transition = None,
                   ):
        model_training_summary = self.dynamics_model.train_step(tran, val)
        return model_training_summary

