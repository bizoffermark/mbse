import jax

from mbse.models.dynamics_model import DynamicsModel
from mbse.optimizers.dummy_optimizer import DummyOptimizer
import gym
from mbse.utils.utils import rollout_actions, sample_trajectories
from mbse.utils.replay_buffer import Transition
from mbse.agents.dummy_agent import DummyAgent
import numpy as np


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
        obs = np.tile(obs, (self.n_particles, 1))
        #eval_func = lambda seq: rollout_actions(seq,
        #                                        initial_state=obs,
        #                                        dynamics_model=self.dynamics_model,
        #                                        reward_model=self.reward_model,
        #                                        rng=rng)
        rollout_rng, optimizer_rng = jax.random.split(rng, 2)
        eval_func = lambda seq: sample_trajectories(
            dynamics_model=self.dynamics_model,
            reward_model=self.reward_model,
            init_state=obs,
            horizon=self.policy_optimzer.action_dim[0],
            key=rollout_rng,
            actions=seq,
        )

        @jax.jit
        def sum_rewards(seq):
            transition = eval_func(seq)
            return transition.reward.sum()
        action_sequence = self.policy_optimzer.optimize(sum_rewards, optimizer_rng)
        return action_sequence[0, ...]

    def train_step(self,
                   rng,
                   tran: Transition,
                   ):
        model_training_summary = self.dynamics_model.train_step(tran)
        return model_training_summary

