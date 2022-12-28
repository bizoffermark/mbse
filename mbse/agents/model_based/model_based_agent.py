import jax

from mbse.models.dynamics_model import DynamicsModel
from mbse.optimizers.dummy_optimizer import DummyOptimizer
import gym
from mbse.utils.utils import rollout_actions
from mbse.utils.replay_buffer import Transition
from mbse.agents.dummy_agent import DummyAgent


class ModelBasedAgent(DummyAgent):

    def __init__(
            self,
            action_space: gym.spaces.box,
            observation_space: gym.spaces.box,
            dynamics_model: DynamicsModel,
            policy_optimizer: DummyOptimizer,
            reward_model,
            discount: float = 0.99,
    ):
        super(ModelBasedAgent, self).__init__()
        self.action_space = action_space
        self.observation_space = observation_space
        self.dynamics_model = dynamics_model
        self.policy_optimzer = policy_optimizer
        self.discount = discount
        self.reward_model = reward_model

    def act(self, obs, rng=None):
        eval_func = lambda seq: rollout_actions(seq,
                                                initial_state=obs,
                                                dynamics_model=self.dynamics_model,
                                                reward_model=self.reward_model,
                                                rng=rng)
        @jax.jit
        def sum_rewards(seq):
            s, rewards = eval_func(seq)
            return rewards.sum(axis=0)
        action_sequence = self.policy_optimzer.optimize(sum_rewards)
        return action_sequence[0, ...]

    def train_step(self,
                   rng,
                   tran: Transition,
                   ):
        model_training_summary = self.dynamics_model.train_step(self, tran)
        return model_training_summary

