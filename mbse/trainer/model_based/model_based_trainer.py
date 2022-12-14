from mbse.models.dynamics_model import DynamicsModel
from mbse.optimizers.dummy_optimizer import DummyOptimizer
import gym.envs as Env
import jax


class ModelBasedTrainer(object):

    def __init__(
            self,
            env: Env,
            dynamics_model: DynamicsModel,
            policy_optimizer: DummyOptimizer,
            buffer_size: int = 1e6,
            max_train_steps: int = 1e6,
            batch_size: int = 256,
            train_freq: int = 100,
            train_steps: int = 100,
            eval_freq: int = 1000,
            seed: int = 0,
            exploration_steps: int = 1e4,
            rollout_steps: int = 200,
            eval_episodes: int = 100,
            agent_name: str = "MBAgent",
            use_wandb: bool = True,
            ):
        self.env = env
        self.dynamics_model = dynamics_model
        self.policy_optimizer = policy_optimizer
        self.buffer_size = buffer_size
        self.max_train_steps = max_train_steps
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.train_steps = train_steps
        self.eval_freq = eval_freq
        self.rng = jax.random.PRNGKey(seed)
        self.exploration_steps = exploration_steps
        self.rollout_steps = rollout_steps
        self.eval_episodes = eval_episodes
        self.agent_name = agent_name
        self.use_wandb = use_wandb


