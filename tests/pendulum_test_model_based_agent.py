from argparse_dataclass import ArgumentParser
from typing import Any
import yaml
from mbse.trainer.model_based.model_based_trainer import ModelBasedTrainer as Trainer
from mbse.agents.model_based.model_based_agent import ModelBasedAgent
from dataclasses import dataclass, field
import wandb
from gym.wrappers.time_limit import TimeLimit
from gym.wrappers.rescale_action import RescaleAction
from mbse.models.environment_models.pendulum_swing_up import PendulumSwingUpEnv, PendulumDynamicsModel
from mbse.optimizers.cross_entropy_optimizer import CrossEntropyOptimizer
from mbse.agents.actor_critic.sac import SACAgent
from mbse.utils.vec_env.env_util import make_vec_env
from mbse.models.bayesian_dynamics_model import BayesianDynamicsModel


OptState = Any


@dataclass
class Experiment:
    """Definition of Experiment dataclass."""
    config: str = field(
        metadata=dict(help="File with config.")
    )


if __name__ == "__main__":
    parser = ArgumentParser(Experiment)
    args = parser.parse_args()
    with open(args.config, "r") as file:
        kwargs = yaml.safe_load(file)
    wrapper_cls = lambda x: RescaleAction(
        TimeLimit(x, max_episode_steps=kwargs['time_limit']),
        min_action=-1,
        max_action=1,
    )
    env = make_vec_env(PendulumSwingUpEnv, wrapper_class=wrapper_cls, n_envs=1)

    reward_model = env.envs[0].reward_model()
    dynamics_model = PendulumDynamicsModel(env=env.envs[0])
    # dynamics_model = BayesianDynamicsModel(action_space=env.action_space,
    #                                       observation_space=env.observation_space)

    model_free_agent = SACAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        discount=kwargs['agent']['discount'],
        lr_actor=kwargs['agent']['lr_actor'],
        lr_critic=kwargs['agent']['lr_critic'],
        lr_alpha=kwargs['agent']['lr_alpha'],
        actor_features=kwargs['agent']['actor_features'],
        critic_features=kwargs['agent']['critic_features'],
        scale_reward=kwargs['agent']['scale_reward'],
        tau=kwargs['agent']['tau'],
    )

    model_based_agent = ModelBasedAgent(
        action_space=env.action_space,
        observation_space=env.action_space,
        dynamics_model=dynamics_model,
        reward_model=reward_model,
        policy_optimizer=CrossEntropyOptimizer(
            upper_bound=1,
            num_samples=500,
            num_elites=50,
            num_steps=5,
            action_dim=(30, env.action_space.shape[0])),
    )

    USE_WANDB = False

    trainer = Trainer(
        agent=model_based_agent,
        # model_free_agent=model_free_agent,
        env=env,
        buffer_size=kwargs['trainer']['buffer_size'],
        max_train_steps=kwargs['trainer']['max_train_steps'],
        exploration_steps=kwargs['trainer']['exploration_steps'],
        batch_size=kwargs['trainer']['batch_size'],
        use_wandb=USE_WANDB,
        eval_episodes=kwargs['trainer']['eval_episodes'],
        eval_freq=kwargs['trainer']['eval_freq'],
        train_freq=kwargs['trainer']['train_freq'],
        train_steps=kwargs['trainer']['train_steps'],
        rollout_steps=kwargs['trainer']['rollout_steps'],
    )
    if USE_WANDB:
        wandb.init(
            project=kwargs['project_name'],
        )
    trainer.train()
