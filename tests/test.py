"""
    Credits: https://github.com/sfujim/TD3
"""
from argparse_dataclass import ArgumentParser
from pathlib import Path
from typing import Any
import yaml
import gym
from mbse.trainer.model_free_trainer import ModelFreeTrainer as Trainer
from mbse.agents.actor_critic.sac import SACAgent
from jax import random
from dataclasses import dataclass, field
from typing import Optional
import wandb
from gym.wrappers.time_limit import TimeLimit
from gym.wrappers.rescale_action import RescaleAction

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
    env = gym.make(kwargs['env_id'])
    env = TimeLimit(env, max_episode_steps=kwargs['time_limit'])
    env = RescaleAction(env, min_action=-1, max_action=1)
    #key = random.PRNGKey(args.seed)
    agent = SACAgent(
        action_dim=env.action_space.shape[0],
        sample_act=env.action_space.sample(),
        sample_obs=env.observation_space.sample(),
        discount=kwargs['agent']['discount'],
        initial_log_alpha=kwargs['agent']['initial_log_alpha'],
        lr_actor=kwargs['agent']['lr_actor'],
        lr_critic=kwargs['agent']['lr_critic'],
        lr_alpha=kwargs['agent']['lr_alpha'],
        actor_features=kwargs['agent']['actor_features'],
        critic_features=kwargs['agent']['critic_features'],
        scale_reward=kwargs['agent']['scale_reward'],
    )

    USE_WANDB = True
    trainer = Trainer(
        env=env,
        agent=agent,
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


