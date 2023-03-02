from gym.wrappers import RescaleAction, TimeLimit
from mbse.utils.vec_env.env_util import make_vec_env
from mbse.models.environment_models.pendulum_swing_up import PendulumReward, CustomPendulumEnv
from mbse.models.active_learning_model import ActiveLearningModel
from mbse.agents.model_based.mb_active_exploration_agent import MBActiveExplorationAgent
from mbse.optimizers.cross_entropy_optimizer import CrossEntropyOptimizer
from mbse.trainer.model_based.model_based_trainer import ModelBasedTrainer as Trainer
import numpy as np
import time
import json
import os
import sys
import argparse
from experiments.util import Logger, hash_dict, NumpyArrayEncoder, DATA_DIR
import wandb


def experiment(use_wandb: bool, exp_name: str, env_name: str, time_limit: int, n_envs: int,
               num_samples: int, num_elites: int, num_steps: int, horizon: int, n_particles: int, reset_model: bool,
               num_ensembles: int, hidden_layers: int, num_neurons: int, beta: float,
               pred_diff: bool, batch_size: int, eval_freq: int, max_train_steps: int, buffer_size: int,
               exploration_steps: int, eval_episodes: int, train_freq: int, train_steps: int, rollout_steps: int,
               normalize: bool, action_normalize: bool, validate: bool, record_test_video: bool,
               validation_buffer_size: int,
               seed: int, exploration_strategy: str):
    """ Run experiment for a given method and environment. """

    """ Environment """
    wrapper_cls = lambda x: RescaleAction(
        TimeLimit(x, max_episode_steps=time_limit),
        min_action=-1,
        max_action=1,
    )
    if env_name == "Pendulum-v1":
        env = make_vec_env(env_id=CustomPendulumEnv, wrapper_class=wrapper_cls, n_envs=n_envs, seed=seed)

    else:
        env = make_vec_env(env_id=env_name, wrapper_class=wrapper_cls, seed=seed, n_envs=n_envs, env_kwargs={
            'render_mode': 'rgb_array'
        }
                           )

    features = [num_neurons] * hidden_layers
    reward_model = PendulumReward(action_space=env.action_space)
    reward_model.set_bounds(max_action=1.0)
    if exploration_strategy == 'Mean':
        beta = 0.0
    dynamics_model = ActiveLearningModel(
        action_space=env.action_space,
        observation_space=env.observation_space,
        num_ensemble=num_ensembles,
        reward_model=reward_model,
        features=features,
        pred_diff=pred_diff,
        beta=beta,
        seed=seed,
    )

    model_based_agent_fn = lambda x, y, z, v: \
        MBActiveExplorationAgent(
            use_wandb=x,
            validate=y,
            train_steps=z,
            batch_size=v,
            action_space=env.action_space,
            observation_space=env.action_space,
            dynamics_model=dynamics_model,
            n_particles=n_particles,
            reset_model=reset_model,
            # policy_optimizer=policy_optimizer,
            policy_optimizer=CrossEntropyOptimizer(
                upper_bound=1,
                num_samples=num_samples,
                num_elites=num_elites,
                num_steps=num_steps,
                action_dim=(horizon, env.action_space.shape[0] + env.observation_space.shape[0])),
        )

    USE_WANDB = use_wandb
    uniform_exploration = False
    if exploration_strategy == 'Uniform':
        uniform_exploration = True
    trainer = Trainer(
        agent_fn=model_based_agent_fn,
        # model_free_agent=model_free_agent,
        env=env,
        buffer_size=buffer_size,
        max_train_steps=max_train_steps,
        exploration_steps=exploration_steps,
        batch_size=batch_size,
        use_wandb=USE_WANDB,
        eval_episodes=eval_episodes,
        eval_freq=eval_freq,
        train_freq=train_freq,
        train_steps=train_steps,
        rollout_steps=rollout_steps,
        normalize=normalize,
        action_normalize=action_normalize,
        learn_deltas=dynamics_model.pred_diff,
        validate=validate,
        record_test_video=record_test_video,
        validation_buffer_size=validation_buffer_size,
        seed=seed,
        uniform_exploration=uniform_exploration,
    )
    if USE_WANDB:
        wandb.init(
            project=exp_name,
            group=exploration_strategy,
        )
    trainer.train()

    result_dict = {
    }
    return result_dict


def main(args):
    """"""
    from pprint import pprint
    print(args)
    """ generate experiment hash and set up redirect of output streams """
    exp_hash = hash_dict(args.__dict__)
    if args.exp_result_folder is not None:
        os.makedirs(args.exp_result_folder, exist_ok=True)
        log_file_path = os.path.join(args.exp_result_folder, '%s.log ' % exp_hash)
        logger = Logger(log_file_path)
        sys.stdout = logger
        sys.stderr = logger

    pprint(args.__dict__)
    print('\n ------------------------------------ \n')

    """ Experiment core """
    t_start = time.time()
    np.random.seed(args.seed + 5)

    eval_metrics = experiment(
        use_wandb=args.use_wandb,
        exp_name=args.exp_name,
        env_name=args.env_name,
        time_limit=args.time_limit,
        n_envs=args.n_envs,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        num_steps=args.num_steps,
        horizon=args.horizon,
        n_particles=args.n_particles,
        reset_model=args.reset_model,
        beta=args.beta,
        num_ensembles=args.num_ensembles,
        pred_diff=args.pred_diff,
        batch_size=args.batch_size,
        eval_freq=args.eval_freq,
        max_train_steps=args.max_train_steps,
        buffer_size=args.buffer_size,
        exploration_steps=args.exploration_steps,
        eval_episodes=args.eval_episodes,
        train_freq=args.train_freq,
        train_steps=args.train_steps,
        rollout_steps=args.rollout_steps,
        normalize=args.normalize,
        action_normalize=args.action_normalize,
        validate=args.validate,
        record_test_video=args.record_test_video,
        validation_buffer_size=args.validation_buffer_size,
        seed=args.seed,
        hidden_layers=args.hidden_layers,
        num_neurons=args.num_neurons,
        exploration_strategy=args.exploration_strategy,
    )

    t_end = time.time()

    """ Save experiment results and configuration """
    results_dict = {
        'evals': eval_metrics,
        'params': args.__dict__,
        'duration_total': t_end - t_start
    }

    if args.exp_result_folder is None:
        from pprint import pprint
        pprint(results_dict)
    else:
        exp_result_file = os.path.join(args.exp_result_folder, '%s.json' % exp_hash)
        with open(exp_result_file, 'w') as f:
            json.dump(results_dict, f, indent=4, cls=NumpyArrayEncoder)
        print('Dumped results to %s' % exp_result_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Active-Exploration-run')

    # general experiment args
    parser.add_argument('--exp_name', type=str, required=True, default='active_exploration')
    parser.add_argument('--use_wandb', default=True, action="store_true")
    # env experiment args
    parser.add_argument('--env_name', type=str, default='Pendulum-v1')
    parser.add_argument('--time_limit', type=int, default=200)
    parser.add_argument('--n_envs', type=int, default=1)

    # optimizer experiment args
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--num_elites', type=int, default=50)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=20)

    # agent experiment args
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_particles', type=int, default=10)
    parser.add_argument('--reset_model', default=True, action="store_true")

    # dynamics_model experiment args
    parser.add_argument('--num_ensembles', type=int, default=5)
    parser.add_argument('--hidden_layers', type=int, default=2)
    parser.add_argument('--num_neurons', type=int, default=128)
    parser.add_argument('--pred_diff', default=True, action="store_true")
    parser.add_argument('--beta', type=float, default=1.0)

    # trainer experiment args
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--max_train_steps', type=int, default=2500)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--exploration_steps', type=int, default=0)
    parser.add_argument('--eval_episodes', type=int, default=1)
    parser.add_argument('--train_freq', type=int, default=1)
    parser.add_argument('--train_steps', type=int, default=5)
    parser.add_argument('--rollout_steps', type=int, default=1)
    parser.add_argument('--normalize', default=True, action="store_true")
    parser.add_argument('--action_normalize', default=True, action="store_true")
    parser.add_argument('--validate', default=True, action="store_true")
    parser.add_argument('--record_test_video', default=True, action="store_true")
    parser.add_argument('--validation_buffer_size', type=int, default=100000)
    parser.add_argument('--exploration_strategy', type=str, default='Optimistic')

    # general args
    parser.add_argument('--exp_result_folder', type=str, default=None)
    parser.add_argument('--seed', type=int, default=834, help='random number generator seed')

    args = parser.parse_args()
    main(args)
