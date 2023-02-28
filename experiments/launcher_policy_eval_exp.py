from experiments.util import generate_base_command, generate_run_commands, hash_dict, sample_flag, RESULT_DIR

import experiments.active_exploration_exp
import argparse
import numpy as np
import copy
import os
import itertools

applicable_configs = {
    'general': ['use_wandb', 'exp_name'],
    'env': ['env_name', 'time_limit', 'n_envs'],
    'optimizer': ['num_samples', 'num_elites', 'num_steps', 'horizon'],
    'agent': ['discount', 'n_particles'],
    'dynamics_model': ['num_ensembles', 'hidden_layers', 'num_neurons', 'pred_diff'],
    'trainer': ['batch_size', 'eval_freq', 'max_train_steps', 'buffer_size',
                'exploration_steps', 'eval_episodes', 'train_freq', 'train_steps', 'rollout_steps',
                'validate', 'normalize', 'action_normalize', 'record_test_video', 'validation_buffer_size'],
}

default_configs = {
    'use_wandb': True,
    'exp_name': 'Pendulum-Active-Exploration',
    'env_name': 'Pendulum-v1',
    'time_limit': 200,
    'n_envs': 5,
    'num_samples': 500,
    'num_elites': 50,
    'num_steps': 10,
    'horizon': 20,
    'discount': 1.0,
    'n_particles': 10,
    'num_ensembles': 5,
    'hidden_layers': 2,
    'num_neurons': 128,
    'pred_diff': True,
    'batch_size': 128,
    'eval_freq': 1000,
    'max_train_steps': 2500,
    'buffer_size': 1000000,
    'exploration_steps': 0,
    'eval_episodes': 1,
    'train_freq': 1,
    'train_steps': 5,
    'rollout_steps': 1,
    'validate': True,
    'normalize': True,
    'action_normalize': True,
    'record_test_video': True,
    'validation_buffer_size': 100000,
}

search_ranges = {
}

EXPLORATION_STRATEGY = ['Uniform', 'Optimistic']
# check consistency of configuration dicts
assert set(itertools.chain(*list(applicable_configs.values()))) == {*default_configs.keys(), *search_ranges.keys()}


def main(args):
    rds = np.random.RandomState(args.seed)
    assert args.num_seeds_per_hparam < 100
    init_seeds = list(rds.randint(0, 10 ** 6, size=(100,)))

    # determine name of experiment
    exp_base_path = os.path.join(RESULT_DIR, args.exp_name)
    exp_path = os.path.join(exp_base_path, f'{args.exp_name}')

    command_list = []
    for _ in range(args.num_hparam_samples):
        # transfer flags from the args
        flags = copy.deepcopy(args.__dict__)
        [flags.pop(key) for key in ['seed', 'num_hparam_samples', 'num_seeds_per_hparam', 'exp_name', 'num_cpus',
                                    'num_gpus', 'launch_mode']]

        # randomly sample flags
        for flag in default_configs:
            if flag in search_ranges:
                flags[flag] = sample_flag(sample_spec=search_ranges[flag], rds=rds)
            else:
                flags[flag] = default_configs[flag]

        for exploration_strategy in EXPLORATION_STRATEGY:
            # determine subdir which holds the repetitions of the exp
            flags_hash = hash_dict(flags)
            flags['exploration_strategy'] = exploration_strategy
            flags['exp_result_folder'] = os.path.join(exp_path, flags_hash)

            for j in range(args.num_seeds_per_hparam):
                seed = init_seeds[j]
                cmd = generate_base_command(experiments.active_exploration_exp, flags=dict(**flags, **{'seed': seed}))
                command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list, num_cpus=args.num_cpus, num_gpus=args.num_gpus, mode=args.launch_mode,
                          promt=True,
                          mem=16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=834, help='random number generator seed')
    parser.add_argument('--exp_name', type=str, default='Pendulum-ActiveExploration')
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--num_cpus', type=int, default=8, help='number of cpus to use')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--launch_mode', type=str, default='euler', help='how to launch the experiments')
    parser.add_argument('--num_hparam_samples', type=int, default=1)
    parser.add_argument('--num_seeds_per_hparam', type=int, default=3)

    args = parser.parse_args()
    main(args)
