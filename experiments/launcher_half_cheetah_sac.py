from experiments.util import generate_base_command, generate_run_commands, hash_dict, sample_flag, RESULT_DIR
import yaml
import argparse
import numpy as np
import copy
import os
import itertools

applicable_configs = {
    'general': ['use_wandb', 'exp_name', 'logs_dir'],
    'env': ['time_limit', 'n_envs'],
    'agent': ['discount', 'lr_actor', 'lr_critic', 'lr_alpha', 'num_neurons_actor', 'hidden_layers_actor',
              'hidden_layers_critic', 'num_neurons_critic', 'tau',
              'scale_reward', 'tune_entropy_coef', 'init_ent_coef', 'batch_size'],
    'trainer': ['eval_freq', 'total_train_steps', 'buffer_size',
                'exploration_steps', 'eval_episodes', 'train_freq', 'train_steps', 'rollout_steps'],
}

default_configs = {
    'use_wandb': True,
    'exp_name': 'env_test_cheetah',
    'logs_dir': './',
    'time_limit': 1000,
    'n_envs': 1,
    'discount': 0.99,
    'lr_actor': 0.0003,
    'lr_critic': 0.0003,
    'lr_alpha': 0.0003,
    'hidden_layers_actor': 2,
    'hidden_layers_critic': 2,
    'num_neurons_actor': 256,
    'num_neurons_critic': 256,
    'tau': 0.005,
    'scale_reward': 1.0,
    'tune_entropy_coef': True,
    'init_ent_coef': 1.0,
    'batch_size': 256,
    'train_steps': 1500,
    'eval_freq': 10,
    'total_train_steps': 1000000,
    'buffer_size': 1000000,
    'exploration_steps': 0,
    'eval_episodes': 1,
    'train_freq': 1,
    'rollout_steps': 1000,
}

search_ranges = {
}


def main(args):
    import half_cheetah_exp.sac_cheetah_test as experiment
    EXPLORATION_STRATEGY = ['PetsCheetah', 'true']
    # check consistency of configuration dicts
    assert set(itertools.chain(*list(applicable_configs.values()))) == {*default_configs.keys(), *search_ranges.keys()}
    rds = np.random.RandomState(args.seed)
    assert args.num_seeds_per_hparam < 100
    init_seeds = list(rds.randint(0, 10 ** 6, size=(100,)))

    command_list = []
    for _ in range(args.num_hparam_samples):
        # transfer flags from the args
        flags = copy.deepcopy(args.__dict__)
        if flags['launch_mode'] == 'euler':
            logs_dir = '/cluster/scratch/'
            logs_dir += flags['user_name'] + '/'
        else:
            logs_dir = 'half_cheetah_exp/'
        [flags.pop(key) for key in ['seed', 'num_hparam_samples', 'num_seeds_per_hparam', 'num_cpus',
                                    'num_gpus', 'launch_mode', 'user_name', 'long_run']]

        exp_base_path = os.path.join(RESULT_DIR, "CheetahTestSac")
        exp_path = os.path.join(exp_base_path, "CheetahTestSac")

        # randomly sample flags
        for flag in default_configs:
            if flag in search_ranges:
                flags[flag] = sample_flag(sample_spec=search_ranges[flag], rds=rds)
            else:
                flags[flag] = default_configs[flag]
        flags['logs_dir'] = logs_dir
        for exploration_strategy in EXPLORATION_STRATEGY:
            # determine subdir which holds the repetitions of the exp
            flags_hash = hash_dict(flags)
            flags['exploration_strategy'] = exploration_strategy
            flags['exp_result_folder'] = os.path.join(exp_path, flags_hash)

            for j in range(args.num_seeds_per_hparam):
                seed = init_seeds[j]
                cmd = generate_base_command(experiment, flags=dict(**flags, **{'seed': seed}))
                command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list, num_cpus=args.num_cpus, num_gpus=args.num_gpus, mode=args.launch_mode,
                          long=args.long_run,
                          promt=True,
                          mem=16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=834, help='random number generator seed')
    parser.add_argument('--num_cpus', type=int, default=8, help='number of cpus to use')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--launch_mode', type=str, default='local', help='how to launch the experiments')
    parser.add_argument('--user_name', type=str, default='sukhijab', help='name of user launching experiments')
    parser.add_argument('--num_hparam_samples', type=int, default=1)
    parser.add_argument('--num_seeds_per_hparam', type=int, default=3)
    parser.add_argument('--long_run', default=False, action="store_true")

    args = parser.parse_args()
    main(args)
