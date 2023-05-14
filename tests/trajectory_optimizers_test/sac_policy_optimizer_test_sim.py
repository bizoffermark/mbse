import numpy as np

import sys
sys.path.append('/home/bizoffermark/workspace/ode/pyur5/include/mbse')
sys.path.append('/home/bizoffermark/workspace/ode/pyur5/include/pyur5')
from mbse.models.ur5_dynamics_model import Ur5PendulumDynamicsModel
from mbse.optimizers.sac_based_optimizer import SACOptimizer
import pickle
import jax 
import jax.numpy as jnp

from mbse.utils.replay_buffer import ReplayBuffer, Transition
import time
import cloudpickle
import wandb 
import matplotlib.pyplot as plt
from gym.envs.classic_control.pendulum import angle_normalize

# wandb.init(project="sac_policy")

def rollout_random_policy(train_horizon, true_dynamics, data, rng):
    # model = true_dynamics.model
    rng, reset_rng = jax.random.split(rng, 2)
    # with open("/home/bizoffermark/workspace/ode/pyur5/data_pkl/data_{}.pkl".format(train_horizon), "rb") as f:
    #     data = pickle.load(f)
    data_train = data['train']
    obs_vec = data_train['x'][:,:-1]    
    action_vec = data_train['u']
    next_obs_vec = data_train['y']
    num_points = obs_vec.shape[0]
    done_vec = jnp.array([False]* num_points).reshape(-1, 1)
    reward_vec = np.zeros((num_points,))

    for i in range(obs_vec.shape[0]):
        _, reward = true_dynamics.evaluate(obs_vec[i], action_vec[i], rescaled=True)
        reward_vec[i: (i + 1)] = reward.reshape(-1)
    
    transitions = Transition(
        obs=obs_vec,
        action=action_vec,
        reward=reward_vec,
        next_obs=next_obs_vec,
        done=done_vec,
    )
    
    # reset_seed = jax.random.randint(
    #     reset_rng,
    #     (1,),
    #     minval=0,
    #     maxval=num_steps).item()

    # reset 
    # obs = model.obs_space.sample()
    print("obs_vec of shape {} is {}".format(obs_vec.shape, obs_vec))
    print("reward_vec of shape {} is {}".format(reward_vec.shape, reward_vec))
    print("next_obs_vec of shape {} is {}".format(next_obs_vec.shape, next_obs_vec))
    print("done_vec of shape {} is {}".format(done_vec.shape, done_vec))
    return transitions


def make_plot(train_horizon, n_model, n_horizon, solver, j_init, sac_policy):
    '''
    
    '''
    true_dynamics.model.n_horizon = n_horizon
    
    x_test, u_test, cost = true_dynamics.model.forward_traj(x[j_init], n_horizon, solver, sac_policy)
    xs = x_test.tolist()
    theta = [x[0] for x in xs]
    theta = [angle_normalize(x) for x in theta]
    theta_dot = [x[1] for x in xs]
    p_ee = [x[2] for x in xs]
    v_ee = [x[3] for x in xs]
    
    datas = [theta, theta_dot, p_ee, v_ee]
    fig, axs = plt.subplots(5, 1, figsize=(20, 20))
    fig.suptitle("{}_{}_{}_{}_{}".format(train_horizon, n_model, n_horizon, solver, j_init))
    for i in range(x_test.shape[1]):
        axs[i].plot(datas[i])
        axs[i].set_title(headers[i])
    axs[-1].plot(u_test)
    axs[-1].set_title("u")
    
    fig.savefig("/home/bizoffermark/workspace/ode/pyur5/figs/{}_{}_{}_{}_{}.png".format(train_horizon, n_model, n_horizon, solver, j_init))
    log = {"plot_{}".format(n_horizon): wandb.Image(fig), "cost_{}".format(n_horizon): cost}

    return log


def train_sac_policy_optimizer(config, rng):
    sac_kwargs = config.sac_kwargs
    # sac_kwargs['lr_actor'] = config.lr_actor
    # sac_kwargs['lr_critic'] = config.lr_critic
    # sac_kwargs['lr_alpha'] = config.lr_alpha
    # sac_kwargs['discount'] = config.discount
    
    train_horizon = config.train_horizon
    n_model = config.n_model
    # n_horizon = config.n_horizon
    roll_horizon = config.roll_horizon
    
    train_rng, rng = jax.random.split(rng, 2)


    # horizon = 5 
    train_summaries = []

    print("action space dimension: ", true_dynamics.model.action_space.shape)
    for roll_horizon in roll_horizons:
        policy_optimizer = SACOptimizer(
            dynamics_model_list=dynamics_model_list,
            horizon=roll_horizon,
            action_dim=true_dynamics.model.action_space.shape,
            train_steps_per_model_update=50,
            transitions_per_update=8000,
            sac_kwargs=sac_kwargs,
            reset_actor_params=False,
        )

        # policy_optimizer.dynamics_model_list[0].reward_model.cost_weights = jnp.array([config.cost_theta, config.cost_theta_dot, config.cost_u])#[0] = config.cost_theta
        # # policy_optimizer.dynamics_model_list[0].reward_model.cost_weights#[1] = config.cost_theta_dot
        # policy_optimizer.dynamics_model_list[0].reward_model.ctrl_cost_weight = config.cost_u
        # policy_optimizer.dynamics_model_list[0]._init_fn()
        
        # for run in range(5):
        t = time.time()
        train_summary = policy_optimizer.train(
            rng=train_rng,
            buffer=buffer,
        )

        print("time taken for optimization ", time.time() - t)
        # for j in range(len(dynamics_model_list)):
        #     actor_rng, rng = jax.random.split(rng, 2)
        #     obs = true_dynamics.model.obs_space.sample()
        #     time_stamps = []
        #     for i in range(200):
        #         start_time = time.time()
        #         action = policy_optimizer.get_action_for_eval(obs=obs, rng=rng, agent_idx=j)
        #         action = true_dynamics.rescale_action(action) # rescale action
        #         time_taken = time.time() - start_time
        #         if i == 0:
        #             print("Time taken", time_taken)
        #         else:
        #             time_stamps.append(time_taken)
        #         obs = true_dynamics.model.predict(obs, action)
        #         # obs, reward, terminate, truncate, info = true_dynamics.model.predict(obs, action)

        #     time_stamps = np.asarray(time_taken)
        #     print("avergage time taken", time_stamps.mean())

        with open("/home/bizoffermark/workspace/ode/pyur5/metadata/sac_policy_{}_{}_{}.pkl".format(train_horizon, n_model, roll_horizon), "wb") as f:
            cloudpickle.dump(policy_optimizer, f)
            
    logs = {}
    for n_horizon in n_horizons:
        log = make_plot(train_horizon, n_model, n_horizon, "sac", 0, policy_optimizer)
        logs.update(log)
    return train_summaries, logs


def main():
    wandb.init(project="sac_policy")
    # rollout_rng, rng = jax.random.split(rng, 2)

    _, logs = train_sac_policy_optimizer(wandb.config, rng)
    # log_final = {}
    # for i, roll_horizon in enumerate(roll_horizons):
    #     for key in logs[i].keys():
    #         key_name = "roll_horizon_{}".format(roll_horizon) + "_" + key
    #         log_final[key_name] = logs[i][key]
    wandb.log(logs)
    
    # wandb.log(log)

if __name__ == '__main__':

    #################### Begin Buffer ####################
    train_horizons = [1] #[1,2,3,4,5] #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    n_models = [5]#[1, 5, 10]
    n_horizons = [200]#, 300, 400, 500]#[5, 10, 20, 30, 40, 50, 100, 150, 200]
    roll_horizons = [5]

    with open("/home/bizoffermark/workspace/ode/pyur5/data_pkl/data_{}.pkl".format(1), "rb") as f:
        data = pickle.load(f)

    train_horizon = train_horizons[0]
    n_model = n_models[0]

    #################### Begin Buffer ####################
    true_dynamics = Ur5PendulumDynamicsModel(train_horizon=train_horizon, n_model=n_model)
    dynamics_model_list = [true_dynamics] # learnt dynamics model
    rng = jax.random.PRNGKey(seed=0)

    buffer = ReplayBuffer(
        obs_shape=true_dynamics.model.obs_space.shape,
        action_shape=true_dynamics.model.action_space.shape,
        max_size=1000000,
        normalize=False,
        action_normalize=False,
        learn_deltas=False
    )
    rollout_rng, rng = jax.random.split(rng, 2)
    transitions = rollout_random_policy(train_horizon, true_dynamics, data, rng=rollout_rng)
    buffer.add(transitions)


    # with open("buffer.pkl", "wb") as f:
    #     pickle.dump(buffer, f)
        
    # with open("buffer.pkl", "rb") as f:
    #     buffer = pickle.load(f)
    #################### End Buffer ####################




    sac_kwargs = {
        'discount': 0.99,
        'init_ent_coef': 1.0,
        'lr_actor': 0.0005,
        'weight_decay_actor': 1e-5,
        'lr_critic': 0.0005,
        'weight_decay_critic': 1e-5,
        'lr_alpha': 0.0005,
        'weight_decay_alpha': 0.0,
        'actor_features': [64, 64],
        'critic_features': [256, 256],
        'scale_reward': 1,
        'tune_entropy_coef': True,
        'tau': 0.005,
        'batch_size': 128,
        'train_steps': 1000,
    }

    sweep_config = {
        'method': 'grid',
        "name": "sac_policy",
        "metric": {"name": "cost_200", "goal": "minimize"},

        "parameters": {
            "train_horizon": {
                "values": train_horizons
            },
            "n_model": {
                "values": n_models
            },
            "sac_kwargs": {
                "values": [sac_kwargs]
            },
            "roll_horizon": {
                "values" : roll_horizons
            }
            # # hyperparameter tuning
            # 'lr_actor': {
            #     'distribution': 'uniform',
            #     'min': 1e-5,
            #     'max': 1e-3
            # },
            # 'lr_critic': {
            #     'distribution': 'uniform',
            #     'min': 1e-5,
            #     'max': 1e-3
            # },
            # 'lr_alpha': {
            #     'distribution': 'uniform',
            #     'min': 1e-5,
            #     'max': 1e-3
            # },
            # 'discount': {
            #     'distribution': 'uniform',
            #     'min': 0.9,
            #     'max': 0.999
            # },
            # # cost function tuning
            # 'cost_theta': {
            #     'distribution': 'uniform',
            #     'min': 1e-3,
            #     'max': 1e+3
            # },
            # 'cost_theta_dot': {
            #     'distribution': 'uniform',
            #     'min': 1e-3,
            #     'max': 1e+3
            # },
            # 'cost_u': {
            #     'distribution': 'uniform',
            #     'min': 1e-3,
            #     'max': 1e+2
            # }
            
        }
    }

    headers = ["theta", "theta_dot", "p_ee", "v_ee"]
        
    data_test = data['test']
    x = data_test['x'][:,:-1]
    u = data_test['u']
    y = data_test['y']

    sweep_id = wandb.sweep(sweep_config, project="sac_policy")

    print("sweep_id is {}".format(sweep_id))
    wandb.agent(sweep_id, function=main, count=1000)

