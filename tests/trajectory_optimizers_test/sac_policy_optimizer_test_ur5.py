import numpy as np

import sys
sys.path.append('/home/honam/workspace/ode/pyur5/include/mbse')

from mbse.models.environment_models.pendulum_swing_up import PendulumDynamicsModel, Ur5PendulumDynamicsModel
from mbse.optimizers.sac_based_optimizer import SACOptimizer
import pickle
import jax 
import jax.numpy as jnp

from gym.wrappers.time_limit import TimeLimit
from gym.wrappers.rescale_action import RescaleAction
from mbse.utils.replay_buffer import ReplayBuffer, Transition
from mbse.utils.vec_env.env_util import make_vec_env
import time
import cloudpickle

def rollout_random_policy(true_dynamics, num_steps, rng):
    model = true_dynamics.model
    rng, reset_rng = jax.random.split(rng, 2)
    reset_seed = jax.random.randint(
        reset_rng,
        (1,),
        minval=0,
        maxval=num_steps).item()
    obs = model.obs_space.sample()
    num_points = num_steps
    obs_shape = (num_points,) + model.obs_space.shape
    action_space = (num_points,) + model.action_space.shape
    obs_vec = np.zeros(obs_shape)
    action_vec = np.zeros(action_space)
    reward_vec = np.zeros((num_points,))
    next_obs_vec = np.zeros(obs_shape)
    done_vec = np.zeros((num_points,))
    next_rng = rng
    
    for step in range(num_steps):
        next_rng, actor_rng = jax.random.split(next_rng, 2)
        action = model.action_space.sample()
        print("action at step {} is {}".format(step, action))
        next_obs, reward = true_dynamics.evaluate(obs, action)
        terminate = jnp.array([False])
        obs = model.obs_space.sample()
        print("next_obs at step {} is {}".format(step, next_obs))
        print("reward at step {} is {}".format(step, reward))
        obs_vec[step: (step + 1)] = obs
        action_vec[step: (step + 1)] = action
        reward_vec[step: (step + 1)] = reward.reshape(-1)
        next_obs_vec[step: (step + 1)] = next_obs
        done_vec[step: (step + 1)] = terminate.reshape(-1)


    transitions = Transition(
        obs=obs_vec,
        action=action_vec,
        reward=reward_vec,
        next_obs=next_obs_vec,
        done=done_vec,
    )
    reset_seed = jax.random.randint(
        reset_rng,
        (1,),
        minval=0,
        maxval=num_steps).item()

    # reset 
    obs = model.obs_space.sample()
    return transitions


wrapper_cls = lambda x: RescaleAction(
    TimeLimit(x, max_episode_steps=200),
    min_action=-1,
    max_action=1,
)

true_dynamics = Ur5PendulumDynamicsModel()
dynamics_model_list = [true_dynamics]

horizon = 20


sac_kwargs = {
    'discount': 0.99,
    'init_ent_coef': 1.0,
    'lr_actor': 0.001,
    'weight_decay_actor': 1e-5,
    'lr_critic': 0.001,
    'weight_decay_critic': 1e-5,
    'lr_alpha': 0.0005,
    'weight_decay_alpha': 0.0,
    'actor_features': [64, 64],
    'critic_features': [256, 256],
    'scale_reward': 1,
    'tune_entropy_coef': True,
    'tau': 0.005,
    'batch_size': 128,
    'train_steps': 350,
}


print("action space dimension: ", true_dynamics.model.action_space.shape)
policy_optimizer = SACOptimizer(
    dynamics_model_list=dynamics_model_list,
    horizon=horizon,
    action_dim=true_dynamics.model.action_space.shape,
    train_steps_per_model_update=10,
    transitions_per_update=200,
    sac_kwargs=sac_kwargs,
    reset_actor_params=False,
)

buffer = ReplayBuffer(
    obs_shape=true_dynamics.model.obs_space.shape,
    action_shape=true_dynamics.model.action_space.shape,
    max_size=100000,
    normalize=False,
    action_normalize=False,
    learn_deltas=False
)

rng = jax.random.PRNGKey(seed=0)
rollout_rng, rng = jax.random.split(rng, 2)
transitions = rollout_random_policy(true_dynamics, num_steps=10000, rng=rollout_rng)

buffer.add(transitions)
train_rng, rng = jax.random.split(rng, 2)


# action = policy_optimizer.get_action_for_eval(obs=obs, rng=rng, agent_idx=0)

# with open("sac_policy.pkl", "wb") as f:
#     pickle.dump(policy_optimizer, f)

for run in range(5):
    t = time.time()
    train_summary = policy_optimizer.train(
        rng=train_rng,
        buffer=buffer,
    )
    print("time taken for optimization ", time.time() - t)
    for j in range(len(dynamics_model_list)):
        actor_rng, rng = jax.random.split(rng, 2)
        obs = true_dynamics.model.obs_space.sample()
        time_stamps = []
        for i in range(200):
            start_time = time.time()
            action = policy_optimizer.get_action_for_eval(obs=obs, rng=rng, agent_idx=j)
            time_taken = time.time() - start_time
            if i == 0:
                print("Time taken", time_taken)
            else:
                time_stamps.append(time_taken)
            
            # obs, reward, terminate, truncate, info = true_dynamics.model.predict(obs, action)

        time_stamps = np.asarray(time_taken)
        print("avergage time taken", time_stamps.mean())

with open("sac_policy.pkl", "wb") as f:
    cloudpickle.dump(policy_optimizer, f)
# action = policy_optimizer.get_action_for_eval(obs=obs, rng=rng, agent_idx=0)
