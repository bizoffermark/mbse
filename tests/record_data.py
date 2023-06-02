import numpy as np
from mbse.utils.vec_env.env_util import make_vec_env
from gym.wrappers import RescaleAction, TimeLimit
from mbse.models.environment_models.pendulum_swing_up import CustomPendulumEnv
import pickle
time_limit = 5
n_envs = 1
seed = 0 
env_kwargs = {}
wrapper_cls = lambda x: RescaleAction(
        TimeLimit(x, max_episode_steps=time_limit),
        min_action=-1,
        max_action=1,
    )

env = make_vec_env(env_id=CustomPendulumEnv, wrapper_class=wrapper_cls)#, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs)

num_data_points = 100000
n_steps = 200
obs_vec = np.zeros((num_data_points, env.observation_space.shape[0]))
action_vec = np.zeros((num_data_points, env.action_space.shape[0]))
next_obs_vec = np.zeros((num_data_points, env.observation_space.shape[0]))
reward_vec = np.zeros((num_data_points, 1))
done_vec = np.zeros((num_data_points, 1))

# generate random seed

for j in range(int(num_data_points/n_steps)):
    # obs = env.observation_space.sample()
    print("reset")
    # env.state = obs
    obs, _ = env.reset()
    for i in range(n_steps):
        action = env.action_space.sample()
        next_obs, reward, done, info, _ = env.step(action)
        print("obs: ", obs)
        print("next_obs: ", next_obs)
        obs_vec[i] = obs
        action_vec[i] = action
        reward_vec[i] = reward
        next_obs_vec[i] = next_obs
        done_vec[i] = done
        obs = next_obs
    
data = dict()
sizes = [0.8, 0.1, 0.1]
num_data_points = obs_vec.shape[0]
num_train = int(num_data_points * sizes[0])
num_vld = int(num_data_points * sizes[1])
num_tst = int(num_data_points - num_train - num_vld)

num_points = [num_train, num_vld, num_tst]

headers = ['train', 'valid', 'test']
for i, header in enumerate(headers):
    num_train = num_points[i]
    data[header] = dict()
    data[header]['x'] = np.concatenate([obs_vec[:num_train], action_vec[:num_train]], axis=1)
    data[header]['y'] = next_obs_vec[:num_train] - obs_vec[:num_train]
    # data[header]['y'][:,0] = angle_normalize(data[header]['y'][:,0])
    obs_vec = obs_vec[num_train:]
    action_vec = action_vec[num_train:]
    next_obs_vec = next_obs_vec[num_train:]

with open("/home/honam/workspace/ode/pyur5/data_pkl/sim_data.pkl", "wb") as f:
    pickle.dump(data, f)
# transitions = Transition(
#     obs=obs_vec,
#     action=action_vec,
#     reward=reward_vec,
#     next_obs=next_obs_vec,
#     done=done_vec,
# )

# buffer = ReplayBuffer(
#     obs_shape=env.observation_space.shape,
#     action_shape=env.action_space.shape,
#     max_size=100000,
#     normalize=False,
#     action_normalize=False,
#     learn_deltas=False
# )

# buffer.add(transitions)

# rng = jax.random.PRNGKey(seed=0)
# rollout_rng, rng = jax.random.split(rng, 2)
# transitions = rollout_random_policy(train_horizon, rng=rollout_rng)

# buffer.add(transitions)
# train_rng, rng = jax.random.split(rng, 2)


# # action = policy_optimizer.get_action_for_eval(obs=obs, rng=rng, agent_idx=0)

# # with open("sac_policy.pkl", "wb") as f:
# #     pickle.dump(policy_optimizer, f)

# # for run in range(5):
# train_summary = policy_optimizer.train(
#     rng=train_rng,
#     buffer=buffer,
# )

# print("time taken for optimization ", time.time() - t)
# for j in range(len(dynamics_model_list)):
#     actor_rng, rng = jax.random.split(rng, 2)
#     obs = true_dynamics.model.obs_space.sample()
#     time_stamps = []
#     for i in range(200):
#         start_time = time.time()
#         action = policy_optimizer.get_action_for_eval(obs=obs, rng=rng, agent_idx=j)
#         time_taken = time.time() - start_time
#         if i == 0:
#             print("Time taken", time_taken)
#         else:
#             time_stamps.append(time_taken)
        
#         # obs, reward, terminate, truncate, info = true_dynamics.model.predict(obs, action)

#     time_stamps = np.asarray(time_taken)
#     print("avergage time taken", time_stamps.mean())

