import jax
from mbse.utils.replay_buffer import Transition, ReplayBuffer
from mbse.utils.vec_env import VecEnv
from typing import Callable, Optional, Union
import wandb
from copy import deepcopy
import numpy as np
from mbse.utils.vec_env import VecEnv
from tqdm import tqdm
from gym.wrappers.record_video import RecordVideo
from gym import Env
import glob
from datetime import datetime


class DummyTrainer(object):
    def __init__(self,
                 env: VecEnv,
                 agent_fn: Callable,
                 buffer_size: int = int(1e6),
                 max_train_steps: int = int(1e6),
                 batch_size: int = 256,
                 train_freq: int = 100,
                 train_steps: int = 100,
                 eval_freq: int = 1000,
                 seed: int = 0,
                 exploration_steps: int = int(1e4),
                 rollout_steps: int = 200,
                 eval_episodes: int = 100,
                 agent_name: str = "DummyAgent",
                 use_wandb: bool = True,
                 validate: bool = False,
                 normalize: bool = False,
                 action_normalize: bool = False,
                 learn_deltas: bool = False,
                 record_test_video: bool = False,
                 video_prefix: str = "",
                 video_folder: str = "./",
                 test_env: Optional[Union[VecEnv, Env]] = None,
                 ):
        self.env = env
        self.num_envs = max(env.num_envs, 1)
        self.buffer = ReplayBuffer(
            obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            max_size=buffer_size,
            normalize=normalize,
            action_normalize=action_normalize,
            learn_deltas=learn_deltas
        )
        self.buffer_size = buffer_size

        self.eval_episodes = eval_episodes
        self.agent_name = agent_name
        self.rng = jax.random.PRNGKey(seed)
        self.use_wandb = use_wandb

        self.max_train_steps = max_train_steps
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.train_steps = int(train_steps*self.num_envs)
        self.eval_freq = eval_freq
        self.exploration_steps = exploration_steps
        self.rollout_steps = rollout_steps
        self.agent = agent_fn(
            self.use_wandb,
            validate,
            self.train_steps,
            self.batch_size,
        )
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        self.video_dir_name = video_folder + 'video' + str(seed) + video_prefix + dt_string
        if record_test_video:
            test_env_wrapper = lambda x: RecordVideo(x, video_folder=self.video_dir_name,
                                                     episode_trigger=lambda x: True)
        else:
            test_env_wrapper = lambda x: x
        if test_env is None:
            self.test_env = [
                test_env_wrapper(deepcopy(self.env.envs[0]))
            ]
        else:
            if isinstance(test_env, list):
                self.test_env = []
                for env in test_env:
                    if isinstance(env, VecEnv):
                        for e in env.envs:
                            self.test_env.append(
                                test_env_wrapper(e)
                            )
                    else:
                        self.test_env.append(
                            test_env_wrapper(env)
                        )
            else:
                if isinstance(test_env, VecEnv):
                    for e in test_env.envs:
                        self.test_env.append(
                            test_env_wrapper(e)
                        )
                else:
                    self.test_env.append(
                        test_env_wrapper(test_env)
                    )
        assert len(self.test_env) == self.num_reward_models, "number of reward models must be the same as number of " \
                                                             "envs"

        self.record_test_video = record_test_video

    def train(self):
        pass

    def save_agent(self, step=0, agent_name=None):
        pass
        # if self.use_wandb:
        #    prefix = str(step)
        #    name = self.agent_name if agent_name is None else agent_name
        #    name = name + "_" + prefix
        #    save_dir = os.path.join(wandb.run.dir, name)
        #    with open(save_dir, 'wb') as outp:
        #        cloudpickle.dump(self.agent, outp)

    def step_env(self, obs, policy, num_steps, rng):
        rng, reset_rng = jax.random.split(rng, 2)
        num_points = int(num_steps*self.num_envs)
        obs_shape = (num_points,) + self.env.observation_space.shape
        action_space = (num_points,) + self.env.action_space.shape
        obs_vec = np.zeros(obs_shape)
        action_vec = np.zeros(action_space)
        reward_vec = np.zeros((num_points,))
        next_obs_vec = np.zeros(obs_shape)
        done_vec = np.zeros((num_points,))
        next_rng = rng
        last_obs = obs
        last_done = False
        for step in range(num_steps):
            next_rng, actor_rng = jax.random.split(next_rng, 2)
            action = policy(obs, actor_rng)
            next_obs, reward, terminate, truncate, info = self.env.step(action)

            obs_vec[step*self.num_envs: (step+1)*self.num_envs] = obs
            action_vec[step*self.num_envs: (step+1)*self.num_envs] = action
            reward_vec[step*self.num_envs: (step+1)*self.num_envs] = reward
            next_obs_vec[step*self.num_envs: (step+1)*self.num_envs] = next_obs
            done_vec[step*self.num_envs: (step+1)*self.num_envs] = terminate
            # obs_vec = obs_vec.at[step].set(jnp.asarray(obs))
            # action_vec = action_vec.at[step].set(jnp.asarray(action))
            # reward_vec = reward_vec.at[step].set(jnp.asarray(reward))
            # next_obs_vec = next_obs_vec.at[step].set(jnp.asarray(next_obs))
            # done_vec = done_vec.at[step].set(jnp.asarray(terminate))

            # for idx, done in enumerate(dones):
            #     if done:
            #         reset_rng, next_reset_rng = jax.random.split(reset_rng, 2)
            #         reset_seed = jax.random.randint(
            #             reset_rng,
            #             (1,),
            #             minval=0,
            #             maxval=num_steps).item()
            #         obs[idx], _ = self.env.reset(seed=reset_seed)
            obs = np.concatenate([x['current_env_state'].reshape(1, -1) for x in info], axis=0)
            dones = np.concatenate([x['last_done'].reshape(1, -1) for x in info], axis=0)

            last_obs = obs
            last_done = dones
        transitions = Transition(
            obs=obs_vec,
            action=action_vec,
            reward=reward_vec,
            next_obs=next_obs_vec,
            done=done_vec,
        )
        return transitions, last_obs, last_done

    def rollout_policy(self, num_steps, policy, rng):
        rng, reset_rng = jax.random.split(rng, 2)
        reset_seed = jax.random.randint(
            reset_rng,
            (1, ),
            minval=0,
            maxval=num_steps).item()
        obs, _ = self.env.reset(seed=reset_seed)
        num_points = int(num_steps * self.num_envs)
        obs_shape = (num_points,) + self.env.observation_space.shape
        action_space = (num_points,) + self.env.action_space.shape
        obs_vec = np.zeros(obs_shape)
        action_vec = np.zeros(action_space)
        reward_vec = np.zeros((num_points,))
        next_obs_vec = np.zeros(obs_shape)
        done_vec = np.zeros((num_points,))
        next_rng = rng
        for step in range(num_steps):
            next_rng, actor_rng = jax.random.split(next_rng, 2)
            action = policy(obs, actor_rng)
            next_obs, reward, terminate, truncate, info = self.env.step(action)

            obs_vec[step * self.num_envs: (step + 1) * self.num_envs] = obs
            action_vec[step * self.num_envs: (step + 1) * self.num_envs] = action
            reward_vec[step * self.num_envs: (step + 1) * self.num_envs] = reward
            next_obs_vec[step * self.num_envs: (step + 1) * self.num_envs] = next_obs
            done_vec[step * self.num_envs: (step + 1) * self.num_envs] = terminate
            # obs_vec = obs_vec.at[step].set(jnp.asarray(obs))
            # action_vec = action_vec.at[step].set(jnp.asarray(action))
            # reward_vec = reward_vec.at[step].set(jnp.asarray(reward))
            # next_obs_vec = next_obs_vec.at[step].set(jnp.asarray(next_obs))
            # done_vec = done_vec.at[step].set(jnp.asarray(terminate))
            obs = np.concatenate([x['current_env_state'].reshape(1, -1) for x in info], axis=0)
            # for idx, done in enumerate(dones):
            #    if done:
            #        reset_rng, next_reset_rng = jax.random.split(reset_rng, 2)
            #        reset_seed = jax.random.randint(
            #            reset_rng,
            #            (1,),
            #            minval=0,
            #            maxval=num_steps).item()
            #        obs[idx], _ = self.env.reset(seed=reset_seed)

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
        obs, _ = self.env.reset(seed=reset_seed)
        return transitions

    def eval_policy(self, step=0, rng=None) -> float:
        reward_log = {

        }
        for i, env in enumerate(self.test_env):
            pbar = tqdm()
            avg_reward = 0.0
            for e in range(self.eval_episodes):
                obs, _ = env.reset(seed=e)
                done = False
                steps = 0
                while not done:
                    action = self.agent.act(obs, rng=rng, eval=True, eval_idx=i)
                    next_obs, reward, terminate, truncate, info = env.step(action)
                    done = terminate or truncate
                    avg_reward += reward
                    obs = next_obs
                    steps += 1
                    pbar.update(1)
                    # print(steps)
                    if done:
                        obs, _ = env.reset(seed=e)
            avg_reward /= self.eval_episodes
            reward_log['reward_task_' + str(i)] = avg_reward
            pbar.close()
            if self.use_wandb and self.record_test_video:
                mp4list = glob.glob(self.video_dir_name + '/*.mp4')
                if len(mp4list) > 0:
                    mp4 = mp4list[-1]
                    # log gameplay video in wandb
                    wandb.log({"gameplays":
                                   wandb.Video(mp4, caption='episode/task: '+str(step) + '/' + str(i),
                                               fps=4, format="gif"),
                               "step": step})
        return reward_log

    @property
    def num_reward_models(self):
        return 1
