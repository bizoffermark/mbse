""" Taken from:  https://github.com/facebookresearch/mbrl-lib/blob/main/mbrl/env/pets_halfcheetah.py"""

from copy import deepcopy
import numpy as np
from mbse.models.environment_models.halfcheetah_reward_model import HalfCheetahReward
from mbse.envs.dm_control_env import DeepMindBridge
from dm_control.suite.cheetah import Cheetah, _DEFAULT_TIME_LIMIT, get_model_and_assets, Physics
from dm_control.rl.control import Environment
import collections
from dm_control.utils import containers

SUITE = containers.TaggedTasks()

@SUITE.add('benchmarking')
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = PetsCheetah(random=random)
  environment_kwargs = environment_kwargs or {}
  return Environment(physics, task, time_limit=time_limit,
                             **environment_kwargs)


class PetsCheetah(Cheetah):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_observation(self, physics):
        """Returns an observation of the state, ignoring horizontal position."""
        obs = collections.OrderedDict()
        # Ignores horizontal position to maintain translational invariance.
        obs['position'] = np.concatenate([np.asarray(physics.speed()).reshape(-1), physics.data.qpos[1:].copy()])
        obs['velocity'] = physics.velocity()
        return obs

    def sample_state(self, physics, factor=5.0, vel_noise_factor=10):
        """Sets the state of the environment at the start of each episode."""
        # The indexing below assumes that all joints have a single DOF.
        assert physics.model.nq == physics.model.njnt
        lower, upper = physics.model.jnt_range.T * factor
        upper_v = np.ones_like(lower) * vel_noise_factor
        lower_v = - upper_v
        physics.data.qpos = self.random.uniform(lower, upper)
        physics.data.qvel = self.random.uniform(upper_v, lower_v)

        # Stabilize the model before the actual simulation.
        physics.step(nstep=200)

        physics.data.time = 0
        super().initialize_episode(physics)


class HalfCheetahEnv(DeepMindBridge):
    def __init__(self, reward_model: HalfCheetahReward, *args, **kwargs):
        self.prev_qpos = None
        self.reward_model = reward_model
        env = run(time_limit=float('inf'), environment_kwargs={'flat_observation': True})
        super().__init__(env=env, *args, **kwargs)
        self.env = env
        self.observation_space

    def step(self, action):
        obs, reward, terminate, truncate, info = super().step(action)
        reward = self.reward_model.predict(obs=obs,action=action, next_obs=obs)
        return obs, reward, terminate, truncate, info

    def sample_obs(self):
        physics = deepcopy(self.env.physics)
        task = deepcopy(self.env._task)
        return task.get_observation(physics)


if __name__ == "__main__":
    from gym.wrappers.record_video import RecordVideo
    env = HalfCheetahEnv(reward_model=HalfCheetahReward(), render_mode="rgb_array")
    env = RecordVideo(env, video_folder='./cheetah/', episode_trigger=lambda x: True)
    obs, _ = env.reset()
    for i in range(1000):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.close()

