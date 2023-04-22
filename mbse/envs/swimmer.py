from mbse.models.environment_models.swimmer_reward import SwimmerRewardModel
from mbse.envs.dm_control_env import DeepMindBridge
from dm_control.suite.swimmer import Swimmer, _DEFAULT_TIME_LIMIT, get_model_and_assets, Physics
from dm_control.rl.control import Environment
import collections
from dm_control.utils import containers

SUITE = containers.TaggedTasks()


@SUITE.add('benchmarking')
def run(time_limit=_DEFAULT_TIME_LIMIT, n_joints=6, random=None, environment_kwargs=None):
    """Returns the run task."""
    physics = Physics.from_xml_string(*get_model_and_assets(n_joints=n_joints))
    task = CustomSwimmer(random=random)
    environment_kwargs = environment_kwargs or {}
    return Environment(physics, task, time_limit=time_limit,
                       **environment_kwargs)


class CustomSwimmer(Swimmer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_observation(self, physics):
        """Returns an observation of joint angles, body velocities and target."""
        obs = collections.OrderedDict()
        obs['joints'] = physics.joints()
        obs['body_velocities'] = physics.body_velocities()
        obs['to_target'] = physics.nose_to_target()
        return obs


class SwimmerEnvDM(DeepMindBridge):
    def __init__(self, reward_model: SwimmerRewardModel = SwimmerRewardModel(), *args, **kwargs):
        self.reward_model = reward_model
        env = run(time_limit=float('inf'), environment_kwargs={'flat_observation': True})
        super().__init__(env=env, *args, **kwargs)
        self.env = env

    def step(self, action):
        obs, reward, terminate, truncate, info = super().step(action)
        reward = self.reward_model.predict(obs=obs, action=action, next_obs=obs)
        reward = reward.astype(float).item()
        return obs, reward, terminate, truncate, info


if __name__ == "__main__":
    from gym.wrappers.time_limit import TimeLimit
    env = SwimmerEnvDM(reward_model=SwimmerRewardModel())
    env = TimeLimit(env, max_episode_steps=1000)
    obs, _ = env.reset(seed=10)
    for i in range(1999):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
