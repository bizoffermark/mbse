from mbse.utils.replay_buffer import Transition


class RewardModel(object):
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, obs, action, next_obs, rng=None):
        pass

    def train_step(self, tran: Transition):
        pass
