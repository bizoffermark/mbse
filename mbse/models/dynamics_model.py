from mbse.utils.replay_buffer import Transition, identity_transform, inverse_identitiy_transform
from typing import Optional, Callable
from flax import struct
import jax.numpy as jnp


@struct.dataclass
class ModelSummary:
    dynamics_loss: jnp.array = 0.0


class DynamicsModel(object):
    def __init__(self,
                 transforms: Callable = identity_transform,
                 inverse_transforms: Callable = inverse_identitiy_transform,
                 *args,
                 **kwargs
                 ):
        self.transforms = transforms
        self.inverse_transforms = inverse_transforms
        pass

    def predict(self, obs, action, rng=None):
        pass

    def evaluate(self, obs, action, rng=None):
        pass

    def _train_step(self, tran: Transition, val: Optional[Transition] = None):
        return {}

    @property
    def model_params(self):
        return None

    @property
    def model_opt_state(self):
        return None

    def update_model(self, model_params, model_opt_state):
        pass

    def set_transforms(self, transforms: Callable, inverse_transforms: Callable):
        self.transforms = transforms
        self.inverse_transforms = inverse_transforms
