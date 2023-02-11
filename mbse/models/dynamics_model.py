from mbse.utils.replay_buffer import Transition, identity_transform, inverse_identitiy_transform
from typing import Optional, Callable, Union
from flax import struct
import jax.numpy as jnp


@struct.dataclass
class ModelSummary:
    model_likelihood: jnp.array = 0.0
    grad_norm: jnp.array = 0.0
    val_logl: jnp.array = 0.0
    val_mse: jnp.array = 0.0

    def dict(self):
        return {
            'model_likelihood': self.model_likelihood.item(),
            'grad_norm': self.grad_norm.item(),
            'val_logl': self.val_logl.item(),
            'val_mse': self.val_mse.item(),
        }


class DynamicsModel(object):
    def __init__(self,
                 bias_obs: Union[jnp.ndarray, float] = 0.0,
                 bias_act: Union[jnp.ndarray, float] = 0.0,
                 bias_out: Union[jnp.ndarray, float] = 0.0,
                 scale_obs: Union[jnp.ndarray, float] = 1.0,
                 scale_act: Union[jnp.ndarray, float] = 1.0,
                 scale_out: Union[jnp.ndarray, float] = 1.0,
                 pred_diff: bool = False,
                 *args,
                 **kwargs
                 ):
        self.bias_obs = bias_obs
        self.bias_act = bias_act
        self.bias_out = bias_out
        self.scale_obs = scale_obs
        self.scale_act = scale_act
        self.scale_out = scale_out
        self.pred_diff = pred_diff
        pass

    def predict(self, obs, action, rng=None):
        pass

    def evaluate(self,
                 parameters,
                 obs,
                 action,
                 rng,
                 sampling_idx=None,
                 bias_obs: Union[jnp.ndarray, float] = 0.0,
                 bias_act: Union[jnp.ndarray, float] = 0.0,
                 bias_out: Union[jnp.ndarray, float] = 0.0,
                 scale_obs: Union[jnp.ndarray, float] = 1.0,
                 scale_act: Union[jnp.ndarray, float] = 1.0,
                 scale_out: Union[jnp.ndarray, float] = 1.0,
                 ):
        pass

    def _train_step(self,
                    tran: Transition,
                    model_params=None,
                    model_opt_state=None,
                    val: Optional[Transition] = None):
        return None, None, ModelSummary()

    @property
    def model_params(self):
        return None

    @property
    def model_opt_state(self):
        return None

    def update_model(self, model_params, model_opt_state):
        pass

    def set_transforms(self,
                       bias_obs: Union[jnp.ndarray, float] = 0.0,
                       bias_act: Union[jnp.ndarray, float] = 0.0,
                       bias_out: Union[jnp.ndarray, float] = 0.0,
                       scale_obs: Union[jnp.ndarray, float] = 1.0,
                       scale_act: Union[jnp.ndarray, float] = 1.0,
                       scale_out: Union[jnp.ndarray, float] = 1.0,
                       ):
        self.bias_obs = bias_obs
        self.bias_act = bias_act
        self.bias_out = bias_out
        self.scale_obs = scale_obs
        self.scale_act = scale_act
        self.scale_out = scale_out
