from mbse.utils.replay_buffer import Transition
from typing import Optional, Union
from flax import struct
import jax.numpy as jnp
from mbse.utils.utils import convert_to_jax
from mbse.utils.type_aliases import ModelProperties


@struct.dataclass
class ModelSummary:
    model_likelihood: jnp.array = 0.0
    grad_norm: jnp.array = 0.0
    val_logl: jnp.array = 0.0
    val_mse: jnp.array = 0.0
    val_al_std: jnp.array = 0.0
    val_eps_std: jnp.array = 0.0
    calibration_alpha: jnp.array = 0.0
    calibration_error: jnp.array = 0.0

    def dict(self):
        return {
            'model_likelihood': self.model_likelihood.item(),
            'grad_norm': self.grad_norm.item(),
            'val_logl': self.val_logl.item(),
            'val_mse': self.val_mse.item(),
            'val_al_std': self.val_al_std.item(),
            'val_eps_std': self.val_eps_std.item(),
            'calibration_alpha': self.calibration_alpha.item(),
            'calibration_error': self.calibration_error.item(),
        }


class DynamicsModel(object):
    def __init__(self,
                 model_props: ModelProperties = ModelProperties(),
                 pred_diff: bool = False,
                 *args,
                 **kwargs
                 ):
        self.model_props = model_props
        self.pred_diff = pred_diff
        self.evaluate_for_exploration = self.evaluate
        pass

    def _init_fn(self):
        pass

    def predict(self, obs, action, rng=None):
        pass

    def predict_raw(self,
                    parameters,
                    tran: Transition,
                    model_props: ModelProperties = ModelProperties()
                    ):
        pass

    def evaluate(self,
                 parameters,
                 obs,
                 action,
                 rng,
                 sampling_idx=None,
                 model_props: ModelProperties = ModelProperties()
                 ):
        pass

    def _train_step(self,
                    tran: Transition,
                    model_params=None,
                    model_opt_state=None,
                    val: Optional[Transition] = None):
        return None, None, None, ModelSummary()

    @property
    def model_params(self):
        return None

    @property
    def model_opt_state(self):
        return None

    @property
    def init_model_params(self):
        return None

    @property
    def init_model_opt_state(self):
        return None

    def update_model(self, model_params, model_opt_state, alpha):
        alpha = convert_to_jax(alpha)
        bias_obs = self.model_props.bias_obs
        bias_act = self.model_props.bias_act
        bias_out = self.model_props.bias_out
        scale_obs = self.model_props.scale_obs
        scale_act = self.model_props.scale_act
        scale_out = self.model_props.scale_out
        self.model_props = ModelProperties(
            bias_obs=bias_obs,
            bias_act=bias_act,
            bias_out=bias_out,
            scale_obs=scale_obs,
            scale_act=scale_act,
            scale_out=scale_out,
            alpha=alpha,
        )
        pass

    def set_transforms(self,
                       bias_obs: Union[jnp.ndarray, float] = 0.0,
                       bias_act: Union[jnp.ndarray, float] = 0.0,
                       bias_out: Union[jnp.ndarray, float] = 0.0,
                       scale_obs: Union[jnp.ndarray, float] = 1.0,
                       scale_act: Union[jnp.ndarray, float] = 1.0,
                       scale_out: Union[jnp.ndarray, float] = 1.0,
                       ):
        alpha = self.model_props.alpha
        bias_obs = convert_to_jax(bias_obs)
        bias_act = convert_to_jax(bias_act)
        bias_out = convert_to_jax(bias_out)
        scale_obs = convert_to_jax(scale_obs)
        scale_act = convert_to_jax(scale_act)
        scale_out = convert_to_jax(scale_out)
        self.model_props = ModelProperties(
            bias_obs=bias_obs,
            bias_act=bias_act,
            bias_out=bias_out,
            scale_obs=scale_obs,
            scale_act=scale_act,
            scale_out=scale_out,
            alpha=alpha,
        )
