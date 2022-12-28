from typing import Sequence, Callable
import numpy as np
import gym
from flax import linen as nn
from mbse.utils.models import fSVGDEnsemble, ProbabilisticEnsembleModel
import jax
import jax.numpy as jnp
from mbse.utils.utils import sample_normal_dist
from flax import struct
from mbse.utils.replay_buffer import Transition
from mbse.models.dynamics_model import DynamicsModel


@struct.dataclass
class BayesianDynamicsModelSummary:
    model_likelihood: jnp.ndarray
    grad_norm: jnp.ndarray

    def dict(self):
        return {
            'model_likelihood': self.model_likelihood,
            'grad_norm': self.grad_norm,
        }


class BayesianDynamicsModel(DynamicsModel):

    def __init__(self,
                 action_space: gym.spaces.box,
                 observation_space: gym.spaces.box,
                 model_class: str = "ProbabilisticEnsembleModel",
                 num_ensemble: int = 10,
                 features: Sequence[int] = [256, 256],
                 non_linearity: Callable = nn.swish,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 sig_min: float = 1e-3,
                 sig_max: float = 1e3,
                 seed: int = 0,
                 *args,
                 **kwargs
                 ):
        
        super(BayesianDynamicsModel, self).__init__()
        if model_class == "ProbabilisticEnsembleModel":
            model_cls = ProbabilisticEnsembleModel
        elif model_class == "fSVGDEnsemble":
            model_cls = fSVGDEnsemble

        else:
            assert False, "Model class must be ProbabilisticEnsembleModel or fSVGDEnsemble."

        obs_dim = np.prod(observation_space.shape)
        sample_obs = observation_space.sample()
        sample_act = action_space.sample()
        obs_action = jnp.concatenate([sample_obs, sample_act], axis=-1)
        self.model = model_cls(
            example_input=obs_action,
            num_ensemble=num_ensemble,
            features=features,
            output_dim=obs_dim,
            non_linearity=non_linearity,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
            sig_min=sig_min,
            sig_max=sig_max,
        )

    def predict(self, obs, action, rng=None):
        obs_action = jnp.concatenate([obs, action], axis=-1)
        next_obs_tot = self.model.predict(obs_action)
        if rng is not None:
            model_rng, sample_rng = jax.random.split(rng, 2)
            model_idx = jax.random.randint(model_rng, maxval=self.model.num_ensembles)
            next_obs = next_obs_tot[model_idx, ...]
            mean, sig = jnp.split(next_obs, 2, axis=-1)
            next_obs = sample_normal_dist(mean, sig, sample_rng)

        else:
            mean, _ = jnp.split(next_obs_tot, 2, axis=-1)
            next_obs = jnp.mean(mean, axis=0)

        return next_obs

    def train_step(self, tran: Transition):
        x = jnp.concatenate([tran.obs, tran.action], axis=-1)
        likelihood, grad_norm = self.model.train_step(x=x, y=tran.next_obs)
        summary = BayesianDynamicsModelSummary(
            model_likelihood=likelihood,
            grad_norm=grad_norm,
        )
        return summary.dict()
