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
from typing import List
from mbse.utils.utils import gaussian_log_likelihood
from mbse.utils.network_utils import mse


@struct.dataclass
class BayesianDynamicsModelSummary:
    model_likelihood: float
    grad_norm: float
    val_logl: float
    val_mse: float

    def dict(self):
        return {
            'model_likelihood': self.model_likelihood,
            'grad_norm': self.grad_norm,
            'val_logl': self.val_logl,
            'val_mse': self.val_mse,
        }


class SamplingType:
    name: str = 'TS1'
    name_types: List[str] = \
        ['All', 'DS', 'TS1', 'TSInf', 'mean']

    def set_type(self, name):
        assert name not in self.name_types, \
            'name must be in ' + ' '.join(map(str, self.name_types))

        self.name = name


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
        self.sampling_type = SamplingType
        self.sampling_idx = jnp.zeros(1)

    def set_sampling_type(self, name):
        self.sampling_type.set_type(name)

    def set_sampling_idx(self, idx):
        self.sampling_idx = jnp.clip(
            idx,
            0,
            self.model.num_ensembles
        )

    def predict(self, obs, action, rng=None):
        """
        Predict using learning model
        :param obs: observation, shape (batch, dim_state)
        :param action: action, shape (batch, dim_action)
        :param rng:
        :return:

        """
        obs_action = jnp.concatenate([obs, action], axis=-1)
        next_obs_tot = self.model.predict(obs_action)
        batch_size = obs.shape[0]
        next_obs = next_obs_tot

        sampling_scheme = 'mean' if rng is None \
            else self.sampling_type.name
        @jax.jit
        def sample(predictions, idx, s_rng):
            """
            :param predictions: (Ne, n_state, 2)
            :param idx: (1, )
            :return:
            """
            pred = predictions[idx]
            mu, sig = jnp.split(pred,
                                  2,
                                  axis=-1
                                  )

            sampled_obs = sample_normal_dist(
                mu,
                sig,
                s_rng
            )
            return sampled_obs
        if sampling_scheme == 'mean':
            mean, _ = jnp.split(next_obs_tot, 2, axis=-1)
            next_obs = jnp.mean(mean, axis=0)

        elif sampling_scheme == 'TS1':
            model_rng, sample_rng = jax.random.split(rng, 2)
            model_idx = jax.random.randint(
                model_rng,
                shape=(batch_size, ),
                minval=0,
                maxval=self.model.num_ensembles)

            sample_rng = jax.random.split(
                sample_rng,
                batch_size
            )

            next_obs = jax.vmap(sample, in_axes=(1, 0, 0), out_axes=0)(
                next_obs_tot,
                model_idx,
                sample_rng
            )
        elif sampling_scheme == 'TSInf':
            assert self.sampling_idx.shape[0] == batch_size, \
                'Set sampling indexes size to be particle size'
            sample_rng = jax.random.split(
                rng,
                batch_size
            )
            next_obs = jax.vmap(sample, in_axes=(1, 1, 1), out_axes=1)(
                next_obs_tot,
                self.sampling_idx,
                sample_rng
            )

        elif sampling_scheme == 'DS':
            mean, std = jnp.split(next_obs_tot, 2, axis=-1)
            obs_mean = jnp.mean(mean, axis=0)
            al_var = jnp.mean(jnp.square(std), axis=0)
            ep_var = jnp.var(mean, axis=0)
            obs_var = al_var + ep_var
            obs_std = jnp.sqrt(obs_var)
            next_obs = sample_normal_dist(
                obs_mean,
                obs_std,
                rng,
            )

        return next_obs

    def train_step(self, tran: Transition, val: Transition = None):
        x = jnp.concatenate([tran.obs, tran.action], axis=-1)
        val_logl = 0
        val_mse = 0
        likelihood, grad_norm = self.model.train_step(
            x=x,
            y=tran.next_obs,
        )
        if val is not None:
            val_x = jnp.concatenate([val.obs, val.action], axis=-1)
            val_y = val.next_obs
            y_pred = self.model.predict(val_x)
            val_likelihood = jax.vmap(
                gaussian_log_likelihood,
                in_axes=(None, 0, 0),
                out_axes=0
            )
            mean, std = jnp.split(y_pred, 2, axis=-1)
            logl = val_likelihood(val_y, mean, std)
            val_logl = logl.mean().item()
            val_mse = jax.vmap(
                lambda pred: mse(val_y, pred),
            )(mean)
            val_mse = val_mse.mean().item()

        summary = BayesianDynamicsModelSummary(
            model_likelihood=likelihood.item(),
            grad_norm=grad_norm.item(),
            val_logl=val_logl,
            val_mse=val_mse,
        )
        return summary.dict()
