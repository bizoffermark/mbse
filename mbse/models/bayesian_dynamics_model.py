from typing import Sequence, Callable, Optional, Union
import numpy as np
import gym
from flax import linen as nn
from mbse.utils.models import FSVGDEnsemble, ProbabilisticEnsembleModel
import jax
import jax.numpy as jnp
from mbse.utils.utils import sample_normal_dist
from mbse.utils.replay_buffer import Transition
from mbse.models.dynamics_model import DynamicsModel, ModelSummary
from mbse.models.reward_model import RewardModel
from typing import List
from mbse.utils.utils import gaussian_log_likelihood
from mbse.utils.network_utils import mse



class SamplingType:
    name: str = 'TS1'
    name_types: List[str] = \
        ['All', 'DS', 'TS1', 'TSInf', 'mean']

    def set_type(self, name):
        assert name not in self.name_types, \
            'name must be in ' + ' '.join(map(str, self.name_types))

        self.name = name


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


class BayesianDynamicsModel(DynamicsModel):

    def __init__(self,
                 action_space: gym.spaces.box,
                 observation_space: gym.spaces.box,
                 reward_model: RewardModel,
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
        
        super(BayesianDynamicsModel, self).__init__(*args, **kwargs)
        self.reward_model = reward_model
        if model_class == "ProbabilisticEnsembleModel":
            model_cls = ProbabilisticEnsembleModel
        elif model_class == "fSVGDEnsemble":
            model_cls = FSVGDEnsemble

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
        self.obs_dim = obs_dim

        def predict(parameters,
                     obs,
                     action,
                     rng, sampling_idx=self.sampling_idx,
                     bias_obs: Union[jnp.ndarray, float] = 0.0,
                     bias_act: Union[jnp.ndarray, float] = 0.0,
                     bias_out: Union[jnp.ndarray, float] = 0.0,
                     scale_obs: Union[jnp.ndarray, float] = 1.0,
                     scale_act: Union[jnp.ndarray, float] = 1.0,
                     scale_out: Union[jnp.ndarray, float] = 1.0,
                     ):
            return self._predict(
                predict_fn=self.model._predict,
                parameters=parameters,
                obs=obs,
                action=action,
                rng=rng,
                sampling_type=self.sampling_type,
                num_ensembles=self.model.num_ensembles,
                sampling_idx=sampling_idx,
                batch_size=obs.shape[0],
                bias_obs=bias_obs,
                bias_act=bias_act,
                bias_out=bias_out,
                scale_obs=scale_obs,
                scale_act=scale_act,
                scale_out=scale_out,
                pred_diff=self.pred_diff,
            )
        self.predict = jax.jit(predict)

        def evaluate(
                parameters,
                obs,
                action,
                rng,
                sampling_idx=self.sampling_idx,
                bias_obs: Union[jnp.ndarray, float] = 0.0,
                bias_act: Union[jnp.ndarray, float] = 0.0,
                bias_out: Union[jnp.ndarray, float] = 0.0,
                scale_obs: Union[jnp.ndarray, float] = 1.0,
                scale_act: Union[jnp.ndarray, float] = 1.0,
                scale_out: Union[jnp.ndarray, float] = 1.0,
        ):
            return self._evaluate(
                    pred_fn=self.predict,
                    reward_fn=self.reward_model.predict,
                    parameters=parameters,
                    obs=obs,
                    action=action,
                    rng=rng,
                    sampling_idx=sampling_idx,
                    bias_obs=bias_obs,
                    bias_act=bias_act,
                    bias_out=bias_out,
                    scale_obs=scale_obs,
                    scale_act=scale_act,
                    scale_out=scale_out,
            )
        self.evaluate = jax.jit(evaluate)

        def _train_step(
                    tran: Transition,
                    model_params,
                    model_opt_state,
                    val: Optional[Transition] = None
        ):

            return self._train(
                train_fn=self.model._train_step,
                predict_fn=self.model._predict,
                tran=tran,
                model_params=model_params,
                model_opt_state=model_opt_state,
                val=val,
            )

        self._train_step = jax.jit(_train_step)

    def set_sampling_type(self, name):
        self.sampling_type.set_type(name)

    def set_sampling_idx(self, idx):
        self.sampling_idx = jnp.clip(
            idx,
            0,
            self.model.num_ensembles
        )

    @staticmethod
    def _predict(predict_fn,
                 parameters,
                 obs,
                 action,
                 rng,
                 sampling_type,
                 num_ensembles,
                 sampling_idx,
                 batch_size,
                 bias_obs: Union[jnp.ndarray, float] = 0.0,
                 bias_act: Union[jnp.ndarray, float] = 0.0,
                 bias_out: Union[jnp.ndarray, float] = 0.0,
                 scale_obs: Union[jnp.ndarray, float] = 1.0,
                 scale_act: Union[jnp.ndarray, float] = 1.0,
                 scale_out: Union[jnp.ndarray, float] = 1.0,
                 pred_diff: bool = 1,
                 ):
        """
                Predict using learning model
                :param obs: observation, shape (batch, dim_state)
                :param action: action, shape (batch, dim_action)
                :param rng:
                :return:

                """
        transformed_obs = (obs - bias_obs)/scale_obs
        transformed_act = (action - bias_act)/scale_act
        obs_action = jnp.concatenate([transformed_obs, transformed_act], axis=-1)
        next_obs_tot = predict_fn(parameters, obs_action)
        next_obs = next_obs_tot

        sampling_scheme = 'mean' if rng is None \
            else sampling_type.name

        if sampling_scheme == 'mean':
            mean, _ = jnp.split(next_obs_tot, 2, axis=-1)
            next_obs = jnp.mean(mean, axis=0)

        elif sampling_scheme == 'TS1':
            model_rng, sample_rng = jax.random.split(rng, 2)
            model_idx = jax.random.randint(
                model_rng,
                shape=(batch_size,),
                minval=0,
                maxval=num_ensembles)

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
            assert sampling_idx.shape[0] == batch_size, \
                'Set sampling indexes size to be particle size'
            sample_rng = jax.random.split(
                rng,
                batch_size
            )
            next_obs = jax.vmap(sample, in_axes=(1, 1, 1), out_axes=1)(
                next_obs_tot,
                sampling_idx,
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
        next_obs = next_obs*scale_out + bias_out + pred_diff * obs
        return next_obs

    # def predict(self, obs, action, rng=None):
    #     """
    #     Predict using learning model
    #     :param obs: observation, shape (batch, dim_state)
    #     :param action: action, shape (batch, dim_action)
    #     :param rng:
    #     :return:
    #
    #     """
    #     obs_action = jnp.concatenate([obs, action], axis=-1)
    #     next_obs_tot = self.model.predict(obs_action)
    #     batch_size = obs.shape[0]
    #     next_obs = next_obs_tot
    #
    #     sampling_scheme = 'mean' if rng is None \
    #         else self.sampling_type.name
    #
    #     if sampling_scheme == 'mean':
    #         mean, _ = jnp.split(next_obs_tot, 2, axis=-1)
    #         next_obs = jnp.mean(mean, axis=0)
    #
    #     elif sampling_scheme == 'TS1':
    #         model_rng, sample_rng = jax.random.split(rng, 2)
    #         model_idx = jax.random.randint(
    #             model_rng,
    #             shape=(batch_size, ),
    #             minval=0,
    #             maxval=self.model.num_ensembles)
    #
    #         sample_rng = jax.random.split(
    #             sample_rng,
    #             batch_size
    #         )
    #
    #         next_obs = jax.vmap(sample, in_axes=(1, 0, 0), out_axes=0)(
    #             next_obs_tot,
    #             model_idx,
    #             sample_rng
    #         )
    #     elif sampling_scheme == 'TSInf':
    #         assert self.sampling_idx.shape[0] == batch_size, \
    #             'Set sampling indexes size to be particle size'
    #         sample_rng = jax.random.split(
    #             rng,
    #             batch_size
    #         )
    #         next_obs = jax.vmap(sample, in_axes=(1, 1, 1), out_axes=1)(
    #             next_obs_tot,
    #             self.sampling_idx,
    #             sample_rng
    #         )
    #
    #     elif sampling_scheme == 'DS':
    #         mean, std = jnp.split(next_obs_tot, 2, axis=-1)
    #         obs_mean = jnp.mean(mean, axis=0)
    #         al_var = jnp.mean(jnp.square(std), axis=0)
    #         ep_var = jnp.var(mean, axis=0)
    #         obs_var = al_var + ep_var
    #         obs_std = jnp.sqrt(obs_var)
    #         next_obs = sample_normal_dist(
    #             obs_mean,
    #             obs_std,
    #             rng,
    #         )
    #
    #     return next_obs

    @staticmethod
    def _train(train_fn, predict_fn, tran: Transition, model_params, model_opt_state, val: Transition = None):

        x = jnp.concatenate([tran.obs, tran.action], axis=-1)
        new_model_params, new_model_opt_state, likelihood, grad_norm = train_fn(
            params=model_params,
            opt_state=model_opt_state,
            x=x,
            y=tran.next_obs,
        )
        val_logl = jnp.zeros_like(likelihood)
        val_mse = jnp.zeros_like(likelihood)
        if val is not None:
            val_x = jnp.concatenate([val.obs, val.action], axis=-1)
            val_y = val.next_obs
            y_pred = predict_fn(new_model_params, val_x)
            val_likelihood = jax.vmap(
                gaussian_log_likelihood,
                in_axes=(None, 0, 0),
                out_axes=0
            )
            mean, std = jnp.split(y_pred, 2, axis=-1)
            logl = val_likelihood(val_y, mean, std)
            val_logl = logl.mean()
            val_mse = jax.vmap(
                lambda pred: mse(val_y, pred),
            )(mean)
            val_mse = val_mse.mean()
        summary = ModelSummary(
            model_likelihood=likelihood.astype(float),
            grad_norm=grad_norm.astype(float),
            val_logl=val_logl.astype(float),
            val_mse=val_mse.astype(float),
        )

        return new_model_params, new_model_opt_state, summary

    @property
    def model_params(self):
        return self.model.particles

    @property
    def model_opt_state(self):
        return self.model.opt_state

    def update_model(self, model_params, model_opt_state):
        self.model.particles = model_params
        self.model.opt_state = model_opt_state

    @staticmethod
    def _evaluate(
            pred_fn,
            reward_fn,
            parameters,
            obs,
            action,
            rng,
            sampling_idx,
            bias_obs: Union[jnp.ndarray, float] = 0.0,
            bias_act: Union[jnp.ndarray, float] = 0.0,
            bias_out: Union[jnp.ndarray, float] = 0.0,
            scale_obs: Union[jnp.ndarray, float] = 1.0,
            scale_act: Union[jnp.ndarray, float] = 1.0,
            scale_out: Union[jnp.ndarray, float] = 1.0,
    ):
        model_rng = None
        reward_rng = None
        if rng is not None:
            rng, model_rng = jax.random.split(rng, 2)
            rng, reward_rng = jax.random.split(rng, 2)
        next_obs = pred_fn(
                     parameters=parameters,
                     obs=obs,
                     action=action,
                     rng=model_rng,
                     sampling_idx=sampling_idx,
                     bias_obs=bias_obs,
                     bias_act=bias_act,
                     bias_out=bias_out,
                     scale_obs=scale_obs,
                     scale_act=scale_act,
                     scale_out=scale_out
        )
        #transformed_obs, transformed_action, _ = self.transforms(obs, action)
        #transformed_next_obs = self.predict(transformed_obs, transformed_action, model_rng)
        #_, _, next_obs = self.inverse_transforms(transformed_obs=transformed_obs, transformed_action=None,
        #                                         transformed_next_state=transformed_next_obs)
        reward = reward_fn(obs, action, next_obs, reward_rng)
        return next_obs, reward


