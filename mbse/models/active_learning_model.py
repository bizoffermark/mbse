from typing import Optional, Union
import jax
import jax.numpy as jnp
from mbse.utils.utils import sample_normal_dist
from mbse.models.hucrl_model import HUCRLModel
from mbse.models.bayesian_dynamics_model import BayesianDynamicsModel, sample


def evaluate_for_exploration(
        pred_fn,
        parameters,
        obs,
        action,
        rng,
        alpha: Union[jnp.ndarray, float] = 1.0,
        bias_obs: Union[jnp.ndarray, float] = 0.0,
        bias_act: Union[jnp.ndarray, float] = 0.0,
        bias_out: Union[jnp.ndarray, float] = 0.0,
        scale_obs: Union[jnp.ndarray, float] = 1.0,
        scale_act: Union[jnp.ndarray, float] = 1.0,
        scale_out: Union[jnp.ndarray, float] = 1.0,
        sampling_idx: Optional[int] = None,
        use_log_uncertainties: bool = False,
        use_al_uncertainties: bool = False,
):
    model_rng = None
    if rng is not None:
        rng, model_rng = jax.random.split(rng, 2)
    next_obs, eps_uncertainty, al_uncertainty = pred_fn(
        parameters=parameters,
        obs=obs,
        action=action,
        rng=model_rng,
        alpha=alpha,
        bias_obs=bias_obs,
        bias_act=bias_act,
        bias_out=bias_out,
        scale_obs=scale_obs,
        scale_act=scale_act,
        scale_out=scale_out,
        sampling_idx=sampling_idx,
    )
    if use_log_uncertainties:
        if use_al_uncertainties:
            frac = eps_uncertainty / (al_uncertainty + 1e-6)
            reward = jnp.sum(jnp.log(1 + jnp.square(frac)), axis=-1)
        else:
            reward = jnp.sum(jnp.log(1 + jnp.square(eps_uncertainty)), axis=-1)
    else:
        reward = jnp.sum(jnp.square(eps_uncertainty), axis=-1)
    return next_obs, reward


class ActiveLearningPETSModel(BayesianDynamicsModel):
    def __init__(self,
                 use_log_uncertainties=False,
                 use_al_uncertainties=False,
                 *args,
                 **kwargs
                 ):
        super(ActiveLearningPETSModel, self).__init__(*args, **kwargs)
        self.use_log_uncertainties = use_log_uncertainties
        self.use_al_uncertainties = use_al_uncertainties
        self._init_fn()

    def _init_fn(self):
        super()._init_fn()
        def predict_with_uncertainty(
                parameters,
                obs,
                action,
                rng,
                sampling_idx=self.sampling_idx,
                alpha: Union[jnp.ndarray, float] = 1.0,
                bias_obs: Union[jnp.ndarray, float] = 0.0,
                bias_act: Union[jnp.ndarray, float] = 0.0,
                bias_out: Union[jnp.ndarray, float] = 0.0,
                scale_obs: Union[jnp.ndarray, float] = 1.0,
                scale_act: Union[jnp.ndarray, float] = 1.0,
                scale_out: Union[jnp.ndarray, float] = 1.0,
        ):
            return self._predict_with_uncertainty(
                predict_fn=self.model._predict,
                parameters=parameters,
                obs=obs,
                action=action,
                rng=rng,
                sampling_type=self.sampling_type,
                num_ensembles=self.model.num_ensembles,
                sampling_idx=sampling_idx,
                batch_size=obs.shape[0],
                alpha=alpha,
                bias_obs=bias_obs,
                bias_act=bias_act,
                bias_out=bias_out,
                scale_obs=scale_obs,
                scale_act=scale_act,
                scale_out=scale_out,
                pred_diff=self.pred_diff,
            )
        self.predict_with_uncertainty = jax.jit(predict_with_uncertainty)

        def _evaluate_for_exploration(
                parameters,
                obs,
                action,
                rng,
                alpha: Union[jnp.ndarray, float] = 1.0,
                bias_obs: Union[jnp.ndarray, float] = 0.0,
                bias_act: Union[jnp.ndarray, float] = 0.0,
                bias_out: Union[jnp.ndarray, float] = 0.0,
                scale_obs: Union[jnp.ndarray, float] = 1.0,
                scale_act: Union[jnp.ndarray, float] = 1.0,
                scale_out: Union[jnp.ndarray, float] = 1.0,
                sampling_idx: Optional[int] = None,
        ):
            return evaluate_for_exploration(
                pred_fn=self.predict_with_uncertainty,
                parameters=parameters,
                obs=obs,
                action=action,
                rng=rng,
                alpha=alpha,
                bias_obs=bias_obs,
                bias_act=bias_act,
                bias_out=bias_out,
                scale_obs=scale_obs,
                scale_act=scale_act,
                scale_out=scale_out,
                sampling_idx=sampling_idx,
                use_log_uncertainties=self.use_log_uncertainties,
                use_al_uncertainties=self.use_al_uncertainties,
            )

        self.evaluate_for_exploration = jax.jit(_evaluate_for_exploration)

    @staticmethod
    def _predict_with_uncertainty(
            predict_fn,
            parameters,
            obs,
            action,
            rng,
            sampling_type,
            num_ensembles,
            sampling_idx,
            batch_size,
            alpha: Union[jnp.ndarray, float] = 1.0,
            bias_obs: Union[jnp.ndarray, float] = 0.0,
            bias_act: Union[jnp.ndarray, float] = 0.0,
            bias_out: Union[jnp.ndarray, float] = 0.0,
            scale_obs: Union[jnp.ndarray, float] = 1.0,
            scale_act: Union[jnp.ndarray, float] = 1.0,
            scale_out: Union[jnp.ndarray, float] = 1.0,
            pred_diff: bool = 1,
    ):
        transformed_obs = (obs - bias_obs) / scale_obs
        transformed_act = (action - bias_act) / scale_act
        obs_action = jnp.concatenate([transformed_obs, transformed_act], axis=-1)
        next_obs_tot = predict_fn(parameters, obs_action)
        mean, std = jnp.split(next_obs_tot, 2, axis=-1)
        epistemic_uncertainty = jnp.std(mean, axis=0)
        aleatoric_uncertainty = jnp.mean(std, axis=0)

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
        next_obs = next_obs * scale_out + bias_out + pred_diff * obs
        return next_obs, epistemic_uncertainty * scale_out, aleatoric_uncertainty * scale_out


class ActiveLearningHUCRLModel(HUCRLModel):

    def __init__(self,
                 use_log_uncertainties=False,
                 use_al_uncertainties=False,
                 *args,
                 **kwargs
                 ):

        super(ActiveLearningHUCRLModel, self).__init__(*args, **kwargs)
        self.use_log_uncertainties = use_log_uncertainties
        self.use_al_uncertainties = use_al_uncertainties
        self._init_fn()

    def _init_fn(self):
        super()._init_fn()

        def predict_with_uncertainty(parameters,
                                     obs,
                                     action,
                                     rng,
                                     alpha: Union[jnp.ndarray, float] = 1.0,
                                     bias_obs: Union[jnp.ndarray, float] = 0.0,
                                     bias_act: Union[jnp.ndarray, float] = 0.0,
                                     bias_out: Union[jnp.ndarray, float] = 0.0,
                                     scale_obs: Union[jnp.ndarray, float] = 1.0,
                                     scale_act: Union[jnp.ndarray, float] = 1.0,
                                     scale_out: Union[jnp.ndarray, float] = 1.0,
                                     sampling_idx: Optional[int] = None,
                                     ):
            return self._predict_with_uncertainty(
                predict_fn=self.model._predict,
                parameters=parameters,
                obs=obs,
                act_dim=self.act_dim,
                action=action,
                rng=rng,
                beta=self.beta,
                batch_size=obs.shape[0],
                alpha=alpha,
                bias_obs=bias_obs,
                bias_act=bias_act,
                bias_out=bias_out,
                scale_obs=scale_obs,
                scale_act=scale_act,
                scale_out=scale_out,
                pred_diff=self.pred_diff,
                sampling_idx=sampling_idx,
            )

        self.predict_with_uncertainty = jax.jit(predict_with_uncertainty)

        def _evaluate_for_exploration(
                parameters,
                obs,
                action,
                rng,
                alpha: Union[jnp.ndarray, float] = 1.0,
                bias_obs: Union[jnp.ndarray, float] = 0.0,
                bias_act: Union[jnp.ndarray, float] = 0.0,
                bias_out: Union[jnp.ndarray, float] = 0.0,
                scale_obs: Union[jnp.ndarray, float] = 1.0,
                scale_act: Union[jnp.ndarray, float] = 1.0,
                scale_out: Union[jnp.ndarray, float] = 1.0,
                sampling_idx: Optional[int] = None,
        ):
            return evaluate_for_exploration(
                pred_fn=self.predict_with_uncertainty,
                parameters=parameters,
                obs=obs,
                action=action,
                rng=rng,
                alpha=alpha,
                bias_obs=bias_obs,
                bias_act=bias_act,
                bias_out=bias_out,
                scale_obs=scale_obs,
                scale_act=scale_act,
                scale_out=scale_out,
                sampling_idx=sampling_idx,
                use_log_uncertainties=self.use_log_uncertainties,
                use_al_uncertainties=self.use_al_uncertainties,
            )

        self.evaluate_for_exploration = jax.jit(_evaluate_for_exploration)

        def predict_without_optimism(parameters,
                                     obs,
                                     action,
                                     rng,
                                     alpha: Union[jnp.ndarray, float] = 1.0,
                                     bias_obs: Union[jnp.ndarray, float] = 0.0,
                                     bias_act: Union[jnp.ndarray, float] = 0.0,
                                     bias_out: Union[jnp.ndarray, float] = 0.0,
                                     scale_obs: Union[jnp.ndarray, float] = 1.0,
                                     scale_act: Union[jnp.ndarray, float] = 1.0,
                                     scale_out: Union[jnp.ndarray, float] = 1.0,
                                     sampling_idx: Optional[int] = None,
                                     ):
            return self._predict(
                predict_fn=self.model._predict,
                parameters=parameters,
                obs=obs,
                act_dim=self.act_dim,
                action=action,
                rng=rng,
                num_ensembles=self.model.num_ensembles,
                beta=self.beta,
                batch_size=obs.shape[0],
                alpha=alpha,
                bias_obs=bias_obs,
                bias_act=bias_act,
                bias_out=bias_out,
                scale_obs=scale_obs,
                scale_act=scale_act,
                scale_out=scale_out,
                pred_diff=self.pred_diff,
                use_optimism=False,
                sampling_idx=sampling_idx,
            )

        self.predict_without_optimism = jax.jit(predict_without_optimism)

        def evaluate(
                parameters,
                obs,
                action,
                rng,
                alpha: Union[jnp.ndarray, float] = 1.0,
                bias_obs: Union[jnp.ndarray, float] = 0.0,
                bias_act: Union[jnp.ndarray, float] = 0.0,
                bias_out: Union[jnp.ndarray, float] = 0.0,
                scale_obs: Union[jnp.ndarray, float] = 1.0,
                scale_act: Union[jnp.ndarray, float] = 1.0,
                scale_out: Union[jnp.ndarray, float] = 1.0,
                sampling_idx: Optional[int] = None,
        ):
            return self._evaluate(
                pred_fn=self.predict_without_optimism,
                reward_fn=self.reward_model.predict,
                parameters=parameters,
                obs=obs,
                action=action,
                rng=rng,
                alpha=alpha,
                bias_obs=bias_obs,
                bias_act=bias_act,
                bias_out=bias_out,
                scale_obs=scale_obs,
                scale_act=scale_act,
                scale_out=scale_out,
                sampling_idx=sampling_idx,
            )

        self.evaluate = jax.jit(evaluate)

    @staticmethod
    def _predict_with_uncertainty(
            predict_fn,
            parameters,
            act_dim: int,
            obs,
            action,
            rng,
            beta,
            batch_size,
            alpha: Union[jnp.ndarray, float] = 1.0,
            bias_obs: Union[jnp.ndarray, float] = 0.0,
            bias_act: Union[jnp.ndarray, float] = 0.0,
            bias_out: Union[jnp.ndarray, float] = 0.0,
            scale_obs: Union[jnp.ndarray, float] = 1.0,
            scale_act: Union[jnp.ndarray, float] = 1.0,
            scale_out: Union[jnp.ndarray, float] = 1.0,
            pred_diff: bool = 1,
            sampling_idx: Optional[int] = None,
    ):
        act, eta = jnp.split(action, axis=-1, indices_or_sections=[act_dim])
        transformed_obs = (obs - bias_obs) / scale_obs
        transformed_act = (act - bias_act) / scale_act
        obs_action = jnp.concatenate([transformed_obs, transformed_act], axis=-1)
        next_obs_tot = predict_fn(parameters, obs_action)
        mean, std = jnp.split(next_obs_tot, 2, axis=-1)

        if rng is None:
            next_obs = jnp.mean(mean, axis=0)
            next_obs_eps_std = jnp.std(mean, axis=0)
            al_uncertainty = jnp.sqrt(jnp.mean(jnp.square(std), axis=0))

        else:
            def get_epistemic_estimate(mean, std, eta, rng):
                next_obs_eps_std = jnp.std(mean, axis=0)
                al_uncertainty = jnp.sqrt(jnp.mean(jnp.square(std), axis=0))
                next_state_mean = jnp.mean(mean, axis=0) + beta * alpha * next_obs_eps_std * eta
                next_obs = sample_normal_dist(
                    next_state_mean,
                    al_uncertainty,
                    rng,
                )
                return next_obs, next_obs_eps_std, al_uncertainty

            sample_rng = jax.random.split(
                rng,
                batch_size
            )
            next_obs, next_obs_eps_std, al_uncertainty = jax.vmap(get_epistemic_estimate,
                                                                  in_axes=(1, 1, 0, 0), out_axes=0) \
                    (
                    mean,
                    std,
                    eta,
                    sample_rng
                )
        next_obs = next_obs * scale_out + bias_out + pred_diff * obs
        return next_obs, next_obs_eps_std * scale_out, al_uncertainty * scale_out
