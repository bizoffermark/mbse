from typing import Optional, Union
import jax
import jax.numpy as jnp
from mbse.utils.utils import sample_normal_dist
from mbse.utils.replay_buffer import Transition
from mbse.models.bayesian_dynamics_model import BayesianDynamicsModel


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


class HUCRLModel(BayesianDynamicsModel):

    def __init__(self,
                 beta: float = 1.0,
                 *args,
                 **kwargs
                 ):

        super(HUCRLModel, self).__init__(*args, **kwargs)
        self.beta = beta
        self._init_fn()

    def _init_fn(self):

        super()._init_fn()

        def predict(parameters,
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
                # batch_size=obs.shape[0],
                alpha=alpha,
                bias_obs=bias_obs,
                bias_act=bias_act,
                bias_out=bias_out,
                scale_obs=scale_obs,
                scale_act=scale_act,
                scale_out=scale_out,
                pred_diff=self.pred_diff,
                use_optimism=True,
                sampling_idx=sampling_idx,
            )

        self.predict = jax.jit(predict)

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
                pred_fn=self.predict,
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

        def _train_step(
                tran: Transition,
                model_params,
                model_opt_state,
                val: Optional[Transition] = None
        ):
            return self._train(
                train_fn=self.model._train_step,
                predict_fn=self.model._predict,
                calibrate_fn=self.model.calculate_calibration_alpha,
                tran=tran,
                model_params=model_params,
                model_opt_state=model_opt_state,
                val=val,
            )

        self._train_step = jax.jit(_train_step)

    @staticmethod
    def _predict(predict_fn,
                 parameters,
                 act_dim,
                 obs,
                 action,
                 rng,
                 num_ensembles,
                 beta,
                 alpha: Union[jnp.ndarray, float] = 1.0,
                 bias_obs: Union[jnp.ndarray, float] = 0.0,
                 bias_act: Union[jnp.ndarray, float] = 0.0,
                 bias_out: Union[jnp.ndarray, float] = 0.0,
                 scale_obs: Union[jnp.ndarray, float] = 1.0,
                 scale_act: Union[jnp.ndarray, float] = 1.0,
                 scale_out: Union[jnp.ndarray, float] = 1.0,
                 pred_diff: bool = 1,
                 use_optimism: bool = 1,
                 sampling_idx: Optional[int] = None,
                 ):
        act, eta = jnp.split(action, axis=-1, indices_or_sections=[act_dim])
        transformed_obs = (obs - bias_obs) / scale_obs
        transformed_act = (act - bias_act) / scale_act
        obs_action = jnp.concatenate([transformed_obs, transformed_act], axis=-1)
        next_obs_tot = predict_fn(parameters, obs_action)
        mean, std = jnp.split(next_obs_tot, 2, axis=-1)

        if rng is None:
            mean, _ = jnp.split(next_obs_tot, 2, axis=-1)
            next_obs = jnp.mean(mean, axis=0)

        else:
            def get_epistemic_estimate(mean, std, eta, rng):
                next_obs_eps_std = alpha * jnp.std(mean, axis=0)
                al_uncertainty = jnp.sqrt(jnp.mean(jnp.square(std), axis=0))
                next_state_mean = jnp.mean(mean, axis=0) + beta * next_obs_eps_std * eta * use_optimism
                next_obs = sample_normal_dist(
                    next_state_mean,
                    al_uncertainty,
                    rng,
                )
                return next_obs

            # sample_rng = jax.random.split(
            #    rng,
            #    batch_size
            # )
            #next_obs = jax.vmap(get_epistemic_estimate, in_axes=(1, 1, 0, 0), out_axes=0)(
            #    mean,
            #    std,
            #    eta,
            #    sample_rng
            #)
            next_obs = get_epistemic_estimate(mean, std, eta, rng)
        next_obs = next_obs * scale_out + bias_out + pred_diff * obs
        return next_obs

    @staticmethod
    def _evaluate(
            pred_fn,
            reward_fn,
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
            alpha=alpha,
            bias_obs=bias_obs,
            bias_act=bias_act,
            bias_out=bias_out,
            scale_obs=scale_obs,
            scale_act=scale_act,
            scale_out=scale_out,
        )
        # transformed_obs, transformed_action, _ = self.transforms(obs, action)
        # transformed_next_obs = self.predict(transformed_obs, transformed_action, model_rng)
        # _, _, next_obs = self.inverse_transforms(transformed_obs=transformed_obs, transformed_action=None,
        #                                         transformed_next_state=transformed_next_obs)
        reward = reward_fn(obs, action, next_obs, reward_rng)
        return next_obs, reward
