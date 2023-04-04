import jax
import jax.numpy as jnp
from mbse.optimizers.cross_entropy_optimizer import CrossEntropyOptimizer
from mbse.utils.utils import sample_trajectories
import functools
from typing import Optional, Union


class CemTO(object):
    def __init__(self,
                 horizon: int,
                 action_dim: tuple,
                 dynamics_model_list: list,
                 n_particles: int = 10,
                 *cem_args,
                 **cem_kwargs
                 ):
        cem_action_dim = (horizon,) + action_dim
        self.horizon = horizon
        self.n_particles = n_particles
        self.optimizer = CrossEntropyOptimizer(action_dim=cem_action_dim, *cem_args, **cem_kwargs)
        assert isinstance(dynamics_model_list, list)
        self.dynamics_model_list = dynamics_model_list
        self._init_fn()

    def _init_fn(self):
        def _get_action_sequence(
                model_index,
                dynamics_params,
                obs,
                key=None,
                optimizer_key=None,
                alpha: Union[jnp.ndarray, float] = 1.0,
                bias_obs: Union[jnp.ndarray, float] = 0.0,
                bias_act: Union[jnp.ndarray, float] = 0.0,
                bias_out: Union[jnp.ndarray, float] = 0.0,
                scale_obs: Union[jnp.ndarray, float] = 1.0,
                scale_act: Union[jnp.ndarray, float] = 1.0,
                scale_out: Union[jnp.ndarray, float] = 1.0,
                initial_actions: Optional[jax.Array] = None,
                sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
        ):
            obs = jnp.repeat(jnp.expand_dims(obs, 0), self.n_particles, 0)
            return self._optimize_action_sequence(
                eval_fn=self.dynamics_model_list[model_index].evaluate,
                optimize_fn=self.optimizer.optimize,
                n_particles=self.n_particles,
                horizon=self.horizon,
                params=dynamics_params,
                init_state=obs,
                key=key,
                optimizer_key=optimizer_key,
                alpha=alpha,
                bias_obs=bias_obs,
                bias_act=bias_act,
                bias_out=bias_out,
                scale_obs=scale_obs,
                scale_act=scale_act,
                scale_out=scale_out,
                sampling_idx=sampling_idx,
                init_action_seq=initial_actions,
            )

        self.optimize_for_eval_fns = []
        for i in range(len(self.dynamics_model_list)):
            self.optimize_for_eval_fns.append(jax.jit(functools.partial(
                _get_action_sequence, model_index=i
            )))
        self.optimize = self.optimize_for_eval_fns[0]

        def _get_action_sequence_for_exploration(
                dynamics_params,
                obs,
                key=None,
                optimizer_key=None,
                alpha: Union[jnp.ndarray, float] = 1.0,
                bias_obs: Union[jnp.ndarray, float] = 0.0,
                bias_act: Union[jnp.ndarray, float] = 0.0,
                bias_out: Union[jnp.ndarray, float] = 0.0,
                scale_obs: Union[jnp.ndarray, float] = 1.0,
                scale_act: Union[jnp.ndarray, float] = 1.0,
                scale_out: Union[jnp.ndarray, float] = 1.0,
                initial_actions: Optional[jax.Array] = None,
                sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
        ):
            obs = jnp.repeat(jnp.expand_dims(obs, 0), self.n_particles, 0)
            return self._optimize_action_sequence(
                eval_fn=self.dynamics_model.evaluate_for_exploration,
                optimize_fn=self.optimizer.optimize,
                n_particles=self.n_particles,
                horizon=self.horizon,
                params=dynamics_params,
                init_state=obs,
                key=key,
                alpha=alpha,
                optimizer_key=optimizer_key,
                bias_obs=bias_obs,
                bias_act=bias_act,
                bias_out=bias_out,
                scale_obs=scale_obs,
                scale_act=scale_act,
                scale_out=scale_out,
                init_action_seq=initial_actions,
                sampling_idx=sampling_idx,
            )

        self.optimize_for_exploration = jax.jit(
            _get_action_sequence_for_exploration
        )

    @staticmethod
    def _optimize_action_sequence(eval_fn,
                                  optimize_fn,
                                  n_particles,
                                  horizon,
                                  params,
                                  init_state,
                                  key,
                                  optimizer_key,
                                  alpha: Union[float, jax.Array],
                                  bias_obs: Union[float, jax.Array],
                                  bias_act: Union[float, jax.Array],
                                  bias_out: Union[float, jax.Array],
                                  scale_obs: Union[float, jax.Array],
                                  scale_act: Union[float, jax.Array],
                                  scale_out: Union[float, jax.Array],
                                  init_action_seq: Optional[jax.Array] = None,
                                  sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
                                  ):
        eval_func = lambda seq, x, k: sample_trajectories(
            evaluate_fn=eval_fn,
            parameters=params,
            init_state=x,
            horizon=horizon,
            key=k,
            actions=seq,
            alpha=alpha,
            bias_obs=bias_obs,
            bias_act=bias_act,
            bias_out=bias_out,
            scale_obs=scale_obs,
            scale_act=scale_act,
            scale_out=scale_out,
            sampling_idx=sampling_idx,
        )

        def sum_rewards(seq):
            # seq = jnp.repeat(jnp.expand_dims(seq, 0), n_particles, 0)

            def get_average_reward(obs, eval_key):
                transition = eval_func(seq, obs, eval_key)
                return transition.reward.mean()

            if key is not None:
                optimizer_key = jax.random.split(key=key, num=n_particles)
                returns = jax.vmap(get_average_reward)(init_state, optimizer_key)
            else:
                returns = jax.vmap(get_average_reward, in_axes=(0, None))(init_state, key)
            return returns.mean()

        action_seq, reward = optimize_fn(
            func=sum_rewards,
            rng=optimizer_key,
            mean=init_action_seq,
        )
        return action_seq, reward

    @property
    def dynamics_model(self):
        return self.dynamics_model_list[0]
