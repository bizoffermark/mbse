from trajax.optimizers import ILQR, ILQRHyperparams
from typing import Union, Callable, Optional
import jax
import jax.numpy as jnp
import functools


def _optimize_with_params(reward_fn: Callable,
                          dynamics_fn: Callable,
                          initial_state: jax.Array,
                          initial_actions: jax.Array,
                          optimizer_params: ILQRHyperparams,
                          dynamics_params,
                          alpha: Union[float, jax.Array],
                          bias_obs: Union[float, jax.Array],
                          bias_act: Union[float, jax.Array],
                          bias_out: Union[float, jax.Array],
                          scale_obs: Union[float, jax.Array],
                          scale_act: Union[float, jax.Array],
                          scale_out: Union[float, jax.Array],
                          init_var: Union[float, jax.Array] = 5.0,
                          cost_params: Optional = None,
                          key: Optional = None,
                          optimizer_key: Optional = None,
                          sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
                          ):
    def dynamics(x, u, t, dynamic_params):
        action = jnp.tanh(u)
        return dynamics_fn(
            parameters=dynamic_params,
            obs=x,
            action=action,
            rng=key,
            alpha=alpha,
            bias_obs=bias_obs,
            bias_act=bias_act,
            bias_out=bias_out,
            scale_obs=scale_obs,
            scale_act=scale_act,
            scale_out=scale_out,
            sampling_idx=sampling_idx,
        )

    def cost_fn(x, u, t, params):
        action = jnp.tanh(u)
        return - reward_fn(x, action).sum()

    if optimizer_key is not None:
        sampled_action = jax.random.multivariate_normal(
            key=optimizer_key,
            mean=jnp.zeros_like(initial_actions.reshape(-1, 1).squeeze()),
            cov=jnp.diag(jnp.ones_like(initial_actions.reshape(-1, 1).squeeze()))
        ) * init_var
        sampled_action = sampled_action.reshape(initial_actions.shape)
        init_act = initial_actions + sampled_action
    else:
        init_act = initial_actions
    ilqr = ILQR(cost_fn, dynamics)
    out = ilqr.solve(cost_params, dynamics_params, initial_state, init_act, optimizer_params)
    return jnp.clip(jnp.tanh(out.us), -1, 1), -out.obj


class TraJaxTO(object):

    def __init__(self,
                 horizon: int,
                 action_dim: tuple,
                 dynamics_model_list: list,
                 n_particles: int = 10,
                 params: ILQRHyperparams = ILQRHyperparams(),
                 initial_actions: Optional[jax.Array] = None,
                 *args,
                 **kwargs):
        self.horizon = horizon
        self.action_dim = action_dim
        self.params = params
        self.n_particles = n_particles
        assert isinstance(dynamics_model_list, list)
        self.dynamics_model_list = dynamics_model_list
        if initial_actions is None:
            self.previous_actions = jnp.zeros((horizon,) + action_dim)
        else:
            assert initial_actions.shape == (self.horizon,) + self.action_dim
            self.previous_actions = initial_actions
        self._init_fn()

    def _init_fn(self):

        def _get_action_sequence(
                model_index,
                obs,
                key=None,
                optimizer_key=None,
                dynamics_params=None,
                alpha: Union[jnp.ndarray, float] = 1.0,
                bias_obs: Union[jnp.ndarray, float] = 0.0,
                bias_act: Union[jnp.ndarray, float] = 0.0,
                bias_out: Union[jnp.ndarray, float] = 0.0,
                scale_obs: Union[jnp.ndarray, float] = 1.0,
                scale_act: Union[jnp.ndarray, float] = 1.0,
                scale_out: Union[jnp.ndarray, float] = 1.0,
                sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
                initial_actions: Optional[jax.Array] = self.previous_actions,
        ):
            dynamics_fn = self.dynamics_model_list[model_index].predict

            reward_fn = self.dynamics_model_list[model_index].reward_model.predict
            if initial_actions is None:
                initial_actions = self.previous_actions
            else:
                assert initial_actions.shape == (self.horizon,) + self.action_dim
            obs = jnp.repeat(jnp.expand_dims(obs, 0), self.n_particles, 0)

            def get_sequence_and_returns_for_init_state(x0, opt_key):
                return _optimize_with_params(
                    reward_fn=reward_fn,
                    dynamics_fn=dynamics_fn,
                    initial_state=x0,
                    initial_actions=initial_actions,
                    optimizer_params=self.params,
                    dynamics_params=dynamics_params,
                    alpha=alpha,
                    bias_obs=bias_obs,
                    bias_act=bias_act,
                    bias_out=bias_out,
                    scale_obs=scale_obs,
                    scale_act=scale_act,
                    scale_out=scale_out,
                    key=key,
                    optimizer_key=opt_key,
                    sampling_idx=sampling_idx,
                )

            if optimizer_key is not None:
                opt_key = jax.random.split(key=key, num=self.n_particles)
                sequence, returns = jax.vmap(get_sequence_and_returns_for_init_state)(obs, opt_key)
            else:
                sequence, returns = jax.vmap(get_sequence_and_returns_for_init_state, in_axes=(0, None)) \
                    (obs, optimizer_key)
            best_elite_idx = jnp.argsort(returns, axis=0).squeeze()[-1]
            elite_action = sequence[best_elite_idx]
            return elite_action, returns[best_elite_idx]

        self.optimize_for_eval_fns = []
        for i in range(len(self.dynamics_model_list)):
            self.optimize_for_eval_fns.append(jax.jit(functools.partial(
                _get_action_sequence, model_index=i
            )))
        self.optimize = self.optimize_for_eval_fns[0]

        def _get_action_sequence_for_exploration(
                obs,
                key=None,
                optimizer_key=None,
                dynamics_params=None,
                alpha: Union[jnp.ndarray, float] = 1.0,
                bias_obs: Union[jnp.ndarray, float] = 0.0,
                bias_act: Union[jnp.ndarray, float] = 0.0,
                bias_out: Union[jnp.ndarray, float] = 0.0,
                scale_obs: Union[jnp.ndarray, float] = 1.0,
                scale_act: Union[jnp.ndarray, float] = 1.0,
                scale_out: Union[jnp.ndarray, float] = 1.0,
                sampling_idx: Optional[Union[jnp.ndarray, int]] = None,
                initial_actions: Optional[jax.Array] = self.previous_actions,
        ):
            dynamics = self.dynamics_model.evaluate_for_exploration
            obs = jnp.concatenate([obs, jnp.zeros(1)], axis=0)

            def dynamics_fn(obs, action, *args, **kwargs):
                state = obs[:-1]
                next_state, reward = dynamics(obs=state, action=action, *args, **kwargs)
                next_state = jnp.concatenate([next_state, reward.reshape(-1)], axis=0)
                return next_state

            def reward_fn(state, action, *args, **kwargs):
                reward = state[-1]
                return reward.mean()

            if initial_actions is None:
                initial_actions = self.previous_actions
            else:
                assert initial_actions.shape == (self.horizon,) + self.action_dim
            obs = jnp.repeat(jnp.expand_dims(obs, 0), self.n_particles, 0)

            def get_sequence_and_returns_for_init_state(x0, opt_key):
                return _optimize_with_params(
                    reward_fn=reward_fn,
                    dynamics_fn=dynamics_fn,
                    initial_state=x0,
                    initial_actions=initial_actions,
                    optimizer_params=self.params,
                    dynamics_params=dynamics_params,
                    alpha=alpha,
                    bias_obs=bias_obs,
                    bias_act=bias_act,
                    bias_out=bias_out,
                    scale_obs=scale_obs,
                    scale_act=scale_act,
                    scale_out=scale_out,
                    key=key,
                    optimizer_key=opt_key,
                    sampling_idx=sampling_idx,
                )

            if optimizer_key is not None:
                opt_key = jax.random.split(key=key, num=self.n_particles)
                sequence, returns = jax.vmap(get_sequence_and_returns_for_init_state)(obs, opt_key)
            else:
                sequence, returns = jax.vmap(get_sequence_and_returns_for_init_state, in_axes=(0, None)) \
                    (obs, optimizer_key)
            best_elite_idx = jnp.argsort(returns, axis=0).squeeze()[-1]
            elite_action = sequence[best_elite_idx]
            return elite_action, returns[best_elite_idx]

        self.optimize_for_exploration = jax.jit(
            _get_action_sequence_for_exploration
        )

    @property
    def dynamics_model(self):
        return self.dynamics_model_list[0]
