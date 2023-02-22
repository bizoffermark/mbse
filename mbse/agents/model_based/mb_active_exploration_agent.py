import jax

import numpy as np
from mbse.agents.model_based.model_based_agent import ModelBasedAgent
import jax.numpy as jnp


class MBActiveExplorationAgent(ModelBasedAgent):

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super(MBActiveExplorationAgent, self).__init__(*args, **kwargs)
        self._init_fn()

    def _init_fn(self):
        super()._init_fn()
        def _optimize_for_exploration(
                params,
                init_state,
                key,
                optimizer_key,
                bias_obs,
                bias_act,
                bias_out,
                scale_obs,
                scale_act,
                scale_out):
            return self._optimize(
                eval_fn=self.dynamics_model.evaluate_for_exploration,
                optimize_fn=self.policy_optimzer.optimize,
                n_particles=self.n_particles,
                horizon=self.policy_optimzer.action_dim[-2],
                params=params,
                init_state=init_state,
                key=key,
                optimizer_key=optimizer_key,
                bias_obs=bias_obs,
                bias_act=bias_act,
                bias_out=bias_out,
                scale_obs=scale_obs,
                scale_act=scale_act,
                scale_out=scale_out,
            )

        self.optimize_for_exploration = jax.jit(
            _optimize_for_exploration
        )
        self.act_in_train = lambda obs, rng: \
            np.asarray(self.act_in_train_jax(jnp.asarray(obs), rng))

    def act_in_train_jax(self, obs, rng):
        def optimize(init_state, key, optimizer_key):
            action_seq, reward = self.optimize_for_exploration(
                params=self.dynamics_model.model_params,
                init_state=init_state,
                key=key,
                optimizer_key=optimizer_key,
                bias_obs=self.dynamics_model.bias_obs,
                bias_act=self.dynamics_model.bias_act,
                bias_out=self.dynamics_model.bias_out,
                scale_obs=self.dynamics_model.scale_obs,
                scale_act=self.dynamics_model.scale_act,
                scale_out=self.dynamics_model.scale_out,
            )
            return action_seq, reward
        n_envs = obs.shape[0]
        obs = jnp.repeat(jnp.expand_dims(obs, 1), self.n_particles, 1)
        rollout_rng, optimizer_rng = jax.random.split(rng, 2)
        rollout_rng = jax.random.split(rollout_rng, n_envs)
        optimizer_rng = jax.random.split(optimizer_rng, n_envs)

        action_sequence, best_reward = jax.vmap(optimize)(
            obs,
            rollout_rng,
            optimizer_rng
        )

        action = action_sequence[:, 0, ...]
        action = action[..., :self.action_space.shape[0]]
        return action


