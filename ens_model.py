import time
from mbse.utils.utils import denormalize, normalize, get_data_jax, create_data
import jax.numpy as jnp 
import matplotlib.pyplot as plt
from jax.lax import cond
from trajax.optimizers import CEMHyperparams, ILQRHyperparams, ilqr_with_cem_warmstart, cem, ilqr
import pickle
import math

class EnsembleModel():
    def __init__(self):#, path_model, paths_params):
        model_path = "checkpoints/revived-morning-99/best_model.pkl"
        params_path = ["checkpoints/revived-morning-99/best_params.pkl",
                    "checkpoints/jumping-cloud-101/best_params.pkl",
                    "checkpoints/vital-morning-100/best_params.pkl",
                    "checkpoints/confused-feather-96/best_params.pkl",
                    "checkpoints/misunderstood-sponge-102/best_params.pkl"]

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.params = []
        for path in params_path:
            with open(path, "rb") as f:
                self.params.append(pickle.load(f))
        self.control_low = jnp.array([-0.7]).reshape(1, )
        self.control_high = jnp.array([0.7]).reshape(1, )
        self.cem_params = CEMHyperparams(max_iter=10, sampling_smoothing=0.0, num_samples=200, evolution_smoothing=0.0,
                        elite_portion=0.1)  
        self.ilqr_params = ILQRHyperparams(maxiter=100)

        self.data = get_data_jax()

    def predict(self, x):
        return jnp.array([self.model.apply(params, x) for params in self.params]).mean(0)
    
    def _cost_fn(self, state, action, t, qs = [100, 1, 1], target_state= [math.pi, 0]): 
        assert state.shape == (x_dim,) and action.shape == (u_dim,)

        theta = state[0]
        theta_dot = state[1]

        theta_star = target_state[0]
        theta_dot_star = target_state[1]

        q_theta = qs[0]
        q_theta_dot = qs[1]
        q_u = qs[2]

        def running_cost(theta, theta_dot, action):
            return q_theta * jnp.sum((theta - theta_star) ** 2) + \
                + q_theta_dot * jnp.sum((theta_dot - theta_dot_star)**2) \
                + q_u * jnp.sum(action ** 2)
        
        def terminal_cost(theta, theta_dot, action):
            return q_theta * jnp.sum((theta - theta_star) ** 2) + \
                + q_theta_dot * jnp.sum((theta_dot - theta_dot_star)**2)

        return cond(t == num_steps, terminal_cost, running_cost, theta, theta_dot, action)

    def _dynamics_fn(self, x, u, t):
        print(x.shape)
        print(u.shape)
        assert x.shape == (x_dim,) and u.shape == (u_dim,)
        inputs = jnp.concatenate([x, u]).transpose()
        print(inputs.shape)
        inputs = normalize(inputs, self.data['train']['mu_x'], self.data['train']['std_x'])
        out = self.predict(inputs)
        # out = self.model.apply(self.params, inputs)
        out = denormalize(out, data['train']['mu_y'], data['train']['std_y'])
        return out + x

    def forward_traj(self, x, n_horizon, optimizer='ilqr_warmup'):
        # x_traj = jnp.zeros((n_horizon, x.shape[0]))
        u = jnp.zeros((n_horizon, u_dim))
        if optimizer == 'ilqr_warmup':
            out = ilqr_with_cem_warmstart(self._cost_fn, self._dynamics_fn, x, u, control_low=self.control_low,
                              control_high=self.control_high, ilqr_hyperparams=self.ilqr_params, cem_hyperparams=self.cem_params)
        elif optimizer == 'cem':
            out = cem(self._cost_fn, self._dynamics_fn, x, u, control_low=self.control_low,
                              control_high=self.control_high, cem_hyperparams=self.cem_params)
        elif optimizer == 'ilqr':
            out = ilqr(self._cost_fn, self._dynamics_fn, x, u, control_low=self.control_low,
                              control_high=self.control_high)

        xs = out[0]
        us = out[1]
        return xs, us

# num_steps = 100
# initial_state = data['test']['states'][0] #jnp.array([jnp.pi / 2, 0.0])
# initial_state = jnp.array([0.0, 0.0, initial_state[2], 0.0])
# initial_actions = jnp.zeros(shape=(num_steps, u_dim)) #jnp.zeros(shape=(int(T / dt), u_dim))
