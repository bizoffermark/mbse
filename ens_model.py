import sys
sys.path.append('../mbse')
from mbse.utils.utils import denormalize, normalize, get_data_jax
import jax.numpy as jnp 
from jax.lax import cond
from trajax.optimizers import CEMHyperparams, ILQRHyperparams, ilqr_with_cem_warmstart, cem, ilqr
import pickle
import math
import jax
from models import MLP
from gym.spaces import Box
import numpy as np
from gym.envs.classic_control.pendulum import angle_normalize
import pandas as pd

class EnsembleModel():
    def __init__(self, train_horizon=1, n_models=5):#, path_model, paths_params):
        # model_path = "checkpoints/revived-morning-99/best_model.pkl"
        # params_path = ["checkpoints/revived-morning-99/best_params.pkl",
        #             "checkpoints/jumping-cloud-101/best_params.pkl",
        #             "checkpoints/vital-morning-100/best_params.pkl",
        #             "checkpoints/confused-feather-96/best_params.pkl",
        #             "checkpoints/misunderstood-sponge-102/best_params.pkl"]
        
        # with open(model_path, "rb") as f:
        #     self.model = pickle.load(f)

        # self.params = []
        # for path in params_path:
        #     with open(path, "rb") as f:
        #         self.params.append(pickle.load(f))
        self.train_horizon = train_horizon
        self.n_models = n_models
        
        self.model_catalog = ModelSelection()
        self.models, self.params = self.model_catalog.load_models(train_horizon=self.train_horizon,
                                                                  n_models=self.n_models)
        self.cem_params = CEMHyperparams(max_iter=10, sampling_smoothing=0.0, num_samples=200, evolution_smoothing=0.0,
                        elite_portion=0.1)  
        # TODO: test with different values of psd_delta 
        self.ilqr_params = ILQRHyperparams(maxiter=100, make_psd=True, psd_delta=1e-2)

        with open("metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
        
        self.x_dim = 4
        self.u_dim = 1
        self.n_horizon = 50
        
        self.obs_min = np.array(self.metadata['min_x'][:-1])
        self.obs_max = np.array(self.metadata['max_x'][:-1])
        
        self.action_min = np.array([self.metadata['min_x'][-1]])
        self.action_max = np.array([self.metadata['max_x'][-1]])
        
        print("obs_min", self.obs_min)
        print("obs_max", self.obs_max)
        
        print("action_min", self.action_min)
        print("action_max", self.action_max)
        
        self.obs_space = Box(low=self.obs_min, high=self.obs_max, dtype=np.float32)
        self.action_space = Box(low=self.action_min, high=self.action_max, dtype=np.float32)

    @staticmethod
    @jax.jit
    def normalize(x, u, metadata):
        inputs = jnp.concatenate([x, u]).transpose()
        inputs = normalize(inputs, metadata['mu_x'], metadata['std_x'])
        return inputs
    
    @staticmethod
    @jax.jit
    def denormalize(out, metadata):    
        out = denormalize(out, metadata['mu_y'], metadata['std_y'])
        return out
    
    def normalize_and_predict(self, x, u):
        '''
            predict the dx given x and u after normalizing x and u
        '''
        print("x before normalize: ", x)
        print("u before normalize: ", u)
        xu = self.normalize(x, u, self.metadata)
        print("xu after normalize: ", xu)
        return self.predict(xu)
        
    def predict(self, x, u=None):
        '''
            predict the dx given x and u
        '''
        if u is None:
            xu = x
        else:
            try:
                assert x.shape[1] == self.x_dim
                axis = 1
            except:
                axis = 0
        
            xu = jnp.concatenate([x, u], axis=axis)
        ds = jnp.array([self.model.apply(params, xu) for params in self.params]).mean(0)
        print("ds before denormalize: ", ds)
        ds = self.denormalize(ds, self.metadata)
        
        return ds
    
    def step(self, x, u):
        '''
            predict the next state given x and u
            
            input:
                x: (x_dim, )
                u: (u_dim, )
            
            output:
                x_next: (x_dim, )
                terminate: bool
                truncate: bool
                output_dict: dict
        '''
        print("x before step: ", x)
        print("u before step: ", u)
        dx = self.normalize_and_predict(x, u)
        x_next = x + dx
        x_next = x_next.at[0].set(angle_normalize(x_next[0]))

        return x_next, False, False, {}
    
    def _cost_fn(self, state, action, t, qs = [100, 1, 1], target_state= [math.pi, 0]): 
        assert state.shape == (self.x_dim,) and action.shape == (self.u_dim,)

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

        return cond(t == self.n_horizon, terminal_cost, running_cost, theta, theta_dot, action)

    def _dynamics_fn(self, x, u, t):
        print(x.shape)
        print(u.shape)
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        
        return self.step(x, u) 
            
    def forward_traj(self, x, n_horizon, optimizer='ilqr_warmup'):
        # x_traj = jnp.zeros((n_horizon, x.shape[0]))

        self.n_horizon = n_horizon

        u = jnp.zeros((n_horizon, self.u_dim))

        print(x.shape)
        print(u.shape)

        if optimizer == 'ilqr_warmup':
            out = ilqr_with_cem_warmstart(self._cost_fn, self._dynamics_fn, x, u, control_low=self.action_min,
                              control_high=self.action_max, ilqr_hyperparams=self.ilqr_params, cem_hyperparams=self.cem_params)
        elif optimizer == 'cem':
            out = cem(self._cost_fn, self._dynamics_fn, x, u, control_low=self.action_min,
                              control_high=self.action_max, cem_hyperparams=self.cem_params)
        elif optimizer == 'ilqr':
            out = ilqr(self._cost_fn, self._dynamics_fn, x, u, control_low=self.action_min,
                              control_high=self.action_max)

        xs = out[0]
        us = out[1]
        return xs, us

# num_steps = 100
# initial_state = data['test']['states'][0] #jnp.array([jnp.pi / 2, 0.0])
# initial_state = jnp.array([0.0, 0.0, initial_state[2], 0.0])
# initial_actions = jnp.zeros(shape=(num_steps, u_dim)) #jnp.zeros(shape=(int(T / dt), u_dim))

class ModelSelection():
    def __init__(self):
        self.df = pd.read_csv("data/paths.csv")

    def load_models(self, train_horizon=1, n_models=5):
        # choose the column "train_horizon" == train_horizon
        self.df_filter = self.df[self.df['train_horizon'] == train_horizon]        
        
        # rank by the column "test_loss"
        self.df_filter = self.df_filter.sort_values(by=['test_loss'])
        
        # choose the top n_models
        self.df_filter = self.df_filter.iloc[:n_models]
        # get Name column
        self.model_names = self.df_filter['Name'].values
        
        

        # load the models
        # self.models = []
        # self.params = []
        # for model_name in self.model_names:
        #     model_path = "checkpoints/" + model_name + "/best_model.pkl"
        #     param_path = "checkpoints/" + model_name + "/best_params.pkl"
        #     with open(model_path, "rb") as f:
        #         model = pickle.load(f)
            
        #     with open(param_path, "rb") as f:
        #         param = pickle.load(f)
                
        #     self.models.append(model)
        #     self.params.append(param)
        self.models, self.params = jax.vmap(self._load_model)(self.model_names)
        return self.models, self.params

    @staticmethod
    @jax.jit
    def _load_model(model_name):
        model_path = "checkpoints/" + model_name + "/best_model.pkl"
        param_path = "checkpoints/" + model_name + "/best_params.pkl"
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        with open(param_path, "rb") as f:
            param = pickle.load(f)
            
        return model, param