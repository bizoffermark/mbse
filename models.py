import jax.numpy as jnp
from jax import grad, jit, vmap
import flax.linen as nn
import jax
import torch

# def train(x_train,y_train, n_horizon):
    
class MLP(nn.Module):
    in_feats: int
    hid_feats: int
    out_feats: int
    n_hidden: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hid_feats)(x)
        x = nn.swish(x)
        for i in range(self.n_hidden):
            x = nn.Dense(features=self.hid_feats)(x)
            x = nn.swish(x)
        x = nn.Dense(features=self.out_feats)(x)
        return x

