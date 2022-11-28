import os
import sys
from functools import partial
from math import comb
from typing import Any, Dict, List, Sequence, Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax import jit, vmap, random
from jax.tree_util import tree_flatten, tree_leaves


class MLP(nn.Module):
    features: Sequence[int]
    output_dim: int
    non_linearity: Callable = nn.swish

    @nn.compact
    def __call__(self, x, train=False):
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = self.non_linearity(x)
        if self.output_dim is not None:
            x = nn.Dense(features=self.output_dim)(x)
        return x


@jax.vmap
def mse(x, y):
    return jnp.square(x-y).mean()

