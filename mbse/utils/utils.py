import jax.numpy as jnp
from jax import jit
import jax
from jax.scipy.stats import multivariate_normal

EPS = 1e-6


@jit
def gaussian_log_likelihood(x, mu, sig):
    log_sig = jnp.log(sig)
    log_l = -0.5 * (2 * log_sig + jnp.log(2*jnp.pi)
                     + jnp.square((x - mu)/(sig + EPS)))
    log_l = jnp.sum(log_l, axis=-1)
    return log_l


@jit
def sample_normal_dist(mu, sig, rng):
    return mu + jax.random.normal(rng, mu.shape)*sig


@jit
def rbf_kernel(x, y, bandwidth=None):
  square_sum = lambda x,y: jnp.sum(jnp.square(x-y))
  pairwise = jax.vmap(lambda y: jax.vmap(lambda x: square_sum(x, y), in_axes=0, out_axes=0)(x))(y)
  n_x = x.shape[-2]
  if bandwidth is None:
      bandwidth = jnp.median(pairwise)
  bandwidth = 0.5 * bandwidth / jnp.log(n_x + 1)
  #bandwidth = jnp.maximum(jax.lax.stop_gradient(bandwidth), 1e-5)
  bandwidth = jnp.maximum(jax.lax.stop_gradient(bandwidth), 1e-5)
  k_xy = jnp.exp(-pairwise / bandwidth / 2)
  return k_xy