from typing import Sequence, Callable, Union
import jax.numpy as jnp
from flax import linen as nn
from jax import jit, vmap, value_and_grad
from mbse.utils.network_utils import MLP
import jax
from functools import partial
from mbse.utils.utils import gaussian_log_likelihood, rbf_kernel
import optax

EPS = 1e-6


class ProbabilisticEnsembleModel(object):

    def __init__(
            self,
            example_input: jnp.ndarray,
            num_ensemble: int = 10,
            features: Sequence[int] = [256, 256],
            output_dim: int = 1,
            non_linearity: Callable = nn.swish,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            seed: int = 0,
            sig_min: float = 1e-3,
            sig_max: float = 1e3,
    ):
        self.output_dim = output_dim
        self.mlp = MLP(
            features=features, 
            output_dim=2*output_dim,
            non_linearity=non_linearity
        )
        self.num_ensembles = num_ensemble
        # vmap init function with respect to seed sequence
        init = vmap(self.mlp.init, (0, None))
        self.net = self.mlp.apply
        self.rng = jax.random.PRNGKey(seed)
        seed_sequence = jax.random.split(self.rng, self.num_ensembles+1)
        self.rng = seed_sequence[0]
        seed_sequence = seed_sequence[1:]
        particles = init(seed_sequence, example_input)
        self.sig_min = sig_min
        self.sig_max = sig_max
        self.optimizer = optax.adamw(learning_rate=lr,
                                     weight_decay=weight_decay)
        optimizer_state = self.optimizer.init(particles)
        self.particles = particles
        self.opt_state = optimizer_state
        self.example_input = example_input

    @property
    def params(self):
        return self.particles

    @partial(jit, static_argnums=0)
    def _predict(self, params, x):
        forward = jax.vmap(self.net, (0, None))
        predictions = forward(params, x)
        mu, sig = jnp.split(predictions, 2, axis=-1)
        sig = nn.softplus(sig)
        sig = jnp.clip(sig, 0, self.sig_max) + self.sig_min
        predictions = jnp.concatenate([mu, sig], axis=-1)
        return predictions

    def predict(self, x):
        return self._predict(self.particles, x)

    @partial(jit, static_argnums=0)
    def _train_step(self, params, opt_state, x, y, prior_particles=None):
        likelihood = jax.vmap(gaussian_log_likelihood, in_axes=(None, 0, 0), out_axes=0)

        def likelihood_loss(model_params):
            predictions = self._predict(model_params, x)
            mu, sig = jnp.split(predictions, 2, axis=-1)
            logl = likelihood(y, mu, sig)
            return -logl.mean()
        # vmap over ensemble
        loss, grads = value_and_grad(likelihood_loss)(params)
        updates, new_opt_state = self.optimizer.update(grads,
                                                             opt_state,
                                                             params=params)
        new_params = optax.apply_updates(params, updates)
        grad_norm = optax.global_norm(grads)
        return new_params, new_opt_state, loss, grad_norm

    def train_step(self, x, y):
        new_params, new_opt_state, loss, grad_norm = self._train_step(
            params=self.particles,
            opt_state=self.opt_state,
            x=x,
            y=y
        )
        self.particles = new_params
        self.opt_state = new_opt_state
        return loss, grad_norm


class fSVGDEnsemble(ProbabilisticEnsembleModel):
    def __init__(self,
                 n_prior_particles: Union[int, None] = None,
                 prior_bandwidth: float = 0.1,
                 k_bandwidth: float = 0.1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_prior_particles = n_prior_particles or self.num_ensembles
        init = vmap(self.mlp.init, (0, None))
        seed_sequence = jax.random.split(self.rng, n_prior_particles+1)
        self.rng = seed_sequence[0]
        seed_sequence = seed_sequence[1:]
        self.priors = init(seed_sequence, self.example_input)
        self.prior_bandwidth = prior_bandwidth
        self.k_bandwidth = k_bandwidth

    def _prior(self, prior_particles, x):
        predictions = self._predict(prior_particles, x)
        altered_predictions = predictions
        altered_predictions = altered_predictions.at[..., self.output_dim:].set(
            jnp.log(altered_predictions[..., self.output_dim:] + EPS))
        var = jax.vmap(lambda x: jnp.cov(x, rowvar=False),
                       in_axes=-1,
                       out_axes=-1)(altered_predictions)
        mean = jnp.mean(altered_predictions, axis=0)
        return mean, var

    @partial(jit, static_argnums=0)
    def _train_step(self, params, opt_state, x, y, prior_particles, rng):
        # mean_prior, k_prior = self._prior(prior_particles, x)
        rbf = lambda z,v: rbf_kernel(z, v, bandwidth=self.prior_bandwidth)
        kernel = lambda x: rbf(x, x) #K(x, x)
        k_prior = kernel(x)
        k_prior = jnp.stack([k_prior, k_prior], axis=-1)

        k_rbf = lambda z, v: rbf_kernel(z, v, bandwidth=self.k_bandwidth)

        def fsvgdloss(model_params):
            predictions, pred_vjp = jax.vjp(lambda p: self._predict(p, x), model_params)
            # k_pred, k_pred_vjp = jax.vjp(
        # lambda x: vmap(kernel, in_axes=-1, out_axes=-1)(x), predictions)
            k_pred, k_pred_vjp = jax.vjp(
                lambda x: k_rbf(x, predictions), predictions)
            grad_k = k_pred_vjp(-jnp.ones(k_pred.shape))[0]

            def neg_log_post(predictions):
                mean_pred, std_pred = jnp.split(predictions, 2, axis=-1)
                log_post = gaussian_log_likelihood(y, mean_pred, std_pred)
                return -log_post.mean()

            likelihood = lambda x, cov_x: \
                jax.scipy.stats.multivariate_normal.logpdf(x,
                                                           mean=jnp.zeros(x.shape[0]),
                                                           cov=cov_x + 1e-4*jnp.eye(x.shape[0]))
            likelihood = jax.vmap(likelihood, in_axes=-1, out_axes=-1)

            def neg_log_prior(predictions):
                mean_pred, std_pred = jnp.split(predictions, 2, axis=-1)
                log_sigma = jnp.log(std_pred + EPS)
                altered_predictions = jnp.stack([mean_pred, log_sigma], axis=-1)
                log_prior = likelihood(altered_predictions, k_prior)
                return -log_prior.mean()/mean_pred.shape[-2]

            def neg_total_likelihood(predictions):
                log_post = neg_log_post(predictions)
                log_pior = neg_log_prior(predictions)
                return log_post + log_pior

            log_post, log_posterior_grad = jax.vmap(value_and_grad(neg_total_likelihood, 0))(predictions)
            stein_grad = (jnp.einsum('ij,jkm', k_pred, log_posterior_grad)
                          + grad_k)
            grad = pred_vjp(stein_grad)[0]
            return log_post.mean(), grad

        loss, grads = fsvgdloss(params)
        updates, new_opt_state = self.optimizer.update(grads,
                                                       opt_state,
                                                       params=params)
        new_params = optax.apply_updates(params, updates)
        grad_norm = optax.global_norm(grads)
        return new_params, new_opt_state, loss, grad_norm

    def train_step(self, x, y):
        self.rng, train_rng = jax.random.split(self.rng)
        new_params, new_opt_state, loss, grad_norm = self._train_step(
            params=self.particles,
            opt_state=self.opt_state,
            x=x,
            y=y,
            prior_particles=self.priors,
            rng=train_rng
        )
        self.particles = new_params
        self.opt_state = new_opt_state
        return loss, grad_norm








