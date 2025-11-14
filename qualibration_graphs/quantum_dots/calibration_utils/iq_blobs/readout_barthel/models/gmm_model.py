"""
Gaussian Mixture Model (GMM) implementations for readout data analysis.

This module provides NumPyro models for fitting Gaussian mixture models to
both 1D (projected voltage) and 2D (IQ) readout data. GMMs serve as:

1. **Baseline models**: Simple alternative to the physics-based Barthel model
2. **Initialization**: Provide good starting points for Barthel MCMC
3. **Model comparison**: Use BIC to compare GMM vs Barthel fits

The module includes:
- 1D GMM: K-component mixture for projected voltage coordinates
- 2D GMM: K-component mixture with diagonal covariance for IQ data
- BIC computation: Bayesian Information Criterion for model selection
- Log-likelihood evaluation: For posterior predictive checks

All models are designed for MCMC inference with NumPyro and include
flexible prior specifications.
"""

from typing import Dict, Optional, Tuple, Callable, Any

import jax.numpy as jnp
import jax.scipy.special as jsp
import numpyro
import numpyro.distributions as dist
from numpyro import plate, sample


def make_gmm_model_factory(K: int) -> Callable[[Optional[Dict[str, Any]]], Callable[[jnp.ndarray], None]]:
    """
    Create a factory for building K-component 1D Gaussian mixture models.

    Returns a model builder that can be configured with custom priors and
    used for MCMC inference. The GMM models data as a weighted mixture of
    K Gaussian components with unknown means, precisions, and mixture weights.

    Model structure:
        π ~ Dirichlet(concentration)
        μ_k ~ Normal(mu_loc[k], mu_scale[k]) for k=1..K
        τ_k ~ Gamma(sigma_conc[k], sigma_rate[k]) where σ_k = 1/√τ_k
        z_i ~ Categorical(π) for each observation i
        y_i ~ Normal(μ_{z_i}, σ_{z_i})

    Args:
        K: Number of Gaussian components in the mixture (typically 2 for S/T states)

    Returns:
        A model builder function that takes an optional priors dictionary and
        returns a NumPyro model suitable for MCMC sampling.

    Priors dictionary keys (all optional):
        - 'pi_conc': Dirichlet concentration parameters, shape (K,). Default: ones(K)
                    Higher values concentrate mass; uniform when all equal.
        - 'mu_loc': Prior means for component centers, shape (K,). Default: linspace(min, max, K)
        - 'mu_scale': Prior std for component centers, shape (K,). Default: 0.5 for all
        - 'sigma_conc': Gamma concentration for precisions, shape (K,). Default: 2.0 for all
        - 'sigma_rate': Gamma rate for precisions, shape (K,). Default: 2.0 for all

    Example:
        >>> # Build a 2-component GMM for singlet/triplet classification
        >>> builder = make_gmm_model_factory(K=2)
        >>> priors = {'mu_loc': jnp.array([-0.5, 0.5])}
        >>> model = builder(priors)
        >>> # Fit with MCMC
        >>> from numpyro.infer import MCMC, NUTS
        >>> mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=500)
        >>> mcmc.run(jax.random.PRNGKey(0), y=voltage_data)
    """

    def model_builder(priors: Optional[Dict] = None):
        priors_local = {} if priors is None else dict(priors)

        def model(y: jnp.ndarray):
            y = jnp.asarray(y, float)
            if y.ndim != 1:
                raise ValueError("1D GMM expects a flat array of observations.")
            N = y.shape[0]

            pi_conc = priors_local.get("pi_conc", jnp.ones(K))
            mu_loc = priors_local.get("mu_loc", jnp.linspace(y.min(), y.max(), K))
            mu_scale = priors_local.get("mu_scale", jnp.ones(K) * 0.5)
            sigma_conc = priors_local.get("sigma_conc", jnp.ones(K) * 2.0)
            sigma_rate = priors_local.get("sigma_rate", jnp.ones(K) * 2.0)

            pi = sample("pi", dist.Dirichlet(pi_conc))
            with plate("components", K):
                mu = sample("mu", dist.Normal(mu_loc, mu_scale))
                tau = sample("tau", dist.Gamma(sigma_conc, sigma_rate))
            sigma = 1.0 / jnp.sqrt(tau)

            with plate("data", N):
                z = sample("z", dist.Categorical(probs=pi))
                sample("obs", dist.Normal(loc=mu[z], scale=sigma[z]), obs=y)

        return model

    return model_builder


def build_gmm_model(priors: Optional[Dict[str, Any]] = None) -> Callable[[jnp.ndarray], None]:
    """
    Build a 2-component 1D Gaussian mixture model (convenience wrapper).

    Backward-compatible function that creates a 2-component GMM by delegating
    to make_gmm_model_factory(K=2). Use this for simple singlet/triplet fitting.

    Args:
        priors: Optional dictionary of prior hyperparameters (see make_gmm_model_factory)

    Returns:
        NumPyro model function for MCMC sampling

    Example:
        >>> model = build_gmm_model()
        >>> mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=500)
        >>> mcmc.run(jax.random.PRNGKey(0), y=voltage_data)
    """
    return make_gmm_model_factory(2)(priors)


def make_gmm_model_factory_2d(K: int = 2) -> Callable[[Optional[Dict[str, Any]]], Callable[[jnp.ndarray], None]]:
    """
    Create a factory for building K-component 2D Gaussian mixture models.

    Builds GMMs for 2D IQ data with diagonal covariance (independent I and Q
    variances within each component). Useful for fitting IQ measurements before
    PCA projection or as a baseline comparison to the Barthel model.

    Model structure:
        π ~ Dirichlet(concentration)
        μ_k ~ Normal(mu_loc[k], mu_scale[k]) for k=1..K, shape (2,) per component
        σ_k ~ HalfCauchy(sigma_scale[k]) for k=1..K, shape (2,) per component
        z_i ~ Categorical(π) for each observation i
        y_i ~ Normal(μ_{z_i}, diag(σ²_{z_i}))

    Args:
        K: Number of Gaussian components (default: 2 for singlet/triplet)

    Returns:
        A model builder function that takes an optional priors dictionary and
        returns a NumPyro model for 2D data.

    Priors dictionary keys (all optional):
        - 'pi_conc': Dirichlet concentration, shape (K,). Default: ones(K)
        - 'mu_loc': Component mean locations, shape (K, 2). Default: linspace from data min/max
        - 'mu_scale': Component mean prior scales, shape (K, 2). Default: 0.5 for all
        - 'sigma_scale': HalfCauchy scales for component std devs, shape (K, 2). Default: 0.5 for all

    Example:
        >>> # Build a 2-component 2D GMM for IQ data
        >>> builder = make_gmm_model_factory_2d(K=2)
        >>> priors = {
        ...     'mu_loc': jnp.array([[0.0, 0.0], [0.5, 0.1]]),  # S at origin, T offset
        ...     'sigma_scale': jnp.ones((2, 2)) * 0.3
        ... }
        >>> model = builder(priors)
        >>> mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=500)
        >>> mcmc.run(jax.random.PRNGKey(0), y=iq_data)  # iq_data: shape (N, 2)
    """

    def model_builder(priors: Optional[Dict] = None):
        priors_local = {} if priors is None else dict(priors)

        def model(y):
            y = jnp.asarray(y)
            if y.ndim != 2 or y.shape[1] != 2:
                raise ValueError("2D GMM expects data of shape (N, 2).")
            N = y.shape[0]

            pi_conc = priors_local.get("pi_conc", jnp.ones(K))
            mu_loc = priors_local.get("mu_loc", jnp.linspace(jnp.min(y, axis=0), jnp.max(y, axis=0), K))
            mu_scale = priors_local.get("mu_scale", jnp.ones((K, 2)) * 0.5)
            sigma_scale = priors_local.get("sigma_scale", jnp.ones((K, 2)) * 0.5)

            pi = sample("pi", dist.Dirichlet(pi_conc))
            with plate("components", K):
                mu = sample("mu", dist.Normal(mu_loc, mu_scale).to_event(1))
                sigma = sample("sigma", dist.HalfCauchy(sigma_scale).to_event(1))

            with plate("data", N):
                z = sample("z", dist.Categorical(probs=pi))
                loc = mu[z]
                scale = sigma[z]
                sample("obs", dist.Independent(dist.Normal(loc, scale), 1), obs=y)

        return model

    return model_builder

def log_likelihood_samples(
    x: jnp.ndarray,
    pi: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute log-likelihood of 1D data under a GMM for each posterior sample.

    Evaluates the log P(x | π, μ, σ) for a 1D Gaussian mixture model across
    multiple posterior samples. Useful for computing BIC, posterior predictive
    checks, and model comparison.

    The log-likelihood is computed via logsumexp for numerical stability:
        log P(x | θ) = Σᵢ log(Σₖ πₖ · N(xᵢ | μₖ, σₖ))

    Args:
        x: Observed data, shape (N,) where N is number of observations
        pi: Mixture weights from posterior samples, shape (S, K) or (K,)
           where S = number of samples, K = number of components
        mu: Component means from posterior samples, shape (S, K) or (K,)
        sigma: Component std devs from posterior samples, shape (S, K) or (K,)

    Returns:
        Log-likelihood for each sample, shape (S,). Each element is the total
        log-likelihood of all N observations under that parameter set.

    Example:
        >>> # After MCMC on 1D GMM
        >>> loglik = log_likelihood_samples(data, samples['pi'],
        ...                                 samples['mu'], samples['sigma'])
        >>> print(f"Max log-lik: {loglik.max():.2f}")
    """
    x = jnp.asarray(x, float)
    pi = jnp.asarray(pi, float)
    mu = jnp.asarray(mu, float)
    sigma = jnp.asarray(sigma, float)

    if pi.ndim == 1:
        pi = pi[None, :]
    if mu.ndim == 1:
        mu = mu[None, :]
    if sigma.ndim == 1:
        sigma = sigma[None, :]

    x_exp = x[None, None, :]
    log_component = (
        jnp.log(pi)[:, :, None]
        - 0.5 * jnp.log(2.0 * jnp.pi * sigma[:, :, None] ** 2)
        - ((x_exp - mu[:, :, None]) ** 2) / (2.0 * sigma[:, :, None] ** 2)
    )
    log_mix = jsp.logsumexp(log_component, axis=1)
    return log_mix.sum(axis=1)


def compute_bic(x: jnp.ndarray, samples: Dict[str, jnp.ndarray]) -> Tuple[float, float]:
    """
    Compute Bayesian Information Criterion (BIC) for 1D GMM model selection.

    BIC penalizes model complexity and favors simpler models:
        BIC = -2 * log L̂ + k * log(N)

    where L̂ is the maximum likelihood, k is the number of parameters,
    and N is the sample size. Lower BIC values indicate better models.

    For a K-component 1D GMM:
        k = 3K - 1 = (K-1 mixture weights) + (K means) + (K variances)

    Args:
        x: Observed 1D data, shape (N,)
        samples: Dictionary of posterior samples with keys:
                'pi': mixture weights, shape (S, K)
                'mu': component means, shape (S, K)
                'tau': precisions (1/σ²), shape (S, K)
                where S = number of MCMC samples

    Returns:
        Tuple of (bic, max_log_likelihood):
            bic: Bayesian Information Criterion (lower is better)
            max_log_likelihood: Maximum log-likelihood across posterior samples

    Example:
        >>> bic, loglik = compute_bic(voltage_data, gmm_samples)
        >>> print(f"BIC: {bic:.1f}, log-lik: {loglik:.1f}")
    """
    x = jnp.asarray(x, float)
    pi = jnp.asarray(samples["pi"])
    mu = jnp.asarray(samples["mu"])
    tau = jnp.asarray(samples["tau"])
    sigma = 1.0 / jnp.sqrt(tau)

    loglik_samples = log_likelihood_samples(x, pi, mu, sigma)
    loglik_hat = float(loglik_samples.max())

    K = pi.shape[-1] if pi.ndim > 1 else 1
    num_params = 3 * K - 1
    bic = -2.0 * loglik_hat + num_params * jnp.log(x.shape[0])
    return float(bic), loglik_hat


def log_likelihood_samples_2d_diag(
    x: jnp.ndarray,
    pi: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute log-likelihood of 2D IQ data under a diagonal-covariance GMM.

    Evaluates log P(x | π, μ, Σ) for 2D Gaussian mixtures with diagonal
    covariance (independent I and Q variances per component). Each component
    has a 2D mean and 2D std vector.

    The log-likelihood uses logsumexp for numerical stability:
        log P(x | θ) = Σᵢ log(Σₖ πₖ · N(xᵢ | μₖ, diag(σₖ²)))

    Args:
        x: Observed 2D IQ data, shape (N, 2) where N is number of observations
        pi: Mixture weights from posterior samples, shape (S, K) or (K,)
           where S = number of samples, K = number of components
        mu: Component means, shape (S, K, 2) or (K, 2)
        sigma: Component std devs (diagonal), shape (S, K, 2) or (K, 2)

    Returns:
        Log-likelihood for each sample, shape (S,). Each element is the total
        log-likelihood of all N observations under that parameter set.

    Example:
        >>> # After MCMC on 2D GMM
        >>> loglik = log_likelihood_samples_2d_diag(
        ...     iq_data, samples['pi'], samples['mu'], samples['sigma']
        ... )
        >>> print(f"Max log-lik: {loglik.max():.2f}")
    """
    x = jnp.asarray(x, float)
    pi = jnp.asarray(pi, float)
    mu = jnp.asarray(mu, float)
    sigma = jnp.asarray(sigma, float)

    if pi.ndim == 1:
        pi = pi[None, :]
    if mu.ndim == 2:
        mu = mu[None, :, :]
    if sigma.ndim == 2:
        sigma = sigma[None, :, :]

    x_exp = x[None, None, :, :]
    log_component = (
        jnp.log(pi)[:, :, None]
        - 0.5 * jnp.log(2.0 * jnp.pi * sigma**2).sum(axis=2)[:, :, None]
        - ((x_exp - mu[:, :, None, :]) ** 2 / (2.0 * sigma[:, :, None, :] ** 2)).sum(axis=3)
    )
    log_mix = jsp.logsumexp(log_component, axis=1)
    return log_mix.sum(axis=1)


def compute_bic_2d_diag(x: jnp.ndarray, samples: Dict[str, jnp.ndarray]) -> Tuple[float, float]:
    """
    Compute BIC for 2D diagonal-covariance GMM model selection.

    Similar to compute_bic but for 2D IQ data with diagonal covariance matrices.
    BIC formula: BIC = -2 * log L̂ + k * log(N)

    For a K-component 2D diagonal GMM:
        k = (K-1) + 2K + 2K = 5K - 1
        where: (K-1) mixture weights + 2K means (I,Q per component)
               + 2K variances (I,Q per component)

    Args:
        x: Observed 2D IQ data, shape (N, 2)
        samples: Dictionary of posterior samples with keys:
                'pi': mixture weights, shape (S, K)
                'mu': component means, shape (S, K, 2)
                'sigma': component std devs, shape (S, K, 2)
                where S = number of MCMC samples

    Returns:
        Tuple of (bic, max_log_likelihood):
            bic: Bayesian Information Criterion (lower is better)
            max_log_likelihood: Maximum log-likelihood across posterior samples

    Example:
        >>> bic, loglik = compute_bic_2d_diag(iq_data, gmm_2d_samples)
        >>> print(f"BIC: {bic:.1f}, log-lik: {loglik:.1f}")
    """
    x = jnp.asarray(x, float)
    pi = jnp.asarray(samples["pi"])
    mu = jnp.asarray(samples["mu"])
    sigma = jnp.asarray(samples["sigma"])

    loglik_samples = log_likelihood_samples_2d_diag(x, pi, mu, sigma)
    loglik_hat = float(loglik_samples.max())

    K = pi.shape[-1] if pi.ndim > 1 else pi.shape[0]
    num_params = (K - 1) + 2 * K + 2 * K
    bic = -2.0 * loglik_hat + num_params * jnp.log(x.shape[0])
    return float(bic), loglik_hat
