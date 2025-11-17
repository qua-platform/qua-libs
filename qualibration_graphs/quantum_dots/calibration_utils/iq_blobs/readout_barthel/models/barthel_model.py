"""
NumPyro probabilistic model for the Barthel two-state spin readout.

This module defines the Bayesian model used for MCMC inference with the Barthel
readout framework. The model uses analytic likelihood functions for efficient
sampling and supports:

- Informative or weakly-informative priors
- Ordering constraint μ_T > μ_S via positive delta parameterization
- Fixed or sampled measurement duration τ_M
- Triplet relaxation during readout (T₁ parameter)
- Mixture model for singlet/triplet populations

The model is designed to work with 1D projected voltage coordinates (after PCA
from IQ data) and accounts for the physical process where triplet states can
decay to singlet during the measurement window.

Mathematical model:
    p(y | θ) = (1-pT)·N(y; μ_S, σ) + pT·n_T(y; μ_S, μ_T, σ, T₁, τ_M)

where n_T is the Barthel triplet PDF with in-flight relaxation.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from typing import Dict, Any, Optional, Callable
from ..analytic import triplet_pdf_analytic, _norm_pdf


def build_barthel_model_1d_analytic(
    priors: Optional[Dict[str, Any]] = None,
    *,
    fix_tau_M: Optional[float] = None,
    enforce_order: bool = True,
) -> Callable[[jnp.ndarray], None]:
    """
    Build a NumPyro model for the Barthel 1D analytic readout with relaxation.

    Constructs a Bayesian probabilistic model that can be sampled with MCMC (NUTS)
    to infer the parameters of the Barthel two-state readout model. The model uses
    analytic likelihood functions for computational efficiency and numerical stability.

    Model parameters (sampled or deterministic):
    - μ_S: Singlet mean voltage
    - μ_T: Triplet mean voltage (or δ = μ_T - μ_S if enforce_order=True)
    - σ: Readout noise (standard deviation)
    - pT: Triplet fraction in the mixture (prior probability of T state)
    - T₁: Triplet relaxation time (LogNormal prior)
    - τ_M: Measurement duration (deterministic, fixed by fix_tau_M)

    The ordering constraint (enforce_order=True):
    Instead of sampling μ_T directly, samples δ ~ HalfNormal and sets μ_T = μ_S + δ,
    ensuring μ_T > μ_S. This removes the label-switching problem common in mixture models.

    Args:
        priors: Dictionary of prior hyperparameters. Supported keys:
               - 'mu_S_loc': Location for Normal prior on μ_S (default: 0.0)
               - 'mu_S_scale': Scale for Normal prior on μ_S (default: 1.0)
               - 'delta_scale': Scale for HalfNormal prior on δ (default: 1.0)
                               Only used when enforce_order=True
               - 'mu_T_loc': Location for Normal prior on μ_T (default: 1.0)
                            Only used when enforce_order=False
               - 'mu_T_scale': Scale for Normal prior on μ_T (default: 1.0)
                              Only used when enforce_order=False
               - 'sigma_scale': Scale for HalfNormal prior on σ (default: 0.3)
               - 'pT_conc0': Beta concentration α for pT (default: 1.0)
               - 'pT_conc1': Beta concentration β for pT (default: 1.0)
               - 'T1_logn_loc': LogNormal location for T₁ (default: log(2.0))
               - 'T1_logn_scale': LogNormal scale for T₁ (default: 0.35)
               - 'tauM': Measurement duration if fix_tau_M not provided (default: 1.0)
        fix_tau_M: Fixed measurement duration τ_M. If None, uses priors['tauM'] or 1.0.
        enforce_order: If True, enforce μ_T > μ_S via delta parameterization.
                      If False, sample μ_T directly (may have label switching).

    Returns:
        A NumPyro model function that takes 1D observed data y and performs
        Bayesian inference on the Barthel model parameters.

    Example:
        >>> # Build model with calibration priors
        >>> priors = {
        ...     'mu_S_loc': -0.5, 'mu_S_scale': 0.1,
        ...     'delta_scale': 0.3, 'sigma_scale': 0.05
        ... }
        >>> model = build_barthel_model_1d_analytic(priors, fix_tau_M=1.0)
        >>> # Use with MCMC
        >>> from numpyro.infer import MCMC, NUTS
        >>> kernel = NUTS(model)
        >>> mcmc = MCMC(kernel, num_warmup=500, num_samples=500)
        >>> mcmc.run(jax.random.PRNGKey(0), y=voltage_data)
    """
    if priors is None:
        priors = {}

    # Means (normalized units)
    mu_S_loc   = priors.get("mu_S_loc", 0.0)
    mu_S_scale = priors.get("mu_S_scale", 1.0)
    delta_scale = priors.get("delta_scale", priors.get("mu_T_scale", 1.0))
    mu_T_loc   = priors.get("mu_T_loc", 1.0)   # only if enforce_order=False
    mu_T_scale = priors.get("mu_T_scale", 1.0)

    # Scales / mixture / times (use floors to avoid degenerate priors in float32)
    sigma_scale   = priors.get("sigma_scale", 0.3)
    pT_conc0      = priors.get("pT_conc0", 1.0)
    pT_conc1      = priors.get("pT_conc1", 1.0)
    T1_logn_loc   = priors.get("T1_logn_loc", jnp.log(2.0))
    T1_logn_scale = priors.get("T1_logn_scale", 0.35)
    tauM_value    = float(priors.get("tauM", fix_tau_M if fix_tau_M is not None else 1.0))

    def model(y: jnp.ndarray) -> None:
        """
        Inner NumPyro model function for MCMC sampling.

        This function defines the probabilistic graphical model and is called
        by NumPyro's MCMC sampler. It samples latent parameters and computes
        the log-likelihood of the observed data.

        Args:
            y: Observed 1D voltage measurements, shape (N,)

        Raises:
            ValueError: If y is not 1D
        """
        y = jnp.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be 1D.")

        # === Sample latent parameters ===

        # 1. Singlet and triplet mean voltages
        # Sample singlet mean from Normal prior
        mu_S = numpyro.sample("mu_S", dist.Normal(mu_S_loc, mu_S_scale))

        if enforce_order:
            # Enforce μ_T > μ_S: sample positive offset δ and compute μ_T = μ_S + δ
            # This prevents label switching in the mixture model
            delta = numpyro.sample("delta", dist.HalfNormal(delta_scale))
            mu_T = numpyro.deterministic("mu_T", mu_S + delta)
        else:
            # Sample μ_T independently (may suffer from label switching)
            mu_T = numpyro.sample("mu_T", dist.Normal(mu_T_loc, mu_T_scale))

        # 2. Readout noise standard deviation
        # Add tiny epsilon for numerical stability in likelihood computation
        sigma = numpyro.sample("sigma", dist.HalfNormal(sigma_scale))
        sigma_used = sigma + 1e-6  # Avoid division by zero in PDF evaluation

        # 3. Mixture weight: prior probability of triplet state
        # Beta(1, 1) is uniform; calibration data provides informative priors
        pT = numpyro.sample("pT", dist.Beta(pT_conc0, pT_conc1))

        # 4. Relaxation time and measurement duration
        # T₁: Triplet → singlet relaxation time (LogNormal is positive-constrained)
        T1 = numpyro.sample("T1", dist.LogNormal(T1_logn_loc, T1_logn_scale))
        # τ_M: Fixed measurement duration (deterministic site for tracking)
        tauM = numpyro.deterministic("tauM", jnp.asarray(tauM_value))

        # === Compute likelihood ===

        # Probability that triplet doesn't decay during measurement
        p_no = jnp.exp(-tauM / T1)

        # Mixture model: weighted sum of singlet and triplet PDFs
        # Singlet: simple Gaussian at μ_S
        n_S = (1.0 - pT) * _norm_pdf(y, mu_S, sigma_used)
        # Triplet: Barthel PDF accounting for in-flight relaxation
        n_T = pT * triplet_pdf_analytic(y, mu_S, mu_T, sigma_used, T1, tauM)
        n_tot = n_S + n_T

        # Robust log-likelihood computation (numerically stable for float32)
        # Clip to avoid log(0) and handle NaN/Inf gracefully
        n_tot = jnp.where(jnp.isfinite(n_tot) & (n_tot > 0.0), n_tot, 0.0)
        log_n_tot = jnp.log(jnp.clip(n_tot, a_min=1e-300))
        log_n_tot = jnp.where(jnp.isfinite(log_n_tot), log_n_tot, -1e30)

        # Add log-likelihood factor to the model (replaces `obs` parameter)
        numpyro.factor("obs_loglik", jnp.sum(log_n_tot))

        # Track probability of no decay as a diagnostic
        numpyro.deterministic("p_no_decay", p_no)

    return model
