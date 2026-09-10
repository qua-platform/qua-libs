"""
Bayesian inference engine for fitting models via MCMC.

This module provides a high-level interface for running NumPyro's NUTS (No-U-Turn Sampler)
on probabilistic models. It handles MCMC configuration, initialization strategies, and
returns posterior samples for parameter estimation and uncertainty quantification.

The NUTS algorithm is a Hamiltonian Monte Carlo method that efficiently explores
high-dimensional posterior distributions using automatic step-size tuning and path length
selection.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Tuple
import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value, init_to_median


@dataclass
class MCMCConfig:
    """
    Configuration parameters for MCMC sampling with NUTS.

    Attributes:
        num_warmup: Number of warmup (burn-in) iterations for adapting step size and mass matrix.
                   During warmup, samples are discarded and used only for tuning.
        num_samples: Number of posterior samples to draw after warmup. These are returned for inference.
        num_chains: Number of independent MCMC chains to run. Multiple chains help assess convergence.
        progress_bar: Whether to display a progress bar during sampling.
        dense_mass: If True, use a dense mass matrix (full covariance). If False, use diagonal mass
                   (assumes independence between parameters). Dense is more flexible but slower.
        target_accept_prob: Target acceptance probability for step size adaptation (typically 0.8-0.9).
                           Higher values give smaller steps and more accurate sampling but slower mixing.
    """

    num_warmup: int = 500
    num_samples: int = 500
    num_chains: int = 1
    progress_bar: bool = True
    dense_mass: bool = False
    target_accept_prob: float = 0.8


def fit_model(
    model_fn: Callable[[Dict[str, Any]], Any],
    data: jnp.ndarray,
    priors: Optional[Dict[str, Any]] = None,
    config: Optional[MCMCConfig] = None,
    init_vals: Optional[Dict[str, jnp.ndarray]] = None,
) -> Tuple[MCMC, Dict[str, jnp.ndarray], MCMC]:
    """
    Fit a NumPyro probabilistic model using NUTS MCMC sampling.

    Runs Hamiltonian Monte Carlo (NUTS) to draw samples from the posterior distribution
    of model parameters given observed data. The function handles MCMC configuration,
    initialization, and returns posterior samples for downstream analysis.

    Workflow:
    1. Initialize NUTS kernel with model and configuration
    2. Run MCMC warmup phase to adapt step size and mass matrix
    3. Draw posterior samples from the adapted chain
    4. Print diagnostic summary (convergence, effective sample size, R-hat)
    5. Return MCMC object, samples, and summary

    Args:
        model_fn: A callable that takes a dictionary of priors and returns a NumPyro model.
                 The model should define the probabilistic generative process using
                 numpyro.sample and numpyro.deterministic.
        data: Observed data array (e.g., 1D projected voltages) to condition the model on.
             Passed to model via the 'y' keyword argument.
        priors: Optional dictionary of prior hyperparameters (e.g., {'mu_S_loc': 0.0}).
               Passed to model_fn to configure priors. Empty dict if None.
        config: MCMC configuration (warmup, samples, chains, etc.). Uses defaults if None.
        init_vals: Optional dictionary of initial parameter values for warm-starting the sampler.
                  If None, initializes at posterior median (computed via prior samples).

    Returns:
        Tuple of (mcmc, samples, summary):
            mcmc: NumPyro MCMC object with full sampling state and diagnostics
            samples: Dictionary mapping parameter names to posterior samples,
                    shape (num_samples * num_chains,) per parameter
            summary: Same as mcmc object (for backward compatibility)

    Example:
        >>> from readout_barthel.models.barthel_model import build_barthel_model_1d_analytic
        >>> # Define model factory
        >>> model_fn = lambda p: build_barthel_model_1d_analytic(p, fix_tau_M=1.0)
        >>> # Run MCMC
        >>> mcmc, samples, _ = fit_model(model_fn, y_data, priors={'sigma_scale': 0.3})
        >>> print(samples.keys())  # ['mu_S', 'mu_T', 'sigma', 'T1', 'pT', ...]
        >>> print(samples['mu_S'].shape)  # (500,) if num_samples=500, num_chains=1
    """
    # Use default config if not provided
    if config is None:
        config = MCMCConfig()
    if priors is None:
        priors = {}

    # Choose initialization strategy
    init = init_to_value(values=init_vals) if init_vals is not None else init_to_median()

    # Build NUTS kernel with the model
    kernel = NUTS(
        model_fn(priors),
        dense_mass=config.dense_mass,
        target_accept_prob=config.target_accept_prob,
        init_strategy=init,
    )

    # Create MCMC sampler
    mcmc = MCMC(
        kernel,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        progress_bar=config.progress_bar,
    )

    # Run sampling (y= is the observed data passed to the model)
    mcmc.run(jax.random.PRNGKey(0), y=data)

    # Extract posterior samples (flatten chains by default)
    samples = mcmc.get_samples(group_by_chain=False)

    # Print convergence diagnostics (R-hat, ESS, etc.)
    mcmc.print_summary(exclude_deterministic=False)

    return mcmc, samples, mcmc
