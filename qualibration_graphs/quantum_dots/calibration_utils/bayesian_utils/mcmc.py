"""Bayesian inference engine for fitting models via MCMC.

This module provides a high-level interface for running NumPyro's NUTS (No-U-Turn Sampler)
on probabilistic models. It handles MCMC configuration, initialization strategies, and
returns posterior samples for parameter estimation and uncertainty quantification.

The NUTS algorithm is a Hamiltonian Monte Carlo method that efficiently explores
high-dimensional posterior distributions using automatic step-size tuning and path length
selection. Use this for calibration fits (Rabi, Ramsey, etc.) where uncertainty
quantification is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp


def _import_numpyro():
    """Lazy import to avoid loading numpyro when not used."""
    import numpyro  # type: ignore[import-untyped]
    from numpyro.infer import MCMC, NUTS  # type: ignore[import-untyped]
    from numpyro.infer.initialization import init_to_median, init_to_value  # type: ignore[import-untyped]
    return numpyro, MCMC, NUTS, init_to_median, init_to_value


@dataclass
class MCMCConfig:
    """
    Configuration parameters for MCMC sampling with NUTS.

    Attributes
    ----------
    num_warmup : int
        Number of warmup (burn-in) iterations for adapting step size and mass matrix.
        During warmup, samples are discarded and used only for tuning.
    num_samples : int
        Number of posterior samples to draw after warmup. These are returned for inference.
    num_chains : int
        Number of independent MCMC chains. Multiple chains help assess convergence (R-hat).
    progress_bar : bool
        Whether to display a progress bar during sampling.
    dense_mass : bool
        If True, use a dense mass matrix (full covariance). If False, use diagonal mass.
        Dense is more flexible but slower for many parameters.
    target_accept_prob : float
        Target acceptance probability for step size adaptation (typically 0.8–0.9).
        Higher values give smaller steps and more accurate sampling but slower mixing.
    """

    num_warmup: int = 500
    num_samples: int = 500
    num_chains: int = 1
    progress_bar: bool = True
    dense_mass: bool = False
    target_accept_prob: float = 0.8


def fit_model(
    model_fn: Callable[[Dict[str, Any]], Callable[..., Any]],
    data: jnp.ndarray,
    *,
    priors: Optional[Dict[str, Any]] = None,
    config: Optional[MCMCConfig] = None,
    init_vals: Optional[Dict[str, Union[jnp.ndarray, float]]] = None,
    data_key: str = "y",
    seed: int = 0,
    print_summary: bool = True,
) -> Tuple[Any, Dict[str, jnp.ndarray], Dict[str, Dict[str, float]]]:
    """
    Fit a NumPyro probabilistic model using NUTS MCMC sampling.

    Runs Hamiltonian Monte Carlo (NUTS) to draw samples from the posterior distribution
    of model parameters given observed data. Returns posterior samples and a summary
    (mean, std, percentiles) for downstream calibration use.

    Workflow
    --------
    1. Initialize NUTS kernel with model and configuration
    2. Run MCMC warmup phase to adapt step size and mass matrix
    3. Draw posterior samples from the adapted chain
    4. Optionally print diagnostic summary (R-hat, ESS)
    5. Return MCMC object, samples, and summary statistics

    Parameters
    ----------
    model_fn : callable
        A factory that takes a dictionary of prior hyperparameters and returns a
        NumPyro model (callable). The model should define the generative process
        using ``numpyro.sample`` and ``numpyro.deterministic``, and accept the
        observed data via a keyword argument (default ``y``).
    data : jnp.ndarray
        Observed data to condition the model on. Passed to the model via
        ``{data_key: data}``.
    priors : dict, optional
        Prior hyperparameters passed to ``model_fn``. Empty dict if None.
    config : MCMCConfig, optional
        MCMC configuration. Uses defaults if None.
    init_vals : dict, optional
        Initial parameter values for warm-starting the sampler. Keys must match
        model parameter names. If None, initializes at posterior median.
    data_key : str, default "y"
        Keyword name for passing data to the model (e.g. ``mcmc.run(key, y=data)``).
    seed : int, default 0
        Random seed for reproducibility.
    print_summary : bool, default True
        Whether to print convergence diagnostics (R-hat, ESS, etc.).

    Returns
    -------
    mcmc : MCMC
        NumPyro MCMC object with full sampling state and diagnostics.
    samples : dict[str, jnp.ndarray]
        Posterior samples per parameter. Shape ``(num_samples * num_chains,)``
        when ``group_by_chain=False``.
    summary : dict[str, dict[str, float]]
        Per-parameter summary with keys ``mean``, ``std``, ``5%``, ``50%``, ``95%``.
        Useful for calibration outputs (e.g. ``optimal_frequency_mean``, ``optimal_frequency_std``).
    """
    numpyro, MCMC_cls, NUTS_cls, init_to_median, init_to_value = _import_numpyro()

    if config is None:
        config = MCMCConfig()
    if priors is None:
        priors = {}

    init = (
        init_to_value(values={k: jnp.asarray(v) for k, v in init_vals.items()})
        if init_vals
        else init_to_median()
    )

    model = model_fn(priors)
    kernel = NUTS_cls(
        model,
        dense_mass=config.dense_mass,
        target_accept_prob=config.target_accept_prob,
        init_strategy=init,
    )

    mcmc = MCMC_cls(
        kernel,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        progress_bar=config.progress_bar,
    )

    mcmc.run(jax.random.PRNGKey(seed), **{data_key: data})

    samples = mcmc.get_samples(group_by_chain=False)

    if print_summary:
        mcmc.print_summary(exclude_deterministic=False)

    summary = posterior_summary(samples)

    return mcmc, samples, summary


def posterior_summary(
    samples: Dict[str, jnp.ndarray],
    percentiles: Optional[Tuple[float, ...]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute summary statistics from posterior samples.

    Parameters
    ----------
    samples : dict[str, jnp.ndarray]
        Posterior samples per parameter. Arrays can be 1D (flattened) or 2D
        (chains × samples).
    percentiles : tuple of float, optional
        Percentiles to compute (e.g. (5, 50, 95)). Default ``(5, 50, 95)``.

    Returns
    -------
    summary : dict[str, dict[str, float]]
        Per-parameter dict with keys ``mean``, ``std``, and ``"5%"``, ``"50%"``, etc.
    """
    if percentiles is None:
        percentiles = (5.0, 50.0, 95.0)

    out: Dict[str, Dict[str, float]] = {}

    for name, arr in samples.items():
        flat = jnp.ravel(arr)
        out[name] = {
            "mean": float(jnp.mean(flat)),
            "std": float(jnp.std(flat)),
        }
        for p in percentiles:
            out[name][f"{p:.0f}%"] = float(jnp.percentile(flat, p))

    return out
