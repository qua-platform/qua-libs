"""Base utilities for Bayesian inference with NumPyro.

Provides a high-level MCMC interface for fitting probabilistic models via NUTS.
Other calibration nodes (e.g. time_rabi_chevron_parity_diff, ramsey) can define
their own NumPyro models and use these utilities for posterior sampling.

Requires the ``bayesian`` optional dependency: ``uv sync --extra bayesian``
"""

from calibration_utils.bayesian_utils.mcmc import (
    MCMCConfig,
    fit_model,
    posterior_summary,
)

__all__ = [
    "MCMCConfig",
    "fit_model",
    "posterior_summary",
]
