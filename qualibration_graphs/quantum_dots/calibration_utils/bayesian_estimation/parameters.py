from typing import Optional

from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    QubitsExperimentNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 1
    """Number of repetitions (outer shot loop). Default is 1."""

    derive_idle_times: bool = True
    """If True, set Ramsey delays from ``detuning``, ``f_span_in_MHz``, ``f_step_in_MHz`` and the Nyquist rule.
    If False, use ``min_wait_time_in_ns`` / ``max_wait_time_in_ns`` / ``wait_time_step_in_ns`` instead."""

    nyquist_margin: float = 20.0
    """Require f_Nyquist ≥ nyquist_margin × max|f| on the hypothesis grid, with f_Nyquist_MHz = 500/Δτ_ns
    for uniform delay steps Δτ (see ``idle_grid``). Default is 20.0 (strict headroom: Nyquist frequency
    must be at least 20× the grid bandwidth max|f|)."""

    min_wait_time_in_ns: int = 16
    """Minimum Ramsey delay (ns), multiple of 4. Used as floor and (in manual mode) sweep start."""
    max_wait_time_in_ns: int = 5_000
    """Maximum Ramsey delay (ns). When ``derive_idle_times`` is True, caps ``tau_max`` after the bound
    from ``f_step_in_MHz`` (see ``idle_grid.max_idle_ns_from_frequency_step``); default 25_000 ns matches
    that bound for the default ``f_step_in_MHz`` (0.02 MHz) and oversampling 0.5. In manual mode
    (``derive_idle_times`` False), sweep end."""
    wait_time_step_in_ns: int = 16
    """Delay step (ns) when ``derive_idle_times`` is False; multiple of 4."""

    detuning: float = 0.05
    """detuning from Qubit Larmor frequency in MHz."""
    f_span_in_MHz: float = 0.05
    """Half-width of the hypothesis grid in MHz: frequencies run from ``detuning - f_span`` to ``detuning + f_span``."""
    f_step_in_MHz: float = 0.005
    """Hypothesis grid step in MHz. Also sets a coarse upper bound on ``tau_max`` via frequency resolution."""

    alpha: float = 0.1
    """Bayesian update coefficient α in (0.5 + rk·(α + β·C)). Default 0.1."""
    beta: float = 0.9
    """Bayesian update coefficient β in (0.5 + rk·(α + β·C)). Default 0.9."""

    synthetic_example_f_mhz: Optional[float] = None
    """When set, skip QOP execute/simulate and fill ``ds_raw`` with Python synthetic Ramsey data
    at this true offset frequency (MHz), for development and plotting without hardware."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for 20a_BayesianEstimation."""

    pass
