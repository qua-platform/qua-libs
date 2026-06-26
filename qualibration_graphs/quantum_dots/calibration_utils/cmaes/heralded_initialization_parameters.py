from __future__ import annotations

from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters as CommonHeraldedInitializeParameters,
    QubitPairExperimentNodeParameters,
)

from .parameters import CMAESParameters


class HeraldedInitializationOptParameters(RunnableParameters):
    """Parameters specific to the heralded initialization optimisation."""

    num_shots: int = 600
    """Number of shots per candidate evaluation."""

    # --- Init ramp duration ---
    init_ramp_duration_min: int = 20
    """Lower bound for init ramp duration (ns)."""
    init_ramp_duration_max: int = 10000
    """Upper bound for init ramp duration (ns)."""

    # --- Init hold duration ---
    init_hold_duration_min: int = 20
    """Lower bound for init hold duration (ns)."""
    init_hold_duration_max: int = 10000
    """Upper bound for init hold duration (ns)."""

    # --- Measure ramp duration ---
    meas_ramp_duration_min: int = 200
    """Lower bound for measure ramp duration (ns)."""
    meas_ramp_duration_max: int = 10000
    """Upper bound for measure ramp duration (ns)."""

    # --- Measure buffer duration ---
    meas_buffer_duration_min: int = 20
    """Lower bound for measure buffer duration (ns)."""
    meas_buffer_duration_max: int = 2000
    """Upper bound for measure buffer duration (ns)."""

    log_norm: bool = True
    """If True, normalise timing parameters in log-space so CMA-ES takes equal
    multiplicative steps across the range. If False, use linear normalisation."""

    compilation_timeout: float = 600.0
    """Timeout in seconds for the QMM gRPC API calls."""


class HeraldedInitializationParameters(
    NodeParameters,
    CommonNodeParameters,
    CMAESParameters,
    HeraldedInitializationOptParameters,
    QubitPairExperimentNodeParameters,
):
    """Composite parameter set for 02b_optimize_heralded_initialize."""

    # CMAESParameters.sigma0 defaults to 0.01. Override here on the composite
    # class so the MRO resolves correctly (same pattern as InitializationParameters).
    # 0.2 = 20 % of the normalised [0, 1] range for each timing parameter.
    sigma0: float = 0.2
    population_size: int = 20
    max_generations: int = 400
    success_threshold: float = 0.95