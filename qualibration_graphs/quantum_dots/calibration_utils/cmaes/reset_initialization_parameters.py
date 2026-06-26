from __future__ import annotations

from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    QubitPairExperimentNodeParameters,
)

from .parameters import CMAESParameters


class ResetInitializationOptParameters(RunnableParameters):
    """Parameters specific to the conditional-drive reset initialisation optimisation."""

    num_shots: int = 100
    """Number of shots per candidate evaluation."""

    # --- x pulse amplitude scale ---
    x_amplitude_initial: float = 1.0
    """Starting amplitude scale for the conditional x pulse. Default is 1.0 (no change)."""
    x_amplitude_min: float = 0.1
    """Lower bound for amplitude scale. Default is 0.1."""
    x_amplitude_max: float = 2.0
    """Upper bound for amplitude scale. Default is 2.0."""

    # --- IF detuning of the drive (Hz) ---
    if_detuning_initial: int = 0
    """Starting IF detuning in Hz from nominal. Default is 0."""
    if_detuning_min: int = -10_000_000
    """Lower bound for IF detuning (Hz). Default is -10 MHz."""
    if_detuning_max: int = 10_000_000
    """Upper bound for IF detuning (Hz). Default is +10 MHz."""

    # --- Ramp duration ---
    ramp_duration_initial: int = 500
    """Starting ramp duration in ns. Default is 500."""
    ramp_duration_min: int = 200
    """Lower bound for ramp duration (ns). Default is 200."""
    ramp_duration_max: int = 2000
    """Upper bound for ramp duration (ns). Default is 2000."""

    # --- Hold duration ---
    hold_duration_initial: int = 500
    """Starting hold duration at the init point in ns. Default is 500."""
    hold_duration_min: int = 200
    """Lower bound for hold duration (ns). Default is 200."""
    hold_duration_max: int = 2000
    """Upper bound for hold duration (ns). Default is 2000."""

    # --- Barrier gate voltage ---
    barrier_initial: float = 0.0
    """Starting barrier gate voltage for the init point (V). Default is 0.0 V."""
    barrier_min: float = -0.1
    """Lower bound for barrier voltage (V). Default is -0.1 V."""
    barrier_max: float = 0.1
    """Upper bound for barrier voltage (V). Default is 0.1 V."""

    # --- Detuning-axis voltage ---
    detuning_initial: float = -0.05
    """Starting detuning-axis voltage for the init point (V). Default is -0.05 V."""
    detuning_min: float = -0.2
    """Lower bound for detuning voltage (V). Default is -0.2 V."""
    detuning_max: float = 0.0
    """Upper bound for detuning voltage (V). Default is 0.0 V."""

    compilation_timeout: float = 600.0
    """Timeout in seconds for the QMM gRPC API calls. Default is 600 s."""

    # Parameters are normalised to [0, 1] before CMA-ES, so sigma0 is in
    # normalised units (0.2 = 20 % of each parameter's physical range).
    sigma0: float = 0.2


class ResetInitializationParameters(
    NodeParameters,
    CommonNodeParameters,
    CMAESParameters,
    ResetInitializationOptParameters,
    QubitPairExperimentNodeParameters,
):
    """Composite parameter set for 02a_optimize_reset_initialize."""