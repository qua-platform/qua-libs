from __future__ import annotations

from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    QubitPairExperimentNodeParameters,
)

from .parameters import CMAESParameters


class InitializationOptParameters(RunnableParameters):
    """Parameters specific to the single-ramp initialization visibility optimisation."""

    num_shots: int = 100
    """Number of shots per candidate evaluation. Default is 100."""

    # --- Initialization voltage point ---
    detuning_initial: float = -0.05
    """Starting detuning-axis voltage for the init point (V). Default is -0.05 V."""
    detuning_min: float = -0.2
    """Lower bound for detuning voltage (V). Default is -0.2 V."""
    detuning_max: float = 0.0
    """Upper bound for detuning voltage (V). Default is 0.0 V."""

    barrier_initial: float = 0.0
    """Starting barrier gate voltage for the init point (V). Default is 0.0 V."""
    barrier_min: float = -0.1
    """Lower bound for barrier voltage (V). Default is -0.1 V."""
    barrier_max: float = 0.1
    """Upper bound for barrier voltage (V). Default is 0.1 V."""

    # --- Ramp duration ---
    ramp_duration_initial: int = 500
    """Starting ramp duration in ns (0<->V segments). Default is 200."""
    ramp_duration_min: int = 200
    """Lower bound for ramp duration (ns). Default is 16."""
    ramp_duration_max: int = 2000
    """Upper bound for ramp duration (ns). Default is 2000."""

    # --- Hold duration ---
    hold_duration_initial: int = 500
    """Starting hold duration at the init point in ns. Default is 200."""
    hold_duration_min: int = 400
    """Lower bound for hold duration (ns). Default is 16."""
    hold_duration_max: int = 2000
    """Upper bound for hold duration (ns). Default is 2000."""

    compilation_timeout: float = 600.0
    """Timeout in seconds for the QMM gRPC API calls. Default is 600 s."""


class InitializationParameters(
    NodeParameters,
    CommonNodeParameters,
    CMAESParameters,
    InitializationOptParameters,
    QubitPairExperimentNodeParameters,
):
    """Composite parameter set for 02_optimize_initialization."""

    # InitializationOptParameters does not inherit from CMAESParameters, so
    # overrides to CMAESParameters fields must be placed here.
    # 5 % of each normalised [0, 1] range (after normalisation in the node)
    # gives adequate initial exploration for both voltage (mV-scale) and
    # timing (ns-scale) parameters.
    sigma0: float = 0.05
