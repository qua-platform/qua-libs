from __future__ import annotations

from typing import Literal

from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    QubitPairExperimentNodeParameters,
)


class CMAESParameters(RunnableParameters):
    """Mixin carrying all CMA-ES hyper-parameters.

    Designed to be combined with node-specific parameters via multiple
    inheritance so that any optimisation node can reuse the same CMA-ES
    controls.
    """

    population_size: int = 10
    """CMA-ES population size (lambda). Default is 10."""
    sigma0: float = 0.01
    """Initial step-size for CMA-ES. Default is 0.01."""
    max_generations: int = 50
    """Maximum number of CMA-ES generations. Default is 50."""
    tolx: float = 1e-6
    """Convergence tolerance on parameter changes. Default is 1e-6."""
    tolfun: float = 1e-6
    """Convergence tolerance on function value changes. Default is 1e-6."""
    success_threshold: float = 0.5
    """Minimum best-score to consider the optimisation successful (default 0.5,
    i.e. better than a random binary guess)."""
    cmaes_log_each_generation: bool = True
    """If True, log a progress line after each CMA-ES generation (counter vs max)."""


class MeasurementFidelityParameters(RunnableParameters):
    """Parameters specific to the measurement-fidelity optimisation node."""

    num_shots: int = 200
    """Number of shots per candidate evaluation. Default is 200."""
    detuning_initial: float = 0.0
    """Starting detuning value in volts. Default is 0.0 V."""
    detuning_min: float = -0.1
    """Lower bound for detuning in volts. Default is -0.1 V."""
    detuning_max: float = 0.1
    """Upper bound for detuning in volts. Default is 0.1 V."""
    ramp_duration_initial: int = 100
    """Starting ramp duration in ns (must be multiple of 4). Default is 100."""
    ramp_duration_min: int = 16
    """Lower bound for ramp duration in ns. Default is 16."""
    ramp_duration_max: int = 2000
    """Upper bound for ramp duration in ns. Default is 2000."""
    barrier_voltage_initial: float = 0.0
    """Starting barrier gate voltage in volts. Default is 0.0 V."""
    barrier_voltage_min: float = -0.1
    """Lower bound for barrier voltage in volts. Default is -0.1 V."""
    barrier_voltage_max: float = 0.1
    """Upper bound for barrier voltage in volts. Default is 0.1 V."""
    buffer_duration: int = 16
    """Buffer duration at the measurement point before readout pulse (ns). Default is 16."""
    operation: Literal["readout", "readout_QND"] = "readout"
    """Resonator operation whose readout parameters are evaluated. Default is 'readout'."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    CMAESParameters,
    MeasurementFidelityParameters,
    QubitPairExperimentNodeParameters,
):
    """Composite parameter set for 01_optimize_measurement_fidelity."""
