from typing import Literal

from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters

from calibration_utils.common_utils.experiment import QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of shots per (detuning, barrier gate voltage) point."""

    ramp_duration: int = 1000
    """Ramp duration for the conditional reset sequence."""
    buffer_duration: int = 524
    """Buffer duration before readout in the conditional reset sequence."""
    hold_duration: int = 1000
    """Hold duration after the conditional drive pulse."""

    detuning_center: float = 0.0
    """Center detuning value for the 2D sweep."""
    detuning_span: float = 0.4
    """Total detuning span for the 2D sweep."""
    detuning_step: float = 0.02
    """Detuning step for the 2D sweep."""

    barrier_gate_voltage_center: float = 0.0
    """Center barrier-gate voltage value for the 2D sweep."""
    barrier_gate_voltage_span: float = 0.2
    """Total barrier-gate voltage span for the 2D sweep."""
    barrier_gate_voltage_step: float = 0.01
    """Barrier-gate voltage step for the 2D sweep."""

    drive_frequency_detuning_MHz: float = 0.0
    """Fixed drive frequency detuning used by the reset pulse."""
    drive_amplitude_scale: float = 1.0
    """Fixed amplitude scale used by the reset pulse."""

    operation: Literal["x180", "x90"] = "x180"
    """Optional operation pulse applied in the with-op branch."""
    reset_operation: Literal["x180", "x90"] = "x180"
    """Reset pulse operation used inside conditional empty()."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for 07d reset-pulse detuning/barrier 2D calibration."""
