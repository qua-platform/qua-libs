"""Parameters for three-tone coupler spectroscopy vs coupler flux (22b)."""

from typing import ClassVar, Literal, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Three-tone 2D spectroscopy sweeping coupler DC flux and drive frequency."""

    num_shots: int = 200
    """Number of averages per (flux, frequency) point."""
    control_drive_operation: Literal["x180_Square", "x180"] = "x180_Square"
    """Control-qubit operation used to drive the coupler."""
    control_pulse_duration_in_ns: int = 800
    """Duration of the control drive pulse in ns."""
    control_pulse_amplitude: float = 0.2
    """Amplitude scale for the control drive pulse."""
    target_drive_operation: str = "saturation"
    """Weak probe operation on the target qubit."""
    target_pulse_amplitude: float = 0.005
    """Amplitude scale for the target probe pulse."""
    target_pulse_duration_in_ns: Optional[int] = 1000
    """Target probe duration in ns; default uses the operation length from state."""
    frequency_span_in_mhz: float = 900.0
    """Total frequency span of the coupler drive sweep in MHz."""
    frequency_step_in_mhz: float = 2.0
    """Frequency step in MHz."""
    coupler_flux_min_in_v: float = 0.0
    """Minimum coupler DC flux bias in V."""
    coupler_flux_max_in_v: float = 0.3
    """Maximum coupler DC flux bias in V."""
    num_coupler_flux_points: int = 51
    """Number of coupler flux points."""
    coupler_flux_settle_in_ns: int = 4000
    """Wait after updating coupler DC offset before the pulse sequence, in ns."""
    rf_frequency_startpoint_in_hz: Optional[float] = 7.2e9
    """Optional coupler RF center in Hz; default uses ``coupler.RF_frequency`` from state."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for 22b three-tone coupler spectroscopy vs flux."""

    targets_name: ClassVar[str] = "qubit_pairs"
