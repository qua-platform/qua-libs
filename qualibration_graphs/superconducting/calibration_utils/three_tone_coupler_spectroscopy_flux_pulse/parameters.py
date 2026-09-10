"""Parameters for three-tone coupler spectroscopy with a coupler flux pulse (22a)."""

from typing import ClassVar, Literal, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Three-tone spectroscopy at fixed coupler flux pulse amplitude."""

    num_shots: int = 1000
    """Number of averages per frequency point."""
    control_drive_operation: Literal["x180_Square", "x180"] = "x180_Square"
    """Control-qubit operation used to drive the coupler."""
    control_pulse_duration_in_ns: int = 500
    """Duration of the control drive pulse in ns."""
    control_pulse_amplitude: float = 0.1
    """Amplitude scale for the control drive pulse."""
    target_drive_operation: str = "saturation"
    """Weak probe operation on the target qubit."""
    target_pulse_amplitude: float = 0.02
    """Amplitude scale for the target probe pulse."""
    target_pulse_duration_in_ns: Optional[int] = 400
    """Target probe duration in ns; default uses the operation length from state."""
    frequency_span_in_mhz: float = 800.0
    """Total frequency span of the coupler drive sweep in MHz."""
    frequency_step_in_mhz: float = 1.0
    """Frequency step in MHz."""
    coupler_flux_in_v: float = 0.02
    """Coupler flux pulse amplitude in V (played relative to decouple_offset)."""
    coupler_flux_settle_in_ns: int = 25
    """Wait after the coupler flux pulse before the control drive, in ns."""
    rf_frequency_startpoint_in_hz: Optional[float] = 7.4e9
    """Optional coupler RF center in Hz; default uses ``coupler.RF_frequency`` from state."""
    update_state: bool = False
    """Write fitted ``coupler.RF_frequency`` into state when analysis succeeds."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for 22a three-tone coupler spectroscopy (flux pulse)."""

    targets_name: ClassVar[str] = "qubit_pairs"
