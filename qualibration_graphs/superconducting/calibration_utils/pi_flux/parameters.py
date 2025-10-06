from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters
from typing import Optional, Literal
from typing import List, Literal, Optional


class NodeSpecificParameters(RunnableParameters):
    """Parameters for 16a_long_cryoscope
    """
    num_shots: int = 30
    """Number of shots to acquire."""
    qubits: Optional[List[str]] = None
    """Qubits to calibrate."""
    operation: str = "x180_Gaussian"
    """Operation to perform."""
    operation_amplitude_factor: float = 1.0
    """Amplitude factor for the operation."""
    duration_in_ns: int = 5000
    """Duration of the operation in nanoseconds."""
    time_axis: Literal["linear", "log"] = "linear"
    """Time axis for the operation."""
    time_step_in_ns: int = 48
    """Time step in nanoseconds. For linear time axis."""
    time_step_num: int = 200
    """Number of time steps. Used for log time axis."""
    min_wait_time_in_ns: int = 32
    """Minimum wait time in nanoseconds."""
    frequency_span_in_mhz: float = 200
    """Frequency span in MHz."""
    frequency_step_in_mhz: float = 0.4
    """Frequency step in MHz."""
    flux_amp: float = 0.06
    """Flux amplitude in volts."""
    update_lo: bool = True
    """Whether to update the LO. This is useful if you detune your qubit frequency outside of the original band for the OPX1K"""
    fitting_base_fractions: List[float] = [0.4, 0.15, 0.05]
    """Base fractions for the fitting of the exponential sum."""
    update_state: bool = False
    """Whether to update the state."""
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    """Whether to use a joint or independent flux point for the fitting."""
    multiplexed: bool = False
    """Whether to use a multiplexed program."""
    reset_type_active_or_thermal: Literal["active", "thermal"] = "active"
    """Whether to use an active or thermal reset."""
    thermal_reset_extra_time_in_us: int = 10_000
    """Extra time in microseconds for the thermal reset."""
    use_state_discrimination: bool = False
    """Whether to use state discrimination. Requires a calibrated rotation angle and threshold."""



class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass