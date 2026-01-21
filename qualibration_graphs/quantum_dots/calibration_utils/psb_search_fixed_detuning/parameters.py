from typing import Optional
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import BaseExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    n_points_slow: int = 101
    """Number of points for the slow axis (voltage sweep). Default is 101."""
    n_points_fast: int = 100
    """Number of points for the fast axis (voltage sweep). Default is 100."""
    voltage_span_slow: float = 3.0
    """Voltage span for the slow axis in volts. Default is 3.0 V."""
    voltage_center_slow: float = 0.0
    """Voltage center for the slow axis in volts. Default is 0.0 V."""
    voltage_span_fast: float = 3.0
    """Voltage span for the fast axis in volts. Default is 3.0 V."""
    voltage_center_fast: float = 0.0
    """Voltage center for the fast axis in volts. Default is 0.0 V."""
    coulomb_amp: float = 0.0
    """Amplitude of the Coulomb pulse. Default is 0.0."""
    duration_empty: int = 5000
    """Duration of the empty state in nanoseconds. Default is 5000 ns."""
    wait_after_voltage_settle: int = 300000
    """Wait time after voltage settle in nanoseconds. Default is 300000 ns (300 µs)."""
    detuning: Optional[float] = None
    """Fixed detuning value. If None, will use default from qubit configuration. Default is None."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    BaseExperimentNodeParameters,
):
    pass
