from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import QuantumDotExperimentNodeParameters

from typing import List, Optional


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    sensor_names: Optional[List[str]] = None
    """List of sensor names to include in the measurement. If None, all sensors are used."""
    v_center: Optional[float] = None
    """Center of the voltage sweep in volts. Default 0.0V."""
    v_span: float = 0.01
    """Span of the voltage sweep in volts. Default 10mV."""
    n_points: int = 101
    """Number of points in the voltage sweep. Default is 101."""
    dwell_time: int = 500
    """Dwell time at each voltage in nanoseconds. Default is 500ns."""
    ramp_duration: int = 100
    """The ramp duration to each voltage point. Default is 100ns."""
    max_compensation_voltage: float = 0.01
    """The maximum height of the compensation pulse. The duration will be calculated using this height."""


class Parameters(
    NodeParameters,
    QuantumDotExperimentNodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    pass
