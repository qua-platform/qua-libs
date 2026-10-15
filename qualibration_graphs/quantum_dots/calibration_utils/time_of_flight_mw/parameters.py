from typing import Optional, List
from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import BaseExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    sensor_names: Optional[List[str]] = None
    """List of SensorDot names to include in the measurement."""
    time_of_flight_in_ns: Optional[int] = 28
    """Time of flight in nanoseconds. Default is 28 ns."""
    readout_amplitude_in_dBm: Optional[float] = -12
    """Readout amplitude in dBm. Default is -12 dBm."""
    readout_length_in_ns: Optional[int] = 1000
    """Readout length in nanoseconds. Default is 1000 ns."""
    use_simulated_data: bool = False
    """Whether to generate simulated data instead of measuring via the OPX. Default False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    BaseExperimentNodeParameters,
):
    pass
