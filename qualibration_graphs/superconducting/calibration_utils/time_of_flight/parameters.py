from typing import Optional
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    time_of_flight_in_ns: Optional[int] = None
    """Time of flight in nanoseconds. Default is 28 ns."""
    readout_amplitude_in_v: Optional[float] = 0.03
    """Readout amplitude in volts. Default is 0.1 V."""
    readout_length_in_ns: Optional[int] = 1000
    """Readout length in nanoseconds. Default is the pulse predefined pulse length."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
