from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import BaseExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    offset_min: float = -0.2
    """Minimum voltage offset for the sensor gate sweep in volts. Default is -0.2 V."""
    offset_max: float = 0.2
    """Maximum voltage offset for the sensor gate sweep in volts. Default is 0.2 V."""
    offset_step: float = 0.005
    """Step size for the voltage offset sweep in volts. Default is 0.005 V."""
    duration_after_step: int = 1000
    """Wait duration after each voltage step in nanoseconds. Default is 1000 ns (1 µs)."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    BaseExperimentNodeParameters,
):
    pass
