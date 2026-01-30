from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import BaseExperimentNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    tau_min: int = 16
    """Minimum wait duration in nanoseconds. Must be larger than 4 clock cycles. Default is 16 ns."""
    tau_max: int = 10_000
    """Maximum wait duration in nanoseconds. Default is 100000 ns (10 µs)."""
    tau_step: int = 16
    """Step size for the wait duration sweep in nanoseconds. Default is 16 ns."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    BaseExperimentNodeParameters,
    QubitPairExperimentNodeParameters,
):
    pass
