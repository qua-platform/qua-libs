from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    tau_min: int = 16
    """Minimum pulse duration in nanoseconds. Must be larger than 4 clock cycles. Default is 16 ns."""
    tau_max: int = 10_000
    """Maximum pulse duration in nanoseconds. Default is 100000 ns (10 µs)."""
    tau_step: int = 16
    """Step size for the pulse duration sweep in nanoseconds. Default is 16 ns."""
    gap_wait_time_in_ns: int = 128
    """Wait time between initialization and X180 pulse in nanoseconds. Default is 128 ns."""
    operation: str = "x180"
    """Name of the qubit operation to perform. Default is 'x180'."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
