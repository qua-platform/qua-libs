from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for 12_hahn_echo_parity_diff."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    tau_min: int = 16
    """Minimum per-arm idle time in nanoseconds. Must be >= 4 clock cycles. Default is 16 ns."""
    tau_max: int = 10_000
    """Maximum per-arm idle time in nanoseconds. Default is 10000 ns (10 µs)."""
    tau_step: int = 16
    """Step size for the per-arm idle time sweep in nanoseconds. Default is 16 ns."""
    gap_wait_time_in_ns: int = 128
    """Wait time between initialisation and the echo sequence in nanoseconds. Default is 128 ns."""
    operation: str = "x180"
    """Name of the qubit pi-pulse operation. Default is 'x180'."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for 12_hahn_echo_parity_diff."""
