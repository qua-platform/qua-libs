from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    tau_min: int = 16
    """Minimum idle time in nanoseconds. Must be larger than 4 clock cycles. Default is 16 ns."""
    tau_max: int = 10_000
    """Maximum idle time in nanoseconds. Default is 10000 ns (10 us)."""
    tau_step: int = 16
    """Step size for the idle time sweep in nanoseconds. Default is 16 ns."""
    frequency_detuning_in_mhz: float = 1.0
    """Frequency detuning in MHz. Default is 1.0 MHz."""
    gap_wait_time_in_ns: int = 2_048
    """Wait time between initialization and first X90 pulse in nanoseconds. Default is 128 ns."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for 10a_ramsey_parity_diff."""

    pass
