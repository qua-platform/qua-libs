from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import BaseExperimentNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    tau_min: int = 16
    """Minimum pulse duration in nanoseconds. Must be larger than 4 clock cycles. Default is 16 ns."""
    tau_max: int = 10_000
    """Maximum pulse duration in nanoseconds. Default is 10000 ns (10 µs)."""
    tau_step: int = 52
    """Step size for the pulse duration sweep in nanoseconds. Default is 52 ns."""
    frequency_min_in_mhz: float = -0.5
    """Minimum frequency detuning in MHz. Default is -0.5 MHz."""
    frequency_max_in_mhz: float = 0.525
    """Maximum frequency detuning in MHz. Default is 0.525 MHz."""
    frequency_step_in_mhz: float = 0.025
    """Step size for the frequency detuning sweep in MHz. Default is 0.025 MHz."""
    operation: str = "x180"
    """Name of the qubit operation to perform. Default is 'x180'."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    BaseExperimentNodeParameters,
    QubitsExperimentNodeParameters,
):
    pass
