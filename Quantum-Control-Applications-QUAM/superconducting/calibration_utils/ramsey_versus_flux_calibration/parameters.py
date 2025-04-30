from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_detuning_in_mhz: float = 1.0
    """Frequency detuning in MHz. Default is 1.0 MHz."""
    min_wait_time_in_ns: int = 16
    """Minimum wait time in nanoseconds. Default is 16."""
    max_wait_time_in_ns: int = 5000
    """Maximum wait time in nanoseconds. Default is 5000."""
    wait_time_step_in_ns: int = 60
    """Step size for the wait time scan in nanoseconds. Default is 60."""
    flux_span: float = 0.01
    """Span of flux values to sweep in volts. Default is 0.01 V."""
    flux_num: int = 21
    """Number of flux points to sample. Default is 21."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
