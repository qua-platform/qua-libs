from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from quam_experiments.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_averages: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_detuning_in_mhz: float = 4.0
    """Frequency detuning in MHz. Default is 4.0 MHz."""
    min_wait_time_in_ns: int = 16
    """Minimum wait time in nanoseconds. Default is 16."""
    max_wait_time_in_ns: int = 2000
    """Maximum wait time in nanoseconds. Default is 2000."""
    wait_time_step_in_ns: int = 20
    """Step size for the wait time scan in nanoseconds. Default is 20."""
    flux_span: float = 0.02
    """Span of flux values to sweep in volts. Default is 0.02 V."""
    flux_step: float = 0.002
    """Step size for the flux scan in volts. Default is 0.002 V."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
