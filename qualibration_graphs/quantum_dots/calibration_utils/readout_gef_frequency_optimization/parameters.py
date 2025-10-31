from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: float = 2
    """Span of frequencies to sweep in MHz. Default is 2 MHz."""
    frequency_step_in_mhz: float = 0.01
    """Step size for frequency sweep in MHz. Default is 0.01 MHz."""
    operation: str = "readout"
    """Operation to perform, e.g., 'readout'. Default is 'readout'."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
