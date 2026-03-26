from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    twpas: list[str] = ["twpaA"]
    """List of twpas to calibrate"""
    num_shots: int = 30
    """Number of averages to perform. Default is 30."""
    frequency_center_in_mhz: float = 6750
    """Center of the readout frequency sweep in MHz. Default is 6750 MHz."""
    frequency_span_in_mhz: float = 400
    """Span of readout frequencies to sweep in MHz. Default is 4 MHz."""
    frequency_step_in_mhz: float = 1
    """Step size for readout frequency sweep in MHz. Default is 0.1 MHz."""

class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
