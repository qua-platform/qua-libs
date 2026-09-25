from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: float = 30.0
    """Span of frequencies to sweep in MHz. Default is 30 MHz."""
    frequency_step_in_mhz: float = 0.1
    """Step size for frequency sweep in MHz. Default is 0.1 MHz."""
    max_iterations: int =1
    """For use with loop. Default only run once"""
    use_simulated_data: bool = False
    """Generate deterministic synthetic data without connecting to the QOP."""

class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
