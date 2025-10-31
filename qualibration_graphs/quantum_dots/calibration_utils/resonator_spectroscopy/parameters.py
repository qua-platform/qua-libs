from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters
from typing import List, Optional

class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    sensor_names: Optional[List[str]] = None
    """Sensors that you would like to sweep"""
    frequency_span_in_mhz: float = 30.0
    """Span of frequencies to sweep in MHz. Default is 30 MHz."""
    frequency_step_in_mhz: float = 0.1
    """Step size for frequency sweep in MHz. Default is 0.1 MHz."""




class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    pass
