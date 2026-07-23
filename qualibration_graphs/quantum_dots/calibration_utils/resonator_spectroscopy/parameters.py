from typing import List, Optional

from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from qualibration_libs.parameters.experiment import BaseExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    sensor_names: Optional[List[str]] = None
    """The list of sensor dot names to be included in the measurement. """
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: int = 30
    """Span of frequencies to sweep in MHz. Default is 30 MHz."""
    frequency_step_in_mhz: float = 0.1
    """Step size for frequency sweep in MHz. Default is 0.1 MHz."""
    use_simulated_data: bool = False
    """Whether to generate simulated data instead of measuring via the OPX. Default False."""


class Parameters(
    NodeParameters,
    BaseExperimentNodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    pass
