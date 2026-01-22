from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from calibration_utils.common_utils.experiment import BaseExperimentNodeParameters
from qualibration_libs.parameters import CommonNodeParameters

from typing import List, Optional


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: int = 30
    """Span of frequencies to sweep in MHz. Default is 30 MHz."""
    frequency_step_in_mhz: float = 0.1
    """Step size for frequency sweep in MHz. Default is 0.1 MHz."""
    dc_control: bool = False
    """If an associated external DC offset exists."""


class Parameters(
    NodeParameters,
    BaseExperimentNodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    pass
