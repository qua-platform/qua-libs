from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import BaseExperimentNodeParameters

from typing import Optional, List


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    sensor_names: Optional[List[str]] = None
    """The list of sensor dot names to be included in the measurement. """
    quantum_dot_pair: str = "virtual_dot_1_virtual_dot_2_pair"
    """The name of the QD pair to sweep detuning."""
    detuning_start: float = 0
    """The first detuning value."""
    detuning_stop: float = 0.5
    """The last detuning value."""
    detuning_step: float = 0.005
    """The step in detuning. """
    point_duration: int = 1000
    """How long to stay on each detuning point for before measurement."""
    frequency_span_in_mhz: float = 15
    """Span of frequencies to sweep in MHz. Default is 15 MHz."""
    frequency_step_in_mhz: float = 0.1
    """Step size for frequency sweep in MHz. Default is 0.1 MHz."""
    use_simulated_data: bool = False
    """Whether to generate simulated data instead of measuring via the OPX. Default False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    BaseExperimentNodeParameters,
):
    pass
