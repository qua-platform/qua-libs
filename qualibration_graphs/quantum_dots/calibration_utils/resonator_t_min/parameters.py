from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from calibration_utils.common_utils.experiment import BaseExperimentNodeParameters
from qualibration_libs.parameters import CommonNodeParameters
from typing import List, Optional


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    sensor_names: Optional[List[str]] = None
    """The list of sensor dot names to be included in the measurement."""
    quantum_dots: List[str] = None
    """The double quantum dots to include in the measurement."""
    integration_time_start: int = 100

    integration_time_stop: int = 1000

    integration_time_step: int = 100

    wait_time: int = 5000
    """The amount of time to ensure that any loaded spin decays to a singlet."""


class Parameters(
    NodeParameters,
    BaseExperimentNodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    pass
