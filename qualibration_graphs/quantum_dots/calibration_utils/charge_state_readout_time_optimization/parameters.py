from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from calibration_utils.common_utils.experiment import (
    BaseExperimentNodeParameters,
    HeraldedInitializeParameters,
)
from qualibration_libs.parameters import CommonNodeParameters
from typing import List, Optional


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    quantum_dots: List[str] = None
    """The double quantum dots to include in the measurement."""
    integration_time_start: int = 100
    """Minimum integration time in nanoseconds."""
    integration_time_stop: int = 10000
    """Maximum integration time in nanoseconds."""
    integration_time_step: int = 100
    """Step size for the integration time sweep in nanoseconds."""
    wait_time: int = 5000
    """The amount of time to ensure that any loaded spin decays to a singlet."""
    threshold_SNR: float = 10.0
    """"The threshold value of the SNR to set the integration time to."""
    use_simulated_data: bool = False
    """Whether to run the node and produce simulated data rather than measuring via the OPX. Default False."""


class Parameters(
    NodeParameters,
    BaseExperimentNodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    pass
