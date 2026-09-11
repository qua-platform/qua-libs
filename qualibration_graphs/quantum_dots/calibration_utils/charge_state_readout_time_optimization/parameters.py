from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters

from qualibration_libs.parameters import CommonNodeParameters
from qualibration_libs.parameters.experiment import BaseExperimentNodeParameters
from calibration_utils.heralded_initialization_utils import HeraldedInitializeParameters
from typing import List, Optional


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    quantum_dot_pairs: Optional[List[str]] = None
    """The quantum dot pairs to include in the measurement."""
    detuning_02: float = 0.05
    """The detuning value that ensures a 02 charge state."""
    integration_time_start: int = 100
    """Minimum integration time in nanoseconds."""
    integration_time_stop: int = 10000
    """Maximum integration time in nanoseconds."""
    integration_time_step: int = 100
    """Step size for the integration time sweep in nanoseconds."""
    wait_time: int = 5000
    """The time to wait once stepped to the detuning_02 point before measurement."""
    ramp_duration: int = 16
    """The ramp duration to step to each point."""
    threshold_SNR: float = 10.0
    """"The threshold value of the SNR to set the integration time to."""
    use_simulated_data: bool = False
    """Whether to run the node and produce simulated data rather than measuring via the OPX. Default False."""


class Parameters(
    NodeParameters,
    BaseExperimentNodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    HeraldedInitializeParameters,
):
    pass
