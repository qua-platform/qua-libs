from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    QubitPairExperimentNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of shots per ramp-duration point. Default is 100."""
    ramp_duration_min: int = 16
    """Minimum ramp duration in ns (must be multiple of 4). Default is 16."""
    ramp_duration_max: int = 2000
    """Maximum ramp duration in ns (must be multiple of 4). Default is 2000."""
    ramp_duration_step: int = 4
    """Ramp duration step in ns (must be multiple of 4). Default is 4."""
    find_minimum: bool = True
    """If True, find the ramp duration yielding the minimum average state
    (purest ground-state preparation). If False, find the maximum. Default is True."""
    ramp_log_scale: bool = False
    """Whether to set the ramp rate axis in log scale. Default False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Parameter set for 07_init_ramp_rate_calibration."""
