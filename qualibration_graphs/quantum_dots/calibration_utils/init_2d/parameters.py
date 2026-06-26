from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    QubitPairExperimentNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of shots per (ramp_duration, wait_duration) point."""
    ramp_duration_min: int = 16
    """Minimum ramp duration in ns (must be multiple of 4)."""
    ramp_duration_max: int = 2000
    """Maximum ramp duration in ns (must be multiple of 4)."""
    ramp_duration_step: int = 40
    """Ramp duration step in ns (must be multiple of 4)."""
    wait_duration_min: int = 16
    """Minimum wait duration between init and measure in ns (must be multiple of 4)."""
    wait_duration_max: int = 2000
    """Maximum wait duration between init and measure in ns (must be multiple of 4)."""
    wait_duration_step: int = 40
    """Wait duration step in ns (must be multiple of 4)."""
    find_minimum: bool = True
    """If True, find the (ramp, wait) pair yielding the minimum average state
    (purest ground-state preparation). If False, find the maximum."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Parameter set for 07a_init_2d_calibration."""
