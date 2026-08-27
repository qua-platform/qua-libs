from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from qualibration_libs.parameters import QubitPairExperimentNodeParameters
from calibration_utils.heralded_initialization_utils import HeraldedInitializeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of shots per (ramp_duration, wait_duration) point."""
    use_simulated_data: bool = False
    """If True, bypass hardware execution and generate a synthetic ``ds_raw``."""
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
    HeraldedInitializeParameters,
    QubitPairExperimentNodeParameters,
):
    """Parameter set for 07a_init_2d_calibration."""
