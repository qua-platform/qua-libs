from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from qualibration_libs.parameters import QubitPairExperimentNodeParameters
from calibration_utils.heralded_initialization_utils import HeraldedInitializeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of shots per (ramp_duration, detuning) point."""
    use_simulated_data: bool = False
    """If True, bypass hardware execution and generate a synthetic ``ds_raw``.

    This is intended to validate the analysis and plotting pipeline for this node
    (dims/coords/variable names), not to replace QUA simulation.
    """
    ramp_duration_min: int = 16
    """Minimum ramp duration in ns (must be multiple of 4)."""
    ramp_duration_max: int = 2000
    """Maximum ramp duration in ns (must be multiple of 4)."""
    ramp_duration_step: int = 40
    """Ramp duration step in ns (must be multiple of 4)."""
    detuning_min: float = -0.4
    """Minimum detuning voltage in V."""
    detuning_max: float = 0.4
    """Maximum detuning voltage in V."""
    detuning_step: float = 0.01
    """Detuning voltage step in V."""
    find_minimum: bool = True
    """If True, find the (ramp, detuning) pair yielding the minimum average state
    (purest ground-state preparation). If False, find the maximum."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    HeraldedInitializeParameters,
    QubitPairExperimentNodeParameters,
):
    """Parameter set for 07b_init_ramp_detuning_calibration."""
