from typing import Optional

from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    ParityDiffAnalysisParameters,
    QubitPairExperimentNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for 17a_iswap_error_amplification."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    exchange_amplitude: Optional[float] = None
    """Fixed exchange amplitude. If None, use the saved CZ voltage point."""
    exchange_duration_in_ns: Optional[int] = None
    """Fixed exchange duration. If None, use the saved CZ macro duration."""
    num_cycle_repetitions: Optional[list[int]] = None
    """Explicit two-cycle repetition counts. If set, overrides the dense range."""
    min_num_cycles: int = 0
    """Minimum two-cycle repetition count for the dense range. Default is 0."""
    max_num_cycles: int = 32
    """Maximum two-cycle repetition count for the dense range. Default is 32."""
    num_cycle_step: int = 2
    """Step size for the dense two-cycle repetition range. Default is 2."""
    max_theta_rad: float = 0.25
    """Upper bound for each residual iSWAP component fit, in radians."""
    min_fit_contrast: float = 1e-4
    """Below this transfer span, report a successful zero-angle diagnostic."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
    ParityDiffAnalysisParameters,
):
    """Parameter set for 17a_iswap_error_amplification."""
