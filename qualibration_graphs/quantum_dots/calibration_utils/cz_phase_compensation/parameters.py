from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    QubitPairExperimentNodeParameters,
    ParityDiffAnalysisParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for 18_cz_phase_compensation."""

    num_shots: int = 100
    """Number of averages to perform."""
    num_frames: int = 21
    """Number of frame rotation points in [0, 1). More points improve sinusoid fit quality."""
    conditional_phase_tolerance: float = 0.15
    """Tolerance for the conditional phase cross-check (units of 2pi).
    If |measured_conditional_phase - pi| > tolerance * 2*pi, a warning is logged."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
    ParityDiffAnalysisParameters,
):
    """Parameter set for 18_cz_phase_compensation."""
