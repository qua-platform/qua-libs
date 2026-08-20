from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
    IdleTimeNodeParameters,
)

from calibration_utils.heralded_initialization_utils.parameters import HeraldedInitializeParameters
from calibration_utils.measurement_utils.parameters import ParityDiffAnalysisParameters


class NodeSpecificParameters(RunnableParameters):
    """Parameters for Ramsey 11a."""

    num_shots: int = 300
    """Number of averages to perform. Default is 100."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
    NodeSpecificParameters,
    HeraldedInitializeParameters,
    QubitsExperimentNodeParameters,
    ParityDiffAnalysisParameters,
):
    """Parameter set for 11c_ramsey_chevron (and related Ramsey chevron nodes)."""

    detuning_span_in_mhz: float = 5.0
    """Frequency detuning span. Default 5MHz."""
    detuning_step_in_mhz: float = 0.1
    """Frequency detuning step. Default 0.1MHz"""
