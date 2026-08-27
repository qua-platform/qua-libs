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
    """Parameter set for 11a_ramsey."""

    frequency_detuning_in_mhz: float = 0.25
    """Frequency detuning in MHz. Default is 1.0 MHz."""
