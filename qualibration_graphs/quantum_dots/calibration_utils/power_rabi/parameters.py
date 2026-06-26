from typing import Literal
from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
)

from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    ParityDiffAnalysisParameters,
)


class BaseRabiSpecificParameters(RunnableParameters):
    """Parameters shared by nodes 09a (Power Rabi), 09b (Error Amplified Power Rabi), and 08c (Error Amplified Power Rabi Overtime)."""

    num_shots: int = 300
    """Number of averages to perform. Default is 100."""
    min_amp_factor: float = 0.75
    """Minimum amplitude factor for the operation. Default is 0.001."""
    max_amp_factor: float = 1.25
    """Maximum amplitude factor for the operation. Default is 1.99."""
    amp_factor_step: float = 0.005
    """Step size for the amplitude factor. Default is 0.005."""
    operation: Literal["x180", "x90", "y90"] = "x180"
    """The operation to perform to drive the qubit."""

class ErrorAmplifiedSpecificParameters(BaseRabiSpecificParameters):
    max_n_pulses: int = 40
    """Number of pulses in the error-amplified power Rabi pulse sequence."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    BaseRabiSpecificParameters,
    QubitsExperimentNodeParameters,
    ParityDiffAnalysisParameters,
):
    """Parameter set for 09a_power_rabi."""
    amp_default: float = 1



class ErrorAmplifiedParameters(
    NodeParameters,
    CommonNodeParameters,
    ErrorAmplifiedSpecificParameters,
    QubitsExperimentNodeParameters,
    ParityDiffAnalysisParameters,
):
    """Parameter set for 09b_power_rabi_error_amplification"""
