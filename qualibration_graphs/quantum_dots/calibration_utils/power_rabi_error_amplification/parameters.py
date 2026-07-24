from typing import Literal

from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters
from calibration_utils.heralded_initialization_utils import HeraldedInitializeParameters
from calibration_utils.measurement_utils import ParityDiffAnalysisParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    min_amp_factor: float = 0.85
    """Minimum amplitude prefactor. Narrow window around expected a_π after node 09a."""
    max_amp_factor: float = 1.15
    """Maximum amplitude prefactor. Narrow window around expected a_π after node 09a."""
    amp_factor_step: float = 0.001
    """Step size for the amplitude prefactor sweep. Default is 0.001."""
    max_n_pulses: int = 40
    """Number of pulses in the error-amplified power Rabi pulse sequence."""
    operation: Literal["x180", "x90", "y90"] = "x180"
    """The operation to perform to drive the qubit."""
    parity_measurement: bool = False
    """Whether or not to perform parity measurement."""
    use_simulated_data: bool = False
    """Whether to generate simulated data instead of measuring via the OPX. Default False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
    HeraldedInitializeParameters,
    ParityDiffAnalysisParameters,
):
    """Parameter set for 09b_power_rabi_error_amplification."""
