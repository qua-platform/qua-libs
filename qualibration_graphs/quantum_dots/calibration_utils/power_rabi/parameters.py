from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters
from calibration_utils.heralded_initialization_utils import HeraldedInitializeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    min_amp_factor: float = 0.001
    """Minimum amplitude factor for the x180 operation. Default is 0.001."""
    max_amp_factor: float = 1.99
    """Maximum amplitude factor for the x180 operation. Default is 1.99."""
    amp_factor_step: float = 0.01
    """Step size for the amplitude factor. Default is 0.01."""
    use_simulated_data: bool = False
    """Whether to generate simulated data instead of measuring via the OPX. Default False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    HeraldedInitializeParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for 09a_power_rabi."""
