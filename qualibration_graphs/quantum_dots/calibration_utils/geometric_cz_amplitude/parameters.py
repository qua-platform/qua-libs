from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    QubitPairExperimentNodeParameters,
    ParityDiffAnalysisParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for 16b_geometric_cz_amplitude_calibration."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    min_exchange_amplitude: float = 0.1
    """Minimum exchange pulse amplitude (virtual barrier gate voltage, V). Default is 0.1."""
    max_exchange_amplitude: float = 0.5
    """Maximum exchange pulse amplitude (virtual barrier gate voltage, V). Default is 0.5."""
    amplitude_step: float = 0.005
    """Step size for the exchange pulse amplitude sweep in Volts. Default is 0.005."""
    quadrature_signal_center: float = 0.5
    """Signal value corresponding to the center of the Ramsey I/Q circle. Default is 0.5."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
    ParityDiffAnalysisParameters,
):
    """Parameter set for 16b_geometric_cz_amplitude_calibration."""