from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    QubitPairExperimentNodeParameters,
    ParityDiffAnalysisParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for 16a_geometric_cz_duration_calibration."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    min_exchange_duration_in_ns: int = 16
    """Minimum exchange pulse duration in nanoseconds. Must be >= 16 ns (4 clock cycles). Default is 16 ns."""
    max_exchange_duration_in_ns: int = 2000
    """Maximum exchange pulse duration in nanoseconds. Default is 2000 ns."""
    duration_step_in_ns: int = 20
    """Step size for the exchange pulse duration sweep in nanoseconds. Default is 20 ns."""
    quadrature_signal_center: float = 0.5
    """Signal value corresponding to the center of the Ramsey I/Q circle. Default is 0.5."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
    ParityDiffAnalysisParameters,
):
    """Parameter set for 16a_geometric_cz_duration_calibration."""
