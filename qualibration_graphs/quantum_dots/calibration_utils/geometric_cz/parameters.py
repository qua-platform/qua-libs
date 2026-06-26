from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    QubitPairExperimentNodeParameters,
    ParityDiffAnalysisParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for 16_geometric_cz_calibration."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    target_duration_ns: int = 100
    """Target exchange pulse duration (ns, must be a multiple of 4).
    The analysis finds the amplitude where the conditional phase reaches pi
    at this duration."""
    quadrature_signal_center: float = 0.5
    """Expected centre of the I/Q quadrature signal (parity expectation at
    zero phase). Default is 0.5."""
    min_exchange_duration_in_ns: int = 16
    """Minimum exchange pulse duration in nanoseconds. Must be >= 16 ns (4 clock cycles). Default is 16 ns."""
    max_exchange_duration_in_ns: int = 2000
    """Maximum exchange pulse duration in nanoseconds. Default is 2000 ns."""
    duration_step_in_ns: int = 20
    """Step size for the exchange pulse duration sweep in nanoseconds. Default is 20 ns."""
    min_exchange_amplitude: float = 0.0
    """Minimum exchange pulse amplitude (virtual barrier gate voltage, V). Default is 0.0."""
    max_exchange_amplitude: float = 0.5
    """Maximum exchange pulse amplitude (virtual barrier gate voltage, V). Default is 0.5."""
    amplitude_step: float = 0.01
    """Step size for the exchange pulse amplitude sweep in Volts. Default is 0.01."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
    ParityDiffAnalysisParameters,
):
    """Parameter set for 16_geometric_cz_calibration."""
