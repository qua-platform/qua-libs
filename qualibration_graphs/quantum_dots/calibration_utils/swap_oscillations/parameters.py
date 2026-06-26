from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    QubitPairExperimentNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for 18a_swap_oscillations."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    min_exchange_amplitude: float = 0.0
    """Minimum exchange pulse amplitude (virtual barrier gate voltage, V). Default is 0.0."""
    max_exchange_amplitude: float = 0.5
    """Maximum exchange pulse amplitude (virtual barrier gate voltage, V). Default is 0.5."""
    amplitude_step: float = 0.01
    """Step size for the exchange pulse amplitude sweep in Volts. Default is 0.01."""
    min_exchange_duration_in_ns: int = 16
    """Minimum exchange pulse duration in nanoseconds. Must be >= 16 ns (4 clock cycles). Default is 16 ns."""
    max_exchange_duration_in_ns: int = 2000
    """Maximum exchange pulse duration in nanoseconds. Default is 2000 ns."""
    duration_step_in_ns: int = 20
    """Step size for the exchange pulse duration sweep in nanoseconds. Default is 20 ns."""
    snr_threshold: float = 20.0
    """Minimum FFT peak-to-noise ratio for accepting a 2π oscillation period. Default is 20.0."""
    analysis_role: str = "best"
    """Which qubit signal to analyse: 'best' (highest SNR per amplitude), 'target', 'control', or 'difference'. Default is 'best'."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Parameter set for 18a_swap_oscillations."""
