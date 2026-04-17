from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for 14a_crot_spectroscopy_parity_diff."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    exchange_min: float = 0.0
    """Minimum virtual barrier / exchange voltage (V). Default is 0.0."""
    exchange_max: float = 0.5
    """Maximum virtual barrier / exchange voltage (V). Default is 0.5."""
    exchange_points: int = 50
    """Number of points in the exchange voltage sweep. Default is 50."""
    esr_frequency_min: float = -50e6
    """Minimum ESR drive frequency offset (Hz). Default is -50 MHz."""
    esr_frequency_max: float = 50e6
    """Maximum ESR drive frequency offset (Hz). Default is 50 MHz."""
    esr_frequency_points: int = 100
    """Number of points in the ESR frequency sweep. Default is 100."""
    duration: int = 1024
    """CROT drive pulse duration in nanoseconds. Default is 1000 ns."""
    hold_duration: int = 1024
    """Hold duration at exchange point before measurement (ns). Default is 100 ns."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Parameter set for 14a_crot_spectroscopy_parity_diff."""
