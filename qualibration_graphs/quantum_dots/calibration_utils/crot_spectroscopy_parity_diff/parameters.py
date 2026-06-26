from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    QubitPairExperimentNodeParameters,
    ParityDiffAnalysisParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for 15_crot_spectroscopy."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    exchange_min: float = 0.0
    """Minimum virtual barrier / exchange voltage (V). Default is 0.0."""
    exchange_max: float = 0.5
    """Maximum virtual barrier / exchange voltage (V). Default is 0.5."""
    exchange_points: int = 50
    """Number of points in the exchange voltage sweep. Default is 50."""
    esr_frequency_min: float = -2
    """Minimum ESR drive frequency offset (Hz). Default is -50 MHz."""
    esr_frequency_max: float = 2
    """Maximum ESR drive frequency offset (Hz). Default is 50 MHz."""
    esr_frequency_points: int = 100
    """Number of points in the ESR frequency sweep. Default is 100."""
    duration: int = 1200
    """CROT drive pulse duration in nanoseconds. Default is 1024 ns."""
    ramp_duration: int = 2000
    """The CROT macro ramp duration."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
    ParityDiffAnalysisParameters,
):
    """Parameter set for 15_crot_spectroscopy."""
