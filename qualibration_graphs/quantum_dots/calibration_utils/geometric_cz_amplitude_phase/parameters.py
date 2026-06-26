from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    QubitPairExperimentNodeParameters,
    ParityDiffAnalysisParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for 18b_geometric_cz_amplitude_phase_calibration."""

    num_shots: int = 100
    """Number of averages per point. Default is 100."""
    min_exchange_amplitude: float = 0.1
    """Minimum exchange pulse amplitude (virtual barrier gate voltage, V)."""
    max_exchange_amplitude: float = 0.5
    """Maximum exchange pulse amplitude (virtual barrier gate voltage, V)."""
    amplitude_step: float = 0.005
    """Step size for the exchange amplitude sweep (V)."""
    num_phases: int = 21
    """Number of analysis phase points uniformly distributed over [0, 2π)."""
    use_t2pi_model: bool = False
    """When True, use the T_2π(V) polynomial model from 18a_swap_oscillations
    to set a per-amplitude exchange duration (T_2π/2, rounded to 4 ns).
    Requires ``exchange_decay_model`` on the CZ macro.
    When False (default), use the single fixed ``wait_duration`` from the macro."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
    ParityDiffAnalysisParameters,
):
    """Parameter set for 18b_geometric_cz_amplitude_phase_calibration."""