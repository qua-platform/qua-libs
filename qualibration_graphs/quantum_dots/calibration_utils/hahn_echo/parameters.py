from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters
from calibration_utils.measurement_utils import ParityDiffAnalysisParameters
from calibration_utils.heralded_initialization_utils import HeraldedInitializeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for 12_hahn_echo."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    tau_min: int = 16
    """Minimum Hahn echo idle delay τ in nanoseconds (each x90–y180 segment; total evolution 2τ).

    Must be a multiple of 4 ns (1 QUA clock cycle). Default is 16 ns."""
    tau_max: int = 10_000
    """Maximum Hahn echo idle delay τ in nanoseconds (each x90–y180 segment; total evolution 2τ).

    Default is 10 000 ns (10 µs per segment; 20 µs total evolution)."""
    tau_step: int = 16
    """Step size for the τ sweep in nanoseconds. Default is 16 ns (4 QUA clock cycles)."""
    use_simulated_data: bool = False
    """Whether to generate simulated data instead of measuring via the OPX."""
    sim_noise_std: float = 0.03
    """Gaussian noise std dev on simulated traces before clipping to [0, 1]."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    HeraldedInitializeParameters,
    QubitsExperimentNodeParameters,
    ParityDiffAnalysisParameters,
):
    """Parameter set for 12_hahn_echo."""
