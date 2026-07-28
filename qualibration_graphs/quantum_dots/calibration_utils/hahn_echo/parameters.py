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
    """Minimum per-arm idle time in nanoseconds. Must be >= 4 clock cycles. Default is 16 ns."""
    tau_max: int = 10_000
    """Maximum per-arm idle time in nanoseconds. Default is 10000 ns (10 µs)."""
    tau_step: int = 16
    """Step size for the per-arm idle time sweep in nanoseconds. Default is 16 ns."""
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
