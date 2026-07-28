from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters
from calibration_utils.measurement_utils import ParityDiffAnalysisParameters
from calibration_utils.heralded_initialization_utils import HeraldedInitializeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for 13_xy8."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    tau_min: int = 16
    """Minimum half inter-pulse spacing in nanoseconds. Must be >= 4 clock cycles. Default is 16 ns."""
    tau_max: int = 10_000
    """Maximum half inter-pulse spacing in nanoseconds. Default is 10000 ns (10 µs)."""
    tau_step: int = 4
    """Step size for the half inter-pulse spacing sweep in nanoseconds. Default is 4 ns (1 clock cycle)."""
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
    """Parameter set for 13_xy8."""
