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
    """Minimum CPMG half-spacing τ in nanoseconds (bookend τ, inter-pulse 2τ; total idle 16τ). Must be a multiple of 4 ns (1 QUA clock cycle). Default is 16 ns."""
    tau_max: int = 4_000
    """Maximum CPMG half-spacing τ in nanoseconds (bookend τ, inter-pulse 2τ; total idle 16τ). Default is 4000 ns (64 µs total idle at τ_max; suitable for T₂ ~ 32 µs)."""
    tau_step: int = 100
    """Step size for the τ sweep in nanoseconds. Default is 100 ns (25 QUA clock cycle)."""
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
