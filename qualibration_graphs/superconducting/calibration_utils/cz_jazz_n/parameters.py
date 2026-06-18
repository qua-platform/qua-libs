"""Parameters module for the JAZZ-N CZ amplitude calibration.

The JAZZ-N protocol (Appendix I.1, Fig. 13(a) of arXiv:2402.18926v3) measures
P_|1> of the target qubit after a refocused train of CZ gates. The number of
X_pi echo pulses N must satisfy N = 4k + 1 (k = 0, 1, 2, ...), which gives a
clean (2k+1)*theta_CZ phase accumulation that peaks at theta_CZ = pi.
"""

# pylint: disable=too-few-public-methods

from typing import ClassVar, Literal

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for the JAZZ-N CZ amplitude calibration."""

    num_shots: int = 100
    """Number of shots to average over. Default is 100."""
    amp_range: float = 0.010
    """Half-width of the amplitude-scale sweep around the stored CZ amplitude (center = 1.0). Default is 0.010."""
    amp_step: float = 0.001
    """Step of the amplitude-scale sweep. Default is 0.001."""
    N_min: int = 1
    """Minimum number of X_pi echo pulses. Required form: N = 4k + 1; auto-coerced if not. Default is 1."""
    N_max: int = 101
    """Maximum number of X_pi echo pulses. Required form: N = 4k + 1; auto-coerced if not. Default is 101."""
    operation: Literal["cz_flattop", "cz_unipolar", "cz_bipolar", "cz_flattop_erf", "cz_SNZ"] = "cz_unipolar"
    """Name of the CZGate macro to drive in place of the bare Z pulse. Default is 'cz_unipolar'."""
    use_state_discrimination: bool = True
    """JAZZ-N reads P_|1> of the target qubit, which requires state discrimination. Setting this to False raises."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for the JAZZ-N CZ amplitude calibration node."""

    targets_name: ClassVar[str] = "qubit_pairs"
