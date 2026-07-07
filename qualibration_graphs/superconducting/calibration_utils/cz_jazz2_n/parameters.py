"""Parameters module for the JAZZ2-N CZ amplitude calibration.

The JAZZ2-N protocol (Appendix I.1, Fig. 13(b) of arXiv:2402.18926v3) is the
two-qubit-superposition variant of JAZZ-N: both qubits receive a boundary
X_{pi/2} pulse, both are read out, and the metric is P_|00>. The repetition
N must satisfy N = 2k (k = 0, 1, 2, ...). The same X_pi refocused CZ train
gives the (2k+1)*theta_CZ phase accumulation, which is maximised when
theta_CZ = pi. Compared to JAZZ-N, the principal-peak fringe in amplitude
is roughly twice as dense for a given total pulse count, so this protocol is
intended as a finer follow-up amplitude calibration (and serves as the reward
signal for downstream Z-pulse shape optimisation).
"""

# pylint: disable=too-few-public-methods

from typing import ClassVar, Literal

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for the JAZZ2-N CZ amplitude calibration."""

    num_shots: int = 100
    """Number of shots to average over. Default is 100."""
    amp_range: float = 0.010
    """Half-width of the amplitude-scale sweep around the stored CZ amplitude (center = 1.0). Default is 0.010."""
    amp_step: float = 0.001
    """Step of the amplitude-scale sweep. Default is 0.001."""
    N_min: int = 0
    """Minimum repetition count. Required form: N = 2k (k = 0, 1, 2, ...); auto-coerced if not. Default is 0."""
    N_max: int = 50
    """Maximum repetition count. Required form: N = 2k (k = 0, 1, 2, ...); auto-coerced if not. Default is 50."""
    operation: Literal["cz_flattop", "cz_unipolar", "cz_bipolar", "cz_flattop_erf", "cz_SNZ"] = "cz_unipolar"
    """Name of the CZGate macro to drive in place of the bare Z pulse. Default is 'cz_unipolar'."""
    use_state_discrimination: bool = True
    """JAZZ2-N reads the joint P_|00> of both qubits, which requires state discrimination. Setting this to False raises."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for the JAZZ2-N CZ amplitude calibration node."""

    targets_name: ClassVar[str] = "qubit_pairs"
