"""Parameters module for Context Aware Fidelity Estimation (CAFE) of a CZ gate."""

# pylint: disable=too-few-public-methods

from typing import ClassVar, Dict, List, Literal

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for CAFE."""

    num_shots: int = 100
    """Number of shots per circuit (one circuit per state, depth and variant). Default is 100."""
    use_state_discrimination: bool = True
    """CAFE needs the joint probability of |00>, so state discrimination is required. Default is True."""
    operation: Literal["cz_flattop", "cz_unipolar", "cz_bipolar", "cz_flattop_erf", "cz_SNZ"] = "cz_unipolar"
    """Type of CZ operation to characterize. Default is 'cz_unipolar'."""
    max_depth: int = 16
    """Largest number of cycle repetitions n. Default is 16."""
    depth_step: int = 2
    """Step between repetition counts, starting at n = 0. The paper uses even n only (step 2). Default is 2."""
    variants: List[Literal["cafe", "decaf"]] = ["cafe"]
    """Cycle variants to measure. 'cafe' repeats the CZ; 'decaf' adds X on both qubits after each CZ,
    which echoes out single-qubit phase errors and low-frequency Z noise. Default is ['cafe']."""
    reference_unitary: Literal["ideal", "characterized"] = "ideal"
    """Reference unitary used to undo the state: the ideal CZ, or a characterized CZ built from
    ``characterized_angles``. Default is 'ideal'."""
    characterized_angles: Dict[str, List[float]] = {}
    """Per qubit pair name, [Δθ, Δγ, Δφ] in radians: swap angle, single-qubit phase and conditional-phase
    error of the characterized CZ (Eq. 2 of arXiv:2303.17565). Pairs not listed use the ideal CZ."""
    quadratic_fit_max_depth: int = 4
    """Largest n used in the quadratic cross-check fit (Eq. 8). Default is 4."""
    record_leakage: bool = False
    """If True, read out with the GEF discriminator and record the |f> population of each qubit.
    Leakage is not used in the fit. Requires a calibrated GEF readout. Default is False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for CAFE."""

    targets_name: ClassVar[str] = "qubit_pairs"
