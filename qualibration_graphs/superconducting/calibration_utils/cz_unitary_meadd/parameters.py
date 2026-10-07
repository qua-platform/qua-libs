from typing import ClassVar, Iterable, Literal

import numpy as np
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters

from calibration_utils.common_utils.confusion_matrix import is_confusion_matrix_valid


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for the CZ unitary reconstruction (MEADD + Floquet)."""

    num_shots: int = 100
    """Number of shots per circuit. Default is 100."""
    operation: Literal["cz_flattop", "cz_unipolar", "cz_bipolar", "cz_flattop_erf", "cz_SNZ"] = "cz_unipolar"
    """CZ macro to characterize. Default is 'cz_unipolar'."""
    max_cz_meadd: int = 24
    """Maximum number of CZ gates in the MEADD circuits; must be a multiple of step_cz_meadd. Default is 24."""
    step_cz_meadd: int = 2
    """Depth step of the MEADD circuits; must be even. A step of 4 cancels microwave pulse errors to first
    order. Default is 2."""
    max_cz_floquet: int = 20
    """Maximum number of CZ gates in the Floquet circuits (depths 0, 1, ..., max). Default is 20."""
    include_floquet_phi: bool = False
    """Add the |11>-reference Floquet circuits that give a second, noisier estimate of phi. Default is False."""
    use_readout_mitigation: bool = True
    """Correct the joint probabilities with the pair's 4x4 confusion matrix from node 35. Default is True."""
    frame_sign: Literal[-1, 1] = 1
    """Sign s such that frame_rotation_2pi(x) acts as Z(s * 2pi * x) in the analysis convention. It sets the
    sign of chi and of the suggested phase corrections. Flip it if applying the suggested corrections doubles
    gamma and zeta instead of zeroing them. Default is 1."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    targets_name: ClassVar[str] = "qubit_pairs"


def require_prerequisites(qubit_pairs: Iterable, operation: str, use_readout_mitigation: bool) -> None:
    """Check that each pair has the CZ macro and, if mitigation is on, a valid confusion matrix."""
    for qp in qubit_pairs:
        if operation not in qp.macros:
            available = sorted(qp.macros.keys())
            raise ValueError(f"Qubit pair {qp.name!r} has no macro {operation!r}. Available macros: {available}")
        if not use_readout_mitigation:
            continue
        if qp.confusion is None or not is_confusion_matrix_valid(np.asarray(qp.confusion)):
            raise ValueError(
                f"Qubit pair {qp.name!r} has no valid readout confusion matrix. "
                "Run node 35_two_qubit_confusion_matrix first, or set use_readout_mitigation=False."
            )
