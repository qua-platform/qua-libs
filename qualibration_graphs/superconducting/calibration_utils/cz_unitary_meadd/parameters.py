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
    use_readout_mitigation: bool = True
    """Correct the joint probabilities with the pair's 4x4 confusion matrix from node 35. Default is True."""
    correction_target: Literal["max_fidelity", "ideal_cz"] = "max_fidelity"
    """Value of gamma that the suggested phase corrections aim for (zeta is always set to 0). 'max_fidelity' sets
    gamma = -(phi - pi) / 2, which gives the highest fidelity to CZ for the measured phi. 'ideal_cz' sets gamma = 0,
    the value of the ideal CZ; use it if phi will be recalibrated afterwards. Default is 'max_fidelity'."""


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
