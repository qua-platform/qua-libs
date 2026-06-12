"""Parameters module for Bell state tomography calibration."""

# pylint: disable=too-few-public-methods

from typing import ClassVar, Iterable, Literal

import numpy as np
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters

from calibration_utils.two_q_confusion_matrix import is_confusion_matrix_valid


def require_bell_tomography_prerequisites(qubit_pairs: Iterable, operation: str) -> None:
    """Validate that each pair has the requested CZ macro and a usable confusion matrix."""
    for qp in qubit_pairs:
        if operation not in qp.macros:
            available = sorted(qp.macros.keys())
            raise ValueError(
                f"Qubit pair {qp.name!r} has no macro {operation!r}. Available macros: {available}"
            )
        if qp.confusion is None:
            raise ValueError(
                f"Qubit pair {qp.name!r} has no readout confusion matrix. "
                "Run node 35_two_qubit_confusion_matrix first."
            )
        if not is_confusion_matrix_valid(np.asarray(qp.confusion)):
            raise ValueError(
                f"Qubit pair {qp.name!r} has an invalid confusion matrix. "
                "Re-run node 35_two_qubit_confusion_matrix."
            )


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for Bell state tomography."""

    num_shots: int = 100
    """Number of shots to perform. Default is 100."""
    operation: Literal["cz_flattop", "cz_unipolar", "cz_bipolar", "cz_flattop_erf", "cz_SNZ"] = "cz_unipolar"
    """Type of CZ operation to perform; one of 'cz_flattop', 'cz_unipolar', 'cz_bipolar', 'cz_flattop_erf', or 'cz_SNZ'. Default is 'cz_unipolar'."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for Bell state tomography calibration."""

    targets_name: ClassVar[str] = "qubit_pairs"
