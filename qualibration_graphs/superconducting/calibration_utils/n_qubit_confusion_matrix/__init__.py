"""N-qubit readout confusion matrix calibration utilities."""

from calibration_utils.common_utils.confusion_matrix import (
    compute_confusion_matrices,
    compute_kron_confusion_matrices,
    compute_marginal_confusion_matrices,
    is_confusion_matrix_valid,
    marginal_assignment_fidelity,
)

from .qubit_groups import MAX_QUBITS, QubitGroup, get_qubit_groups, save_confusion_to_qubit_pair_extras
from .parameters import Parameters
from .plotting import plot_confusion_matrices

__all__ = [
    "MAX_QUBITS",
    "Parameters",
    "QubitGroup",
    "compute_confusion_matrices",
    "compute_kron_confusion_matrices",
    "compute_marginal_confusion_matrices",
    "get_qubit_groups",
    "is_confusion_matrix_valid",
    "marginal_assignment_fidelity",
    "plot_confusion_matrices",
    "save_confusion_to_qubit_pair_extras",
]
