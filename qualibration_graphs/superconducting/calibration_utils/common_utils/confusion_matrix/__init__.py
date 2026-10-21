"""Shared helpers for multi- and two-qubit readout confusion matrix calibrations."""

from .compute import (
    compute_confusion_matrices,
    compute_kron_confusion_matrices,
    compute_marginal_confusion_matrices,
    marginal_assignment_fidelity,
    recover_prepared_probs,
    resolve_target_dim,
)
from .load_confusion import get_nq_confusion_matrix, reorder_confusion_matrix
from .plotting import get_state_labels
from .validation import is_confusion_matrix_valid

__all__ = [
    "compute_confusion_matrices",
    "compute_kron_confusion_matrices",
    "compute_marginal_confusion_matrices",
    "get_nq_confusion_matrix",
    "get_state_labels",
    "is_confusion_matrix_valid",
    "marginal_assignment_fidelity",
    "recover_prepared_probs",
    "resolve_target_dim",
    "reorder_confusion_matrix",
]
