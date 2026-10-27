"""g/e/f populations for leakage RB: confusion-matrix correction and statistics over sequences."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def read_confusion_matrix(qubit) -> Optional[np.ndarray]:
    """The qubit's g/e/f confusion matrix (rows prepared), or ``None`` if missing or not a valid 3x3."""
    matrix = getattr(qubit.resonator, "gef_confusion_matrix", None)
    if matrix is None:
        return None
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.allclose(matrix.sum(axis=1), 1.0, atol=0.02):
        return None
    return matrix


def correct_populations(measured, confusion_matrices: Sequence[Optional[np.ndarray]]) -> np.ndarray:
    """Correct measured g/e/f population distributions for readout misassignment.

    ``measured`` holds the joint outcomes on its last axis (``3**n`` for ``n`` qubits), ordered like
    ``np.kron`` of the per-qubit matrices: the first qubit is the most significant digit, e.g.
    ``3 * control + target``. Each matrix has rows prepared and columns measured; ``None`` stands
    for an uncorrected qubit (identity). ``true = measured @ inv(C)`` with ``C`` the Kronecker
    product, negative shot-noise values clipped and each distribution renormalized. Returned
    unchanged when no qubit has a matrix.
    """
    measured = np.asarray(measured, dtype=float)
    if all(matrix is None for matrix in confusion_matrices):
        return measured
    confusion = np.ones((1, 1))
    for matrix in confusion_matrices:
        confusion = np.kron(confusion, np.eye(3) if matrix is None else matrix)
    corrected = np.clip(measured @ np.linalg.inv(confusion), 0.0, None)
    total = corrected.sum(axis=-1, keepdims=True)
    return np.divide(corrected, total, out=np.zeros_like(corrected), where=total > 0)


def mean_and_stderr_over_sequences(
    per_sequence: np.ndarray, *, shots_per_sequence: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard error over random sequences (the last axis) of shot-averaged values.

    The SEM is across sequences only: shots repeat the same circuit, so they are not independent
    RB samples. With ``shots_per_sequence`` the error is floored at the binomial shot noise of all
    shots (used for leakage: a depth where every sequence reads exactly 0 would otherwise get zero
    error and pin the weighted fit). The survival leaves it unset.
    """
    per_sequence = np.asarray(per_sequence, dtype=float)
    count = per_sequence.shape[-1]
    mean = per_sequence.mean(axis=-1)
    stderr = per_sequence.std(axis=-1) / np.sqrt(count)
    if shots_per_sequence:
        samples = count * shots_per_sequence
        p = np.clip(mean, 1.0 / samples, 1.0 - 1.0 / samples)
        stderr = np.fmax(stderr, np.sqrt(p * (1.0 - p) / samples))
    return mean, stderr
