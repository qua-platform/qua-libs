"""Validation helpers for readout confusion matrices."""

import numpy as np

# Matches QUAM single-qubit resonator.confusion_matrix layout:
# conf[measured, prepared] = P(measure row | prepare column).


def is_confusion_matrix_valid(
    conf: np.ndarray,
    col_sum_tol: float = 0.05,
    *,
    prepared_axis: int = 1,
) -> bool:
    """Return True if ``conf`` is finite and normalized along the prepared axis.

    Parameters
    ----------
    conf
        Square confusion matrix in ``conf[measured, prepared]`` layout.
    col_sum_tol
        Allowed deviation from unity for each prepared-state marginal.
    prepared_axis
        Axis indexing prepared states. Defaults to ``1`` (columns are prepared).
    """
    conf = np.asarray(conf)
    n_states = conf.shape[0]
    if conf.ndim != 2 or conf.shape != (n_states, n_states) or not np.all(np.isfinite(conf)):
        return False

    measure_axis = 1 - prepared_axis
    prepared_marginals = conf.sum(axis=measure_axis)
    return bool(np.all(np.abs(prepared_marginals - 1.0) <= col_sum_tol))
