"""Compute and apply readout confusion matrices."""

import itertools
import warnings
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import xarray as xr
from scipy.optimize import minimize

from .validation import is_confusion_matrix_valid


def _resolve_target_dim(ds: xr.Dataset) -> str:
    for candidate in ("qubit_pair", "qubit_group"):
        if candidate in ds.dims:
            return candidate
    raise ValueError("Dataset must contain a 'qubit_pair' or 'qubit_group' dimension.")


def compute_confusion_matrices(
    ds: xr.Dataset,
    target_names: Iterable[str],
    num_shots: int,
    init_coords: Sequence[str],
    *,
    target_dim: Optional[str] = None,
    log_callable: Optional[Callable[[str], None]] = None,
) -> Dict[str, np.ndarray]:
    """Compute confusion matrices from joint readout data.

    Matrices are returned in ``conf[measured, prepared]`` layout, matching
    QUAM ``resonator.confusion_matrix``. Each column (fixed prepared state)
    sums to one.

    Parameters
    ----------
    ds
        Dataset with integer ``state`` and preparation coordinates.
    target_names
        Names selecting each target on ``target_dim`` (pair or group name).
    num_shots
        Number of shots used to normalise counts.
    init_coords
        Preparation coordinates swept in product order, e.g.
        ``["init_state_control", "init_state_target"]`` or ``["init_0", "init_1", ...]``.
    target_dim
        Dataset dimension for targets. Auto-detected when omitted.
    log_callable
        Optional logger for validation warnings.

    Returns
    -------
    dict[str, np.ndarray]
        Confusion matrix per target name with rows=measured, cols=prepared.
    """
    if "state" not in ds.data_vars:
        raise ValueError("Dataset must contain 'state'.")

    target_dim = target_dim or _resolve_target_dim(ds)
    shot_dim = "n" if "n" in ds.dims else "N"
    num_qubits = len(init_coords)
    num_states = 2**num_qubits

    for coord in (target_dim, shot_dim, *init_coords):
        if coord not in ds.dims and coord not in ds.coords:
            raise ValueError(f"Dataset missing required dimension/coordinate '{coord}'.")

    confusions = {}
    for name in target_names:
        conf_rows = []
        for init_values in itertools.product([0, 1], repeat=num_qubits):
            sel_dict = dict(zip(init_coords, init_values))
            measured = ds.sel({target_dim: name}).state.sel(**sel_dict).values
            conf_rows.append(np.bincount(measured.astype(int), minlength=num_states))

        conf = np.array(conf_rows).T / num_shots
        confusions[name] = conf
        if log_callable is not None and not is_confusion_matrix_valid(conf):
            marginals = conf.sum(axis=0)
            log_callable(f"{name}: confusion matrix failed validation (prepared marginals = {marginals}).")

    return confusions


def compute_marginal_confusion_matrices(conf: np.ndarray, qubit_names: Sequence[str]) -> Dict[str, np.ndarray]:
    """Compute single-qubit marginal confusion matrices from a joint ``conf[measured, prepared]`` matrix.

    ``qubit_names`` must be ordered to match the joint matrix's bit order (qubit 0 is the most
    significant bit, matching ``compute_confusion_matrices`` and the node's state expression).

    Each marginal conditions on qubit i's prepared bit and averages (unweighted, since every joint
    prepared state is measured with the same number of shots) over the other qubits' prepared
    states, then sums over the other qubits' measured outcomes. Returned as
    ``conf[measured_bit, prepared_bit]`` per qubit name, matching the joint-matrix convention.

    Note that these marginals include readout crosstalk from the other qubits being driven and
    measured at the same time, unlike a single-qubit confusion matrix measured in isolation.
    """
    conf = np.asarray(conf, dtype=float)
    num_qubits = len(qubit_names)
    num_states = conf.shape[0]
    bit_labels = [format(state, f"0{num_qubits}b") for state in range(num_states)]
    bits = np.array([[int(bit) for bit in label] for label in bit_labels])  # (num_states, num_qubits)

    marginals = {}
    for q_idx, name in enumerate(qubit_names):
        qubit_bits = bits[:, q_idx]
        marginal = np.zeros((2, 2))
        for prepared_bit in (0, 1):
            columns = np.flatnonzero(qubit_bits == prepared_bit)
            averaged_column = conf[:, columns].mean(axis=1)
            for measured_bit in (0, 1):
                rows = np.flatnonzero(qubit_bits == measured_bit)
                marginal[measured_bit, prepared_bit] = averaged_column[rows].sum()
        marginals[name] = marginal
    return marginals


def marginal_assignment_fidelity(marginal: np.ndarray) -> float:
    """Mean of the diagonal of a 2x2 marginal confusion matrix."""
    marginal = np.asarray(marginal, dtype=float)
    return float(0.5 * (marginal[0, 0] + marginal[1, 1]))


def compute_kron_confusion_matrices(qubits_by_name: Mapping[str, Iterable]) -> Dict[str, np.ndarray]:
    """Compute Kronecker reference matrices from single-qubit readout matrices.

    Returns matrices in ``conf[measured, prepared]`` layout, matching
    ``compute_confusion_matrices`` and QUAM ``resonator.confusion_matrix``.

    Parameters
    ----------
    qubits_by_name
        Mapping from target name to ordered qubit objects, e.g.
        ``{pair.name: [pair.qubit_control, pair.qubit_target]}``.
    """
    kron_confs = {}
    for name, qubits in qubits_by_name.items():
        conf_mat = np.array([[1.0]])
        for q in qubits:
            conf_mat = np.kron(conf_mat, q.resonator.confusion_matrix)
        kron_confs[name] = conf_mat
    return kron_confs


def _least_squares_prepared_probs(conf: np.ndarray, measured_probs: np.ndarray) -> np.ndarray:
    n_states = len(measured_probs)

    def objective(rho: np.ndarray) -> float:
        residual = measured_probs - conf @ rho
        return float(np.sum(residual**2))

    result = minimize(
        objective,
        measured_probs,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_states,
        constraints={"type": "eq", "fun": lambda rho: np.sum(rho) - 1.0},
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    if not result.success:
        warnings.warn(
            f"Least-squares readout correction did not converge: {result.message}",
            stacklevel=2,
        )
    return np.asarray(result.x, dtype=float)


def recover_prepared_probs(conf: np.ndarray, measured_probs: np.ndarray) -> np.ndarray:
    """Recover prepared-state probabilities from measured outcome probabilities.

    Expects ``conf`` in ``conf[measured, prepared]`` layout, where
    ``measured_probs ≈ conf @ prepared_probs``.

    Uses an exact matrix solve when that yields a valid probability vector.
    Otherwise falls back to constrained least squares
    (``rho >= 0``, ``sum(rho) = 1``, minimize ``||measured_probs - conf @ rho||^2``).
    """
    conf = np.asarray(conf, dtype=float)
    measured_probs = np.asarray(measured_probs, dtype=float)
    if conf.ndim != 2 or conf.shape[0] != conf.shape[1]:
        raise ValueError("Confusion matrix must be square.")
    if measured_probs.shape != (conf.shape[0],):
        raise ValueError("measured_probs length must match confusion matrix size.")

    try:
        prepared_probs = np.linalg.solve(conf, measured_probs)
    except np.linalg.LinAlgError:
        return _least_squares_prepared_probs(conf, measured_probs)

    if np.any(prepared_probs < -1e-12):
        return _least_squares_prepared_probs(conf, measured_probs)

    prepared_probs = prepared_probs * (prepared_probs > 0)
    total = prepared_probs.sum()
    if total <= 0:
        return _least_squares_prepared_probs(conf, measured_probs)
    return prepared_probs / total
