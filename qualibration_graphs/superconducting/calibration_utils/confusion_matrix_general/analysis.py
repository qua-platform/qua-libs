"""Analysis module for N-qubit readout confusion matrix calibration."""

import itertools
from typing import Callable, Dict, List, Optional

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode

from .helpers import QubitGroup, state_to_label


def is_confusion_matrix_valid(conf: np.ndarray, col_sum_tol: float = 0.05) -> bool:
    """Return True if ``conf`` is a finite square matrix with column sums near unity."""
    conf = np.asarray(conf)
    n_states = conf.shape[0]
    if conf.ndim != 2 or conf.shape != (n_states, n_states) or not np.all(np.isfinite(conf)):
        return False
    col_sums = conf.sum(axis=0)
    return bool(np.all(np.abs(col_sums - 1.0) <= col_sum_tol))


def compute_confusion_matrices(
    ds: xr.Dataset,
    qubit_groups: List[QubitGroup],
    num_shots: int,
    log_callable: Optional[Callable[[str], None]] = None,
) -> Dict[str, np.ndarray]:
    """Compute direct N-qubit confusion matrices from measured integer states."""
    if "state" not in ds.data_vars:
        raise ValueError("Dataset must contain 'state'.")

    num_qubits = qubit_groups[0].num_qubits
    num_states = 2**num_qubits
    pair_dim = "qubit" if "qubit" in ds.dims else "qubit_group"
    shot_dim = "n" if "n" in ds.dims else "N"

    init_coords = [f"init_{idx}" for idx in range(num_qubits)]
    for coord in (pair_dim, shot_dim, *init_coords):
        if coord not in ds.dims and coord not in ds.coords:
            raise ValueError(f"Dataset missing required dimension/coordinate '{coord}'.")

    confusions = {}
    for qg in qubit_groups:
        conf_rows = []
        for init_values in itertools.product([0, 1], repeat=num_qubits):
            sel_dict = {f"init_{idx}": val for idx, val in enumerate(init_values)}
            measured = ds.sel({pair_dim: qg.name}).state.sel(**sel_dict).values
            conf_rows.append(np.bincount(measured.astype(int), minlength=num_states))
        confusions[qg.name] = np.array(conf_rows) / num_shots

        if log_callable is not None and not is_confusion_matrix_valid(confusions[qg.name]):
            log_callable(
                f"Group {qg.name}: confusion matrix failed validation "
                f"(column sums = {confusions[qg.name].sum(axis=0)})."
            )

    return confusions


def compute_kron_confusion_matrices(qubit_groups: List[QubitGroup]) -> Dict[str, np.ndarray]:
    """Compute Kronecker-product reference matrices from per-qubit readout matrices."""
    kron_confs = {}
    for qg in qubit_groups:
        conf_mat = np.array([[1.0]])
        for q in qg.qubits:
            conf_mat = np.kron(conf_mat, q.resonator.confusion_matrix)
        kron_confs[qg.name] = conf_mat
    return kron_confs


def get_state_labels(num_qubits: int) -> List[str]:
    """Return binary labels for all computational basis states."""
    num_states = 2**num_qubits
    return [state_to_label(state, num_qubits) for state in range(num_states)]


def find_qubit_pair_for_group(machine, qg: QubitGroup):
    """Find a qubit pair entry associated with the first two qubits in a group."""
    if qg.num_qubits < 2:
        return None

    q1_name = qg.qubits[0].name
    q2_name = qg.qubits[1].name
    for pair_name in (f"{q1_name}-{q2_name}", f"{q2_name}-{q1_name}"):
        if pair_name in machine.qubit_pairs:
            return machine.qubit_pairs[pair_name]
    return None


def save_confusion_to_qubit_pair_extras(
    machine,
    qubit_groups: List[QubitGroup],
    confusions: Dict[str, np.ndarray],
    log_callable: Optional[Callable[[str], None]] = None,
) -> None:
    """Save measured confusion matrices into qubit pair extras."""
    for qg in qubit_groups:
        if qg.num_qubits < 2:
            if log_callable is not None:
                log_callable(
                    f"Warning: Qubit group {qg.name} has less than 2 qubits. "
                    "Cannot save to qubit pair extras."
                )
            continue

        qp = find_qubit_pair_for_group(machine, qg)
        if qp is None:
            q1_name = qg.qubits[0].name
            q2_name = qg.qubits[1].name
            if log_callable is not None:
                log_callable(
                    f"Warning: Qubit pair {q1_name}-{q2_name} or {q2_name}-{q1_name} "
                    "not found in machine.qubit_pairs. Skipping confusion matrix save."
                )
            continue

        if not hasattr(qp, "extras") or qp.extras is None:
            qp.extras = {}

        qubit_names = [q.name for q in qg.qubits]
        candidate_names = [qg.name, "-".join(sorted(qubit_names))]
        confusion_key = f"confusion_{qg.num_qubits}q"

        for name_to_try in candidate_names:
            if name_to_try not in qp.extras:
                qp.extras[name_to_try] = {}
            qp.extras[name_to_try][confusion_key] = confusions[qg.name].tolist()
            break
