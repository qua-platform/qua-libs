"""Load N-qubit confusion matrices from QUAM state."""

from typing import Callable, Optional, Sequence, Tuple

import numpy as np


def reorder_confusion_matrix(
    confusion_matrix: np.ndarray,
    source_order: Sequence[str],
    target_order: Sequence[str],
) -> np.ndarray:
    """Reorder a confusion matrix to match a different qubit order."""
    confusion_matrix = np.asarray(confusion_matrix)

    if len(source_order) != len(target_order):
        raise ValueError(
            f"source_order and target_order must have the same length: "
            f"{len(source_order)} != {len(target_order)}"
        )

    if set(source_order) != set(target_order):
        raise ValueError(
            f"source_order and target_order must contain the same qubits: "
            f"{set(source_order)} != {set(target_order)}"
        )

    matrix_size = confusion_matrix.shape[0]
    if confusion_matrix.shape[1] != matrix_size:
        raise ValueError(f"Confusion matrix must be square, got shape {confusion_matrix.shape}")

    num_qubits = int(np.log2(matrix_size))
    if 2**num_qubits != matrix_size:
        raise ValueError(f"Confusion matrix size {matrix_size} is not a power of 2")

    if len(source_order) != num_qubits:
        raise ValueError(
            f"Mismatch between confusion matrix size and qubit order length: "
            f"matrix size {matrix_size} implies {num_qubits} qubits, "
            f"but source_order has {len(source_order)} qubits"
        )

    num_states = matrix_size
    state_mapping = {}
    for source_state_idx in range(num_states):
        source_binary = format(source_state_idx, f"0{num_qubits}b")
        source_state_dict = {source_order[i]: int(source_binary[i]) for i in range(num_qubits)}
        target_binary = "".join(str(source_state_dict[q]) for q in target_order)
        target_state_idx = int(target_binary, 2)
        state_mapping[source_state_idx] = target_state_idx

    reordered = np.zeros_like(confusion_matrix)
    for source_row in range(num_states):
        target_row = state_mapping[source_row]
        for source_col in range(num_states):
            target_col = state_mapping[source_col]
            reordered[target_row, target_col] = confusion_matrix[source_row, source_col]

    return reordered


def get_nq_confusion_matrix(
    qubit_names: Sequence[str],
    machine,
    log_callable: Optional[Callable[[str], None]] = None,
) -> Optional[np.ndarray]:
    """Search qubit pair extras for an N-qubit confusion matrix matching ``qubit_names``."""
    target_qubit_order = list(qubit_names)
    qubit_group_name = "-".join(target_qubit_order)
    target_qubit_set = set(target_qubit_order)
    num_qubits = len(target_qubit_set)
    confusion_key = f"confusion_{num_qubits}q"

    def check_in_pair(qp, pair_name: str = "") -> Tuple[Optional[np.ndarray], Optional[str]]:
        if qp is None:
            return None, None

        if num_qubits == 2:
            if hasattr(qp, "confusion") and qp.confusion is not None:
                pair_qubit_names = [qp.qubit_control.name, qp.qubit_target.name]
                if set(pair_qubit_names) == target_qubit_set:
                    confusion_matrix = np.asarray(qp.confusion)
                    if pair_qubit_names != target_qubit_order:
                        confusion_matrix = reorder_confusion_matrix(
                            confusion_matrix, pair_qubit_names, target_qubit_order
                        )
                    return confusion_matrix, f"pair {pair_name} (qp.confusion)"

        if not hasattr(qp, "extras") or qp.extras is None:
            return None, None

        for entry_name, entry_data in qp.extras.items():
            if not hasattr(entry_data, "keys") or not hasattr(entry_data, "__getitem__"):
                continue
            if confusion_key not in entry_data:
                continue
            try:
                entry_qubit_list = entry_name.split("-")
                if set(entry_qubit_list) != target_qubit_set:
                    continue
                confusion_matrix = np.asarray(entry_data[confusion_key])
                if entry_qubit_list != target_qubit_order:
                    confusion_matrix = reorder_confusion_matrix(
                        confusion_matrix, entry_qubit_list, target_qubit_order
                    )
                return confusion_matrix, f"pair {pair_name}, entry '{entry_name}'"
            except (TypeError, ValueError, KeyError):
                continue

        return None, None

    checked_pairs_set = set()
    for i in range(num_qubits):
        for j in range(num_qubits):
            if i == j:
                continue

            q1_name = target_qubit_order[i]
            q2_name = target_qubit_order[j]
            canonical_pair = tuple(sorted([q1_name, q2_name]))
            if canonical_pair in checked_pairs_set:
                continue
            checked_pairs_set.add(canonical_pair)

            pair_name = f"{q1_name}-{q2_name}"
            qp = None
            if pair_name in machine.qubit_pairs:
                qp = machine.qubit_pairs[pair_name]
            elif f"{q2_name}-{q1_name}" in machine.qubit_pairs:
                qp = machine.qubit_pairs[f"{q2_name}-{q1_name}"]
            else:
                q1_obj = machine.qubits[q1_name]
                q2_obj = machine.qubits[q2_name]
                for qp_candidate in machine.qubit_pairs.values():
                    if qp_candidate.qubit_control in [q1_obj, q2_obj] and qp_candidate.qubit_target in [
                        q1_obj,
                        q2_obj,
                    ]:
                        qp = qp_candidate
                        break

            if qp is None:
                continue

            result, location = check_in_pair(qp, pair_name)
            if result is not None:
                if log_callable is not None:
                    log_callable(f"Found {num_qubits}Q confusion matrix for {qubit_group_name} in {location}")
                return result

    if log_callable is not None:
        log_callable(
            f"{num_qubits}Q confusion matrix not found for {qubit_group_name} "
            f"(qubits: {sorted(target_qubit_set)})"
        )
    return None
