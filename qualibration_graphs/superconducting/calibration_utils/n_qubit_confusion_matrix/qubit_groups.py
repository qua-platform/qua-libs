"""Qubit group types, resolution, and persistence for N-qubit confusion matrix calibration."""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
from qualibrate import QualibrationNode
from qualibration_libs.core.exceptions import format_available_items

MAX_QUBITS = 5


@dataclass
class QubitGroup:
    """Container for a group of qubits measured together."""

    qubits: List
    num_qubits: int
    name: str

    @classmethod
    def from_qubits(cls, qubits: List) -> "QubitGroup":
        if not qubits:
            raise ValueError("Each qubit group must contain at least one qubit")
        if len(qubits) > MAX_QUBITS:
            raise ValueError(f"Each qubit group may contain at most {MAX_QUBITS} qubits")
        return cls(
            qubits=qubits,
            num_qubits=len(qubits),
            name="-".join(q.name for q in qubits),
        )


def get_qubit_groups(node: QualibrationNode) -> List[QubitGroup]:
    """Resolve configured qubit groups from the node parameters and machine state."""
    qubit_groups_param = node.parameters.qubit_groups
    if qubit_groups_param is None or qubit_groups_param == "":
        raise ValueError("qubit_groups must be provided")

    qubit_groups = []
    for group in qubit_groups_param:
        try:
            qubits = [node.machine.qubits[q] for q in group]
        except KeyError as exc:
            qubits_list = format_available_items(node.machine.qubits, item_type="qubits")
            missing = next(q for q in group if q not in node.machine.qubits)
            raise KeyError(f"Qubit '{missing}' not found in machine. {qubits_list}") from exc
        qubit_groups.append(QubitGroup.from_qubits(qubits))

    num_qubits_values = {qg.num_qubits for qg in qubit_groups}
    if len(num_qubits_values) > 1:
        raise ValueError("All qubit groups must have the same number of qubits")

    return qubit_groups


def find_qubit_pair_for_group(machine, group: QubitGroup):
    """Find a qubit pair entry associated with the first two qubits in a group."""
    if group.num_qubits < 2:
        return None

    q1_name = group.qubits[0].name
    q2_name = group.qubits[1].name
    for pair_name in (f"{q1_name}-{q2_name}", f"{q2_name}-{q1_name}"):
        if pair_name in machine.qubit_pairs:
            return machine.qubit_pairs[pair_name]
    return None


def save_confusion_to_qubit_pair_extras(
    machine,
    qubit_groups: Iterable[QubitGroup],
    confusions: Dict[str, np.ndarray],
    log_callable: Optional[Callable[[str], None]] = None,
) -> None:
    """Save measured N-qubit confusion matrices into qubit pair extras."""
    for group in qubit_groups:
        if group.num_qubits < 2:
            if log_callable is not None:
                log_callable(
                    f"Warning: Qubit group {group.name} has less than 2 qubits. "
                    "Cannot save to qubit pair extras."
                )
            continue

        qp = find_qubit_pair_for_group(machine, group)
        if qp is None:
            q1_name = group.qubits[0].name
            q2_name = group.qubits[1].name
            if log_callable is not None:
                log_callable(
                    f"Warning: Qubit pair {q1_name}-{q2_name} or {q2_name}-{q1_name} "
                    "not found in machine.qubit_pairs. Skipping confusion matrix save."
                )
            continue

        if not hasattr(qp, "extras") or qp.extras is None:
            qp.extras = {}

        qubit_names = [q.name for q in group.qubits]
        candidate_names = [group.name, "-".join(sorted(qubit_names))]
        confusion_key = f"confusion_{group.num_qubits}q"

        for name_to_try in candidate_names:
            if name_to_try not in qp.extras:
                qp.extras[name_to_try] = {}
            qp.extras[name_to_try][confusion_key] = confusions[group.name].tolist()
            break
