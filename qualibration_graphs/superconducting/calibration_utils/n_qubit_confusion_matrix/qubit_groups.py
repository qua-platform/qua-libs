"""Qubit group types, resolution, and persistence for N-qubit confusion matrix calibration."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

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
    qubit_pairs: Dict[str, Any] = field(default_factory=dict)
    """Nearest-neighbor pair objects keyed as ``pair_01``, ``pair_12``, etc."""

    @classmethod
    def from_qubits(cls, qubits: List, machine=None, *, resolve_adjacent_pairs: bool = False) -> "QubitGroup":
        if not qubits:
            raise ValueError("Each qubit group must contain at least one qubit")
        if len(qubits) > MAX_QUBITS:
            raise ValueError(f"Each qubit group may contain at most {MAX_QUBITS} qubits")
        qg = cls(
            qubits=qubits,
            num_qubits=len(qubits),
            name="-".join(q.name for q in qubits),
        )
        if resolve_adjacent_pairs:
            if machine is None:
                raise ValueError("machine is required when resolve_adjacent_pairs=True")
            qg.qubit_pairs = resolve_adjacent_qubit_pairs(qubits, machine)
        return qg


def resolve_adjacent_qubit_pairs(qubits: List, machine) -> Dict[str, Any]:
    """Resolve nearest-neighbor qubit pair objects along a qubit chain."""
    qubit_pairs: Dict[str, Any] = {}
    for i in range(len(qubits) - 1):
        q1 = qubits[i]
        q2 = qubits[i + 1]
        pair_key = f"pair_{i}{i + 1}"
        pair_name_1 = f"{q1.name}-{q2.name}"
        pair_name_2 = f"{q2.name}-{q1.name}"

        if pair_name_1 in machine.qubit_pairs:
            qubit_pairs[pair_key] = machine.qubit_pairs[pair_name_1]
        elif pair_name_2 in machine.qubit_pairs:
            qubit_pairs[pair_key] = machine.qubit_pairs[pair_name_2]
        else:
            for qp in machine.qubit_pairs.values():
                if qp.qubit_control in [q1, q2] and qp.qubit_target in [q1, q2]:
                    qubit_pairs[pair_key] = qp
                    break

    return qubit_pairs


def get_qubit_groups(
    node: QualibrationNode,
    *,
    min_qubits: int = 1,
    max_qubits: int = MAX_QUBITS,
    resolve_adjacent_pairs: bool = False,
) -> List[QubitGroup]:
    """Resolve configured qubit groups from the node parameters and machine state."""
    qubit_groups_param = node.parameters.qubit_groups
    if qubit_groups_param is None or qubit_groups_param == "":
        raise ValueError("qubit_groups must be provided")

    qubit_groups = []
    for group in qubit_groups_param:
        group = group.strip()
        if not group:
            raise ValueError("Each qubit group string must contain at least one qubit name")
        qubit_names = [q.strip() for q in group.split("-") if q.strip()]
        try:
            qubits = [node.machine.qubits[q] for q in qubit_names]
        except KeyError as exc:
            qubits_list = format_available_items(node.machine.qubits, item_type="qubits")
            missing = next(q for q in qubit_names if q not in node.machine.qubits)
            raise KeyError(f"Qubit '{missing}' not found in machine. {qubits_list}") from exc
        qubit_groups.append(
            QubitGroup.from_qubits(
                qubits,
                node.machine,
                resolve_adjacent_pairs=resolve_adjacent_pairs,
            )
        )

    num_qubits_values = {qg.num_qubits for qg in qubit_groups}
    if len(num_qubits_values) > 1:
        raise ValueError("All qubit groups must have the same number of qubits")

    num_qubits = qubit_groups[0].num_qubits
    if num_qubits < min_qubits or num_qubits > max_qubits:
        raise ValueError(f"Number of qubits must be between {min_qubits} and {max_qubits}, got {num_qubits}")

    return qubit_groups


def _cz_flux_pulse_name(cz_macro) -> str:
    """Return the flux-pulse operation name played by a CZ macro on the moving qubit."""
    flux_pulse = cz_macro.flux_pulse_qubit
    if isinstance(flux_pulse, str):
        return flux_pulse
    pulse_id = getattr(flux_pulse, "id", None)
    if pulse_id:
        return pulse_id
    return cz_macro.flux_pulse_qubit_label


def require_adjacent_cz_macros(qubit_groups: Iterable[QubitGroup], operation: str) -> None:
    """Validate that each chain has adjacent pairs and the requested CZ macro."""
    for qg in qubit_groups:
        for pair_idx in range(qg.num_qubits - 1):
            pair_key = f"pair_{pair_idx}{pair_idx + 1}"
            if pair_key not in qg.qubit_pairs:
                q1_name = qg.qubits[pair_idx].name
                q2_name = qg.qubits[pair_idx + 1].name
                raise ValueError(
                    f"Qubit group {qg.name!r} is missing adjacent pair {pair_key!r} "
                    f"({q1_name}–{q2_name}): no qubit_pair entry in the machine. "
                    "Reorder the chain so each consecutive pair is physically coupled "
                    "(GHZ prep applies CZ only on neighbors in list order)."
                )
            qp = qg.qubit_pairs[pair_key]
            if operation not in qp.macros:
                available = sorted(qp.macros.keys())
                raise ValueError(
                    f"Pair for group {qg.name!r} has no macro {operation!r}. Available macros: {available}"
                )
            cz_macro = qp.macros[operation]
            moving = qp.qubit_control if qp.moving_qubit == "control" else qp.qubit_target
            pulse_name = _cz_flux_pulse_name(cz_macro)
            if pulse_name not in moving.z.operations:
                raise ValueError(
                    f"Qubit group {qg.name!r}: {pair_key} uses pair {qp.id!r}, whose {operation!r} "
                    f"macro plays {pulse_name!r} on {moving.name}.z, but that pulse is not on "
                    f"{moving.name}'s flux line. Re-run CZ calibration for {qp.id!r}, or pick a "
                    "chain order where every step's moving qubit has the flux pulse registered."
                )


def find_qubit_pair_for_group(machine, group: QubitGroup):
    """Find a qubit pair entry associated with any two qubits in a group."""
    if group.num_qubits < 2:
        return None

    qubit_names = [q.name for q in group.qubits]
    for i in range(group.num_qubits):
        for j in range(group.num_qubits):
            if i == j:
                continue
            q1_name = qubit_names[i]
            q2_name = qubit_names[j]
            for pair_name in (f"{q1_name}-{q2_name}", f"{q2_name}-{q1_name}"):
                if pair_name in machine.qubit_pairs:
                    return pair_name, machine.qubit_pairs[pair_name]
    return None, None


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
                    f"Warning: Qubit group {group.name} has less than 2 qubits. " "Cannot save to qubit pair extras."
                )
            continue

        pair_name, qp = find_qubit_pair_for_group(machine, group)
        if qp is None:
            qubit_names = [q.name for q in group.qubits]
            if log_callable is not None:
                log_callable(
                    f"Warning: No qubit pair found for group {group.name} "
                    f"(qubits: {', '.join(qubit_names)}). Skipping confusion matrix save."
                )
            continue

        qp = machine.qubit_pairs[getattr(qp, "id", pair_name)]

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
