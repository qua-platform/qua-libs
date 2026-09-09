"""Helper utilities for N-qubit readout confusion matrix calibration."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import List

from qm.qua import for_
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


@contextmanager
def nested_binary_loops(loop_vars, idx=0):
    """Recursively create nested QUA loops over binary variables."""
    if idx == len(loop_vars):
        yield
        return

    with for_(loop_vars[idx], 0, loop_vars[idx] < 2, loop_vars[idx] + 1):
        with nested_binary_loops(loop_vars, idx + 1):
            yield


def state_to_label(state_int: int, num_qubits: int) -> str:
    """Convert an integer state index to a binary string label."""
    return format(state_int, f"0{num_qubits}b")
