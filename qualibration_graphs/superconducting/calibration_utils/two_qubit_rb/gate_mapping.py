"""Integer gate identifiers to human-readable labels for RB circuit debugging.

Opcodes match :func:`~calibration_utils.two_qubit_rb.circuit_utils.get_layer_integer`
and :func:`~calibration_utils.two_qubit_rb.qua_utils.play_gate` (0–37 encoding).
Opcode 38 is not a gate; readout is not part of this table.

Single-qubit layers use ``g_control * 6 + g_target``. Slot names depend on the basis:
``XY`` is analog Y (0 X90, 1 X180, 2 Y90, 3 Y180, 4 -Y90) and ``XZ`` is virtual Z
(0 X90, 1 X180, 2 Z90, 3 Z180, 4 Z270, applied as ``frame_rotation``). Slot 5 is idle.
CZ flux phase compensation stays inside CZ.
"""

from __future__ import annotations

from typing import Callable

from calibration_utils.two_qubit_rb.packing import SingleQubitBasis

IDLE_SLOT = 5
N_SLOTS = 6

SLOT_NAMES: dict[str, tuple[str, ...]] = {
    "XY": ("X90", "X180", "Y90", "Y180", "-Y90"),
    "XZ": ("X90", "X180", "Z90", "Z180", "Z270"),
}
TWO_QUBIT_LABELS: dict[int, str] = {36: "CZ gate", 37: "Idle_2q (Wait(4) both qubits)"}


def _layer_label(control_slot: int, target_slot: int, names: tuple[str, ...]) -> str:
    parts = []
    if control_slot != IDLE_SLOT:
        parts.append(f"{names[control_slot]}(control)")
    if target_slot != IDLE_SLOT:
        parts.append(f"{names[target_slot]}(target)")
    return " + ".join(parts) or "Idle (both qubits)"


def gate_mapping(single_qubit_basis: SingleQubitBasis) -> dict[int, str]:
    """Return the opcode-to-label table (0–37) for one single-qubit basis."""
    names = SLOT_NAMES[single_qubit_basis]
    table = {
        control_slot * N_SLOTS + target_slot: _layer_label(control_slot, target_slot, names)
        for control_slot in range(N_SLOTS)
        for target_slot in range(N_SLOTS)
    }
    table.update(TWO_QUBIT_LABELS)
    return table


def layer_label(opcode: int, single_qubit_basis: SingleQubitBasis) -> str:
    """Return the human-readable label for one layer opcode."""
    return gate_mapping(single_qubit_basis).get(opcode, f"Unknown opcode {opcode}")


def format_circuit(
    circuit_as_ints: list[int],
    single_qubit_basis: SingleQubitBasis,
    *,
    include_index: bool = True,
) -> str:
    """Format an encoded RB circuit as an explicit layer-by-layer string."""
    table = gate_mapping(single_qubit_basis)
    lines = []
    for idx, opcode in enumerate(circuit_as_ints):
        prefix = f"{idx:4d}: " if include_index else ""
        label = table.get(opcode, f"Unknown opcode {opcode}")
        lines.append(f"{prefix}[{opcode:2d}] {label}")
    return "\n".join(lines)


def print_circuit(
    circuit_as_ints: list[int],
    single_qubit_basis: SingleQubitBasis,
    *,
    header: str | None = None,
    log_callable: Callable[[str], None] = print,
) -> None:
    """Print an encoded RB circuit with explicit per-layer opcode labels."""
    if header:
        log_callable(header)
    log_callable(format_circuit(circuit_as_ints, single_qubit_basis))
