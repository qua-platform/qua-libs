"""Integer gate identifiers to human-readable labels for RB circuit debugging.

Opcodes match :func:`~calibration_utils.two_qubit_rb.circuit_utils.get_layer_integer`
and :func:`~calibration_utils.two_qubit_rb.qua_utils.play_gate` (0–37 encoding).
Opcode 38 is not a gate; readout is not part of this table.

Single-qubit layers use ``g_control * 6 + g_target`` with
``g ∈ {0:sx/X90, 1:x/X180, 2:ry(π/2)/Y90, 3:y/Y180, 4:ry(3π/2)/-Y90, 5:idle}``.
1Q Z is compiled into analog XY; these labels are physical envelopes, not
``frame_rotation`` layers. CZ flux phase compensation stays inside CZ.
"""

from __future__ import annotations

from typing import Callable

# Explicit opcode table (0–37). Prefer this over generating labels at runtime.
gate_mapping: dict[int, str] = {
    # Parallel single-qubit layers: g_control * 6 + g_target
    0: "X90(control) + X90(target)",
    1: "X90(control) + X180(target)",
    2: "X90(control) + Y90(target)",
    3: "X90(control) + Y180(target)",
    4: "X90(control) + -Y90(target)",
    5: "X90(control)",
    6: "X180(control) + X90(target)",
    7: "X180(control) + X180(target)",
    8: "X180(control) + Y90(target)",
    9: "X180(control) + Y180(target)",
    10: "X180(control) + -Y90(target)",
    11: "X180(control)",
    12: "Y90(control) + X90(target)",
    13: "Y90(control) + X180(target)",
    14: "Y90(control) + Y90(target)",
    15: "Y90(control) + Y180(target)",
    16: "Y90(control) + -Y90(target)",
    17: "Y90(control)",
    18: "Y180(control) + X90(target)",
    19: "Y180(control) + X180(target)",
    20: "Y180(control) + Y90(target)",
    21: "Y180(control) + Y180(target)",
    22: "Y180(control) + -Y90(target)",
    23: "Y180(control)",
    24: "-Y90(control) + X90(target)",
    25: "-Y90(control) + X180(target)",
    26: "-Y90(control) + Y90(target)",
    27: "-Y90(control) + Y180(target)",
    28: "-Y90(control) + -Y90(target)",
    29: "-Y90(control)",
    30: "X90(target)",
    31: "X180(target)",
    32: "Y90(target)",
    33: "Y180(target)",
    34: "-Y90(target)",
    35: "Idle (both qubits)",
    # Two-qubit layers (readout is not an opcode)
    36: "CZ gate",
    37: "Idle_2q (Wait(4) both qubits)",
}


def layer_label(opcode: int) -> str:
    """Return the human-readable label for one layer opcode."""
    return gate_mapping.get(opcode, f"Unknown opcode {opcode}")


def format_circuit(circuit_as_ints: list[int], *, include_index: bool = True) -> str:
    """Format an encoded RB circuit as an explicit layer-by-layer string."""
    lines = []
    for idx, opcode in enumerate(circuit_as_ints):
        prefix = f"{idx:4d}: " if include_index else ""
        lines.append(f"{prefix}[{opcode:2d}] {layer_label(opcode)}")
    return "\n".join(lines)


def print_circuit(
    circuit_as_ints: list[int],
    *,
    header: str | None = None,
    log_callable: Callable[[str], None] = print,
) -> None:
    """Print an encoded RB circuit with explicit per-layer opcode labels."""
    if header:
        log_callable(header)
    log_callable(format_circuit(circuit_as_ints))
