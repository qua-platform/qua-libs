"""Integer gate identifiers to human-readable labels for RB circuit debugging.

Opcodes match :func:`~calibration_utils.two_qubit_rb.circuit_utils.get_layer_integer`
and :func:`~calibration_utils.two_qubit_rb.qua_utils.play_gate` (0–38 encoding).

Single-qubit layers use ``g_control * 6 + g_target`` with
``g ∈ {0:sx/X90, 1:x/X180, 2:rz(π/2)/Z90, 3:rz(π)/Z180, 4:rz(3π/2)/Z270, 5:idle}``.
Z rotations are virtual (``frame_rotation`` in QUA), not physical Y pulses.
"""

from __future__ import annotations

from typing import Callable

# Explicit opcode table (0–38). Prefer this over generating labels at runtime.
gate_mapping: dict[int, str] = {
    # Parallel single-qubit layers: g_control * 6 + g_target
    0: "X90(control) + X90(target)",
    1: "X90(control) + X180(target)",
    2: "X90(control) + Z90(target)",
    3: "X90(control) + Z180(target)",
    4: "X90(control) + Z270(target)",
    5: "X90(control)",
    6: "X180(control) + X90(target)",
    7: "X180(control) + X180(target)",
    8: "X180(control) + Z90(target)",
    9: "X180(control) + Z180(target)",
    10: "X180(control) + Z270(target)",
    11: "X180(control)",
    12: "Z90(control) + X90(target)",
    13: "Z90(control) + X180(target)",
    14: "Z90(control) + Z90(target)",
    15: "Z90(control) + Z180(target)",
    16: "Z90(control) + Z270(target)",
    17: "Z90(control)",
    18: "Z180(control) + X90(target)",
    19: "Z180(control) + X180(target)",
    20: "Z180(control) + Z90(target)",
    21: "Z180(control) + Z180(target)",
    22: "Z180(control) + Z270(target)",
    23: "Z180(control)",
    24: "Z270(control) + X90(target)",
    25: "Z270(control) + X180(target)",
    26: "Z270(control) + Z90(target)",
    27: "Z270(control) + Z180(target)",
    28: "Z270(control) + Z270(target)",
    29: "Z270(control)",
    30: "X90(target)",
    31: "X180(target)",
    32: "Z90(target)",
    33: "Z180(target)",
    34: "Z270(target)",
    35: "Idle (both qubits)",
    # Two-qubit / measurement markers
    36: "CZ gate",
    37: "Idle_2q (Wait(4) both qubits)",
    38: "Readout and thermalization (both qubits)",
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
