"""Single-qubit Clifford group algebra for PPU-based randomized benchmarking.

This module builds the lookup tables that the QUA program uses to generate,
compose, and invert random Clifford circuits entirely on the PPU.  The
tables are computed once in Python and loaded as QUA ``declare(int, value=...)``
arrays.

Native gate set
---------------
The 24 single-qubit Cliffords are decomposed into physical and virtual gates:

    Physical (Gaussian pulses on the XY channel):
        x90, x180, -x90, y90, y180, -y90

    Virtual (frame rotations, zero duration):
        z90   (π/2 about Z)
        z180  (π about Z)
        z270  (3π/2 about Z)

Note: ±180° rotations about the same axis are identical Cliffords
(differ only by a global phase), so -x180 and -y180 are omitted.
Zero-angle rotations (identity) are filtered out during decomposition.

The Qiskit transpiler with ``basis_gates=["rx", "ry", "rz"]`` decomposes
each Clifford into continuous-rotation gates.  Since Cliffords only involve
multiples of π/2, the produced angles are snapped to the discrete native
gates above.  This basis allows the transpiler to use both X- and Y-axis
rotations natively, yielding shorter decompositions than the older
``["rz", "sx", "x"]`` basis which was restricted to X-axis physical gates.

Gate integer encoding
---------------------
Each native gate is assigned a unique integer for efficient QUA lookup:

    0: x90     1: x180    2: -x90
    3: y90     4: y180    5: -y90
    6: z90     7: z180    8: z270
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Clifford

# ---------------------------------------------------------------------------
# Gate integer map
# ---------------------------------------------------------------------------

NATIVE_GATE_MAP: dict[str, int] = {
    "x90": 0,
    "x180": 1,
    "-x90": 2,
    "y90": 3,
    "y180": 4,
    "-y90": 5,
    "z90": 6,
    "z180": 7,
    "z270": 8,
}

EPS = 1e-8


# ---------------------------------------------------------------------------
# Qiskit gate → native gate name
# ---------------------------------------------------------------------------


def _snap_rotation_angle(gate) -> tuple[float, str]:
    """Normalise a parameterised rotation angle to [0, 2π) and format it.

    Returns ``(angle_mod_2pi, label)`` where *label* is a human-readable
    string used in error messages.
    """
    raw = float(gate.params[0])
    angle = raw % (2 * np.pi)
    label = f"{raw:.6f} rad ({raw * 180 / np.pi:.1f}°)"
    return angle, label


_QUARTER_PI_NAMES = {
    # angle mod 2π  → gate-name pairs for the three rotation axes
    # π/2
    "x_quarter": "x90",
    "y_quarter": "y90",
    "z_quarter": "z90",
    # π
    "x_half": "x180",
    "y_half": "y180",
    "z_half": "z180",
    # 3π/2  (= −π/2 mod 2π)
    "x_three_quarter": "-x90",
    "y_three_quarter": "-y90",
    "z_three_quarter": "z270",
}


def _rotation_to_native(axis: str, angle: float, label: str) -> str:
    """Map a rotation angle (already mod 2π) on a given axis to a native gate name."""
    if np.isclose(angle, np.pi / 2, atol=EPS):
        return _QUARTER_PI_NAMES[f"{axis}_quarter"]
    if np.isclose(angle, np.pi, atol=EPS):
        return _QUARTER_PI_NAMES[f"{axis}_half"]
    if np.isclose(angle, 3 * np.pi / 2, atol=EPS):
        return _QUARTER_PI_NAMES[f"{axis}_three_quarter"]
    if np.isclose(angle, 0, atol=EPS) or np.isclose(angle, 2 * np.pi, atol=EPS):
        return "idle"
    raise ValueError(f"Unsupported R{axis.upper()} angle: {label}")


def get_gate_name(gate) -> str:
    """Map a Qiskit gate to a native gate name.

    Handles both parameterised rotation gates (``rx``, ``ry``, ``rz``)
    and named discrete gates (``sx``, ``sxdg``, ``x``, ``y``, ``id``).

    Parameters
    ----------
    gate : qiskit.circuit.Instruction
        A gate from a transpiled Qiskit circuit.

    Returns
    -------
    str
        One of the keys in :data:`NATIVE_GATE_MAP`, or ``"idle"`` for
        zero-angle rotations (filtered out by :func:`process_circuit_to_integers`).

    Raises
    ------
    ValueError
        If the gate or rotation angle is not a supported multiple of π/2.
    """
    name = gate.name.lower()

    # Parameterised rotation gates (default basis: rx, ry, rz)
    if name in ("rx", "ry", "rz"):
        angle, label = _snap_rotation_angle(gate)
        return _rotation_to_native(name[1], angle, label)

    # Named discrete gates (backward-compatible with basis ["rz", "sx", "x"])
    if name == "sx":
        return "x90"
    if name == "sxdg":
        return "-x90"
    if name == "x":
        return "x180"
    if name == "y":
        return "y180"
    if name == "id":
        return "idle"

    return name


def process_circuit_to_integers(circuit: QuantumCircuit) -> list[int]:
    """Convert a transpiled Qiskit circuit to a list of gate integers.

    Identity / idle gates are dropped (they have no physical effect).

    Parameters
    ----------
    circuit : QuantumCircuit
        A single-qubit circuit transpiled into native gates.

    Returns
    -------
    list[int]
        Gate integers for QUA switch/case execution.
    """
    result: list[int] = []
    for instruction in circuit:
        gate_name = get_gate_name(instruction.operation)
        if gate_name == "idle":
            continue
        if gate_name not in NATIVE_GATE_MAP:
            raise ValueError(f"Unsupported gate: {gate_name}")
        result.append(NATIVE_GATE_MAP[gate_name])
    return result


# ---------------------------------------------------------------------------
# Clifford group generation
# ---------------------------------------------------------------------------


def _generate_single_qubit_clifford_group() -> list[Clifford]:
    """Enumerate the 24 single-qubit Cliffords via H and S generators.

    Returns
    -------
    list[Clifford]
        Ordered list of 24 Clifford elements (index 0 = identity).

    Raises
    ------
    ValueError
        If the enumeration does not produce exactly 24 elements.
    """
    identity = Clifford.from_label("I")
    generators = [Clifford.from_label("H"), Clifford.from_label("S")]

    cliffords: list[Clifford] = [identity]
    queue: list[Clifford] = [identity]

    while queue:
        current = queue.pop(0)
        for gen in generators:
            candidate = gen @ current
            if not any(candidate == existing for existing in cliffords):
                cliffords.append(candidate)
                queue.append(candidate)

    if len(cliffords) != 24:
        raise ValueError(f"Expected 24 single-qubit Cliffords, got {len(cliffords)}")

    return cliffords


def _find_clifford_index(target: Clifford, cliffords: list[Clifford]) -> int:
    """Return the index of *target* in the Clifford list."""
    for idx, cliff in enumerate(cliffords):
        if target == cliff:
            return idx
    raise ValueError("Clifford not found in group list")


# ---------------------------------------------------------------------------
# Public API — build all tables
# ---------------------------------------------------------------------------


def build_single_qubit_clifford_tables(
    basis_gates: list[str] | None = None,
) -> dict[str, list[int] | int]:
    """Build PPU lookup tables for single-qubit randomized benchmarking.

    The tables are loaded into QUA ``declare(int, value=...)`` arrays and
    used by the PPU to compose random Cliffords, compute inverses, and
    decompose Cliffords into native gate sequences — all without host
    communication during the experiment.

    Parameters
    ----------
    basis_gates : list[str] | None
        Qiskit basis gates for transpilation.  Defaults to
        ``["rx", "ry", "rz"]`` which allows the transpiler to use all
        physical X/Y rotations and virtual Z rotations natively.

    Returns
    -------
    dict
        Keys:

        - ``num_cliffords`` (int): Always 24.
        - ``compose`` (list[int]): Flattened 24×24 composition table.
          ``compose[left * 24 + right]`` = index of ``left ∘ right``.
        - ``inverse`` (list[int]): ``inverse[i]`` = index of ``C_i^{-1}``.
        - ``decomp_flat`` (list[int]): Concatenated gate-integer sequences
          for all 24 Cliffords.
        - ``decomp_offsets`` (list[int]): Start offset into ``decomp_flat``
          for each Clifford.
        - ``decomp_lengths`` (list[int]): Length of each decomposition.
        - ``max_decomp_length`` (int): Longest decomposition across all
          Cliffords.
    """
    basis_gates = basis_gates or ["rx", "ry", "rz"]
    cliffords = _generate_single_qubit_clifford_group()
    num_cliffords = len(cliffords)

    # Inverse table
    inverse: list[int] = []
    for cliff in cliffords:
        inverse.append(_find_clifford_index(cliff.adjoint(), cliffords))

    # Composition table (flattened row-major)
    compose_flat: list[int] = []
    for cliff_left in cliffords:
        for cliff_right in cliffords:
            composed = cliff_left @ cliff_right
            compose_flat.append(_find_clifford_index(composed, cliffords))

    # Decomposition into native gates
    decomp_flat: list[int] = []
    decomp_offsets: list[int] = []
    decomp_lengths: list[int] = []

    for cliff in cliffords:
        qc = QuantumCircuit(1)
        qc.append(cliff, [0])
        transpiled = transpile(qc, basis_gates=basis_gates, optimization_level=1)
        gate_seq = process_circuit_to_integers(transpiled)

        decomp_offsets.append(len(decomp_flat))
        decomp_lengths.append(len(gate_seq))
        decomp_flat.extend(gate_seq)

    return {
        "num_cliffords": num_cliffords,
        "compose": compose_flat,
        "inverse": inverse,
        "decomp_flat": decomp_flat,
        "decomp_offsets": decomp_offsets,
        "decomp_lengths": decomp_lengths,
        "max_decomp_length": max(decomp_lengths) if decomp_lengths else 0,
    }
