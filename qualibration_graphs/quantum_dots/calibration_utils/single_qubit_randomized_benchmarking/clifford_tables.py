"""Single-qubit Clifford group algebra for PPU-based randomized benchmarking.

This module builds the lookup tables that the QUA program uses to generate,
compose, and invert random Clifford circuits entirely on the PPU.  The
tables are computed once in Python and loaded as QUA ``declare(int, value=...)``
arrays.

Native gate set
---------------
The 24 single-qubit Cliffords are decomposed into physical and virtual gates:

    Physical (Gaussian pulses on the XY channel):
        x90, x180, -x90, -x180, y90, y180, -y90, -y180

    Virtual (frame rotations, zero duration):
        z90   (π/2 about Z)
        z180  (π about Z)
        z270  (3π/2 about Z)

    Identity:
        idle  (no-op, skipped in decomposition)

The Qiskit transpiler with ``basis_gates=["rz", "sx", "x"]`` produces
decompositions using ``{x90, x180, rz(θ)}``.  The ``rz`` angles are
snapped to the three discrete virtual-Z gates above.  Only a subset of
the 8 physical gates are reached by the default basis, but all are
available in the QUA switch/case for future basis expansions.

Gate integer encoding
---------------------
Each native gate is assigned a unique integer for efficient QUA lookup:

    0: x90     1: x180    2: -x90    3: -x180
    4: y90     5: y180    6: -y90    7: -y180
    8: z90     9: z180   10: z270   11: idle
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
    "-x180": 3,
    "y90": 4,
    "y180": 5,
    "-y90": 6,
    "-y180": 7,
    "z90": 8,
    "z180": 9,
    "z270": 10,
    "idle": 11,
}

EPS = 1e-8


# ---------------------------------------------------------------------------
# Qiskit gate → native gate name
# ---------------------------------------------------------------------------


def get_gate_name(gate) -> str:
    """Map a Qiskit gate to a native gate name.

    Parameters
    ----------
    gate : qiskit.circuit.Instruction
        A gate from a transpiled Qiskit circuit.

    Returns
    -------
    str
        One of the keys in :data:`NATIVE_GATE_MAP`.

    Raises
    ------
    ValueError
        If the gate or RZ angle is not supported.
    """
    name = gate.name.lower()

    if name == "rz":
        angle = float(gate.params[0]) % (2 * np.pi)
        if np.isclose(angle, np.pi / 2, atol=EPS):
            return "z90"
        if np.isclose(angle, np.pi, atol=EPS):
            return "z180"
        if np.isclose(angle, 3 * np.pi / 2, atol=EPS) or np.isclose(
            angle, (-np.pi / 2) % (2 * np.pi), atol=EPS
        ):
            return "z270"
        if np.isclose(angle, 0, atol=EPS) or np.isclose(angle, 2 * np.pi, atol=EPS):
            return "idle"
        raise ValueError(
            f"Unsupported RZ angle: {angle:.6f} rad ({angle * 180 / np.pi:.1f}°)"
        )

    if name == "sx":
        return "x90"
    if name == "x":
        return "x180"
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
        ``["rz", "sx", "x"]`` which produces ``{x90, x180, z90/z180/z270}``.

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
    basis_gates = basis_gates or ["rz", "sx", "x"]
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
