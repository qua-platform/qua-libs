"""Circuit processing utilities for two-qubit randomized benchmarking.

This module provides functions for converting quantum circuits to integer
representations and processing circuit layers for RB experiments. The single-qubit
slot table depends on the single-qubit basis: ``XY`` (analog Y) or ``XZ`` (virtual Z).
"""

from collections import Counter
from typing import Callable, List, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit

from calibration_utils.two_qubit_rb.packing import SingleQubitBasis

# Gate name to single-qubit slot per basis (6 slots; do not add aliases). Slot 5 is idle.
# XY slot 3 is physical Y: both Qiskit ``y`` and ``ry(π)`` normalize to ``"y"``.
# XZ slots 2-4 are virtual Z rotations (``rz`` is a frame change, not a pulse).
SINGLE_QUBIT_GATE_MAPS: dict[str, dict[str, int]] = {
    "XY": {"sx": 0, "x": 1, "ry(pi/2)": 2, "y": 3, "ry(3pi/2)": 4, "idle": 5},
    "XZ": {"sx": 0, "x": 1, "rz(pi/2)": 2, "rz(pi)": 3, "rz(3pi/2)": 4, "idle": 5},
}
N_SINGLE_QUBIT_SLOTS = 6

# Two-qubit gate mapping
TWO_QUBIT_GATE_MAP = {"cz": 36, "idle_2q": 37}

# Gates that are frame changes in QUA (no pulse), excluded from physical-gate counts.
VIRTUAL_1Q_GATE_NAMES: dict[str, frozenset[str]] = {
    "XY": frozenset(),
    "XZ": frozenset({"rz"}),
}

IDLE_QUBIT_GATE = SINGLE_QUBIT_GATE_MAPS["XY"]["idle"]
EMPTY_LAYER = [[IDLE_QUBIT_GATE, IDLE_QUBIT_GATE]]


def _quarter_turn_name(axis: str, angle: float) -> str:
    wrapped = float(np.mod(angle, 2.0 * np.pi))
    if np.isclose(wrapped, np.pi / 2):
        return f"{axis}(pi/2)"
    if np.isclose(wrapped, np.pi):
        return f"{axis}(pi)"
    if np.isclose(wrapped, 3.0 * np.pi / 2):
        return f"{axis}(3pi/2)"
    raise ValueError(f"Unsupported angle: {angle}")


def get_gate_name(gate, single_qubit_basis: SingleQubitBasis) -> str:
    """Extract the gate name in the slot form of *single_qubit_basis*.

    XY: ``sx``, ``x``, ``y``, and ``ry`` at Clifford angles; ``rz`` is rejected.
    XZ: ``sx``, ``x``, and ``rz`` at Clifford angles (π/2, π, 3π/2 mod 2π); ``y`` and ``ry`` are rejected.
    Leftover fractional angles (e.g. from ``optimization_level>1``) are rejected in both bases.
    """
    name = gate.name.lower()
    if single_qubit_basis == "XZ":
        if name == "rz":
            return _quarter_turn_name("rz", float(gate.params[0]))
        if name in ("y", "ry"):
            raise ValueError(f"{name} is not in the XZ RB basis; use {{cz, rz, sx, x}}")
        return name
    if name == "rz":
        raise ValueError("rz is not in the analog-XY RB basis; use {cz, sx, x, ry, y}")
    if name == "y":
        return "y"
    if name == "ry":
        angle = _quarter_turn_name("ry", float(gate.params[0]))
        return "y" if angle == "ry(pi)" else angle
    return name


def get_layer_integer(layer: Union[list[int, int], str]) -> int:
    """
    Convert a layer configuration to an integer.

    Args:
        layer: Either a tuple of two single-qubit gate integers or a two-qubit gate name

    Returns:
        Integer representing the layer configuration
    """
    if isinstance(layer[0], list):
        # For single-qubit gate pairs: g0 * N_SINGLE_QUBIT_SLOTS + g1 → [0, 35]
        return layer[0][0] * N_SINGLE_QUBIT_SLOTS + layer[0][1]
    if isinstance(layer[0], str):
        return TWO_QUBIT_GATE_MAP[layer[0]]
    raise ValueError(f"Unsupported layer : {layer[0]}")


def _instructions_to_layer_int(instructions: QuantumCircuit, single_qubit_basis: SingleQubitBasis) -> int:
    """Encode one parallel gate layer as a single ``play_gate`` integer.

    Each DAG layer represents gates that can execute simultaneously on the
    two-qubit pair. This function collects the gate on each qubit (or the
    single two-qubit gate) and packs them into one opcode consumed by
    :func:`~calibration_utils.two_qubit_rb.qua_utils.play_gate`.

    Single-qubit layers are encoded as ``g0 * N_SINGLE_QUBIT_SLOTS + g1``
    where ``g0``/``g1`` index into ``SINGLE_QUBIT_GATE_MAPS[single_qubit_basis]``
    (5 = idle). Two-qubit layers map to :data:`TWO_QUBIT_GATE_MAP` (36 = ``cz``,
    37 = ``idle_2q``).

    Args:
        instructions: One layer of the circuit, typically a small
            :class:`~qiskit.circuit.QuantumCircuit` from
            :func:`~qiskit.converters.dag_to_circuit` on a single
            :meth:`~qiskit.dagcircuit.DAGCircuit.layers` entry.
        single_qubit_basis: ``XY`` or ``XZ`` gate set for the single-qubit slots.

    Returns:
        Integer opcode in ``[0, 35]`` for parallel single-qubit layers, or
        ``36``/``37`` for two-qubit layers.

    Raises:
        ValueError: If a gate is unsupported in *single_qubit_basis*, or if a
            two-qubit gate appears in the same layer as single-qubit gates.
    """
    slots = SINGLE_QUBIT_GATE_MAPS[single_qubit_basis]
    current_layer = [row[:] for row in EMPTY_LAYER]
    for instruction in instructions:
        gate_name = get_gate_name(instruction.operation, single_qubit_basis)

        if gate_name in ("cz", "idle_2q"):
            if current_layer != EMPTY_LAYER:
                raise ValueError(f"{gate_name} gate found with single qubit gates in the layer")
            current_layer = [gate_name]
        elif gate_name in slots:
            qubit_index = instruction.qubits[0]._index
            current_layer[0][qubit_index] = slots[gate_name]
        else:
            raise ValueError(f"Unsupported gate: {gate_name}")

    return get_layer_integer(current_layer)


def circuit_to_layer_ints(qc: QuantumCircuit, single_qubit_basis: SingleQubitBasis) -> List[int]:
    """Convert a transpiled 2-qubit circuit to ``play_gate`` integer opcodes.

    Uses :func:`~qiskit.converters.circuit_to_dag` and
    :meth:`~qiskit.dagcircuit.DAGCircuit.layers` to walk parallel time steps
    in order, encoding each layer via :func:`_instructions_to_layer_int`.
    Empty DAG layers and barrier-only layers are skipped, so this works
    directly on circuits produced by
    :meth:`~calibration_utils.two_qubit_rb.rb_utils.RBBase.transpile_per_clifford_barriered`
    (which leaves barriers in place to preserve per-Clifford isolation
    during transpilation).

    Args:
        qc: Transpiled two-qubit circuit in the basis of *single_qubit_basis*
            (possibly containing barriers).
        single_qubit_basis: ``XY`` or ``XZ``.

    Returns:
        Ordered list of layer opcodes in ``[0, 37]``, one per parallel
        timestep. Gate-only: does not append a readout marker. The executor
        reads out once per circuit after the last gate (empty lists included).

    Raises:
        ValueError: Propagated from :func:`_instructions_to_layer_int` if any
            layer contains unsupported (non-barrier) gates or invalid gate
            combinations.
    """
    dag = circuit_to_dag(qc)
    result = []
    for layer in dag.layers():
        layer_circ = dag_to_circuit(layer["graph"])
        # Drop barrier-only layers (and any leftover empty layers).
        non_barrier_instructions = [instr for instr in layer_circ if instr.operation.name != "barrier"]
        if not non_barrier_instructions:
            continue
        result.append(_instructions_to_layer_int(non_barrier_instructions, single_qubit_basis))
    return result


def _count_transpiled_circuit_stats(qc: QuantumCircuit, single_qubit_basis: SingleQubitBasis) -> dict:
    """Return Clifford, 1Q, and CZ gate counts for one barriered transpiled circuit.

    Returns both full transpiled gate totals and physical gate totals. Virtual
    1Q gates (``rz`` in XZ) are excluded from the physical totals.
    """
    gate_counts = Counter(instr.operation.name for instr in qc)
    n_cliffords = gate_counts.pop("barrier", 0)
    two_qubit_names = {"cz", "idle_2q"}
    virtual_names = VIRTUAL_1Q_GATE_NAMES[single_qubit_basis]
    n_transpiled_gates = sum(gate_counts.values())
    n_transpiled_physical_gates = sum(count for name, count in gate_counts.items() if name not in virtual_names)
    n_1q_gates = sum(
        count for name, count in gate_counts.items() if name not in two_qubit_names and name not in virtual_names
    )
    n_cz_gates = gate_counts.get("cz", 0)
    return {
        "n_cliffords": n_cliffords,
        "n_transpiled_gates": n_transpiled_gates,
        "n_transpiled_physical_gates": n_transpiled_physical_gates,
        "n_1q_gates": n_1q_gates,
        "n_cz_gates": n_cz_gates,
    }


def log_depth_summary(summary: dict, *, log_callable: Callable[[str], None] = print) -> None:
    """Log a one-line per-depth transpile summary (e.g. from cache)."""
    log_callable(
        f"depth={summary['depth']}: {summary['num_circuits']} circuits, "
        f"{summary['total_cliffords']} cliffords, "
        f"n_total_transpiled_gates={summary['n_total_transpiled_gates']}, "
        f"n_total_transpiled_physical_gates={summary['n_total_transpiled_physical_gates']}, "
        f"avg physical gates/Clifford={summary['avg_physical_gates_per_clifford']:.2f}, "
        f"avg 1Q/Clifford={summary['avg_1q_per_clifford']:.2f}, "
        f"avg CZ/Clifford={summary['avg_cz_per_clifford']:.2f}"
    )


def summarize_transpiled_depth(
    depth: int,
    circuits: list[QuantumCircuit],
    *,
    single_qubit_basis: SingleQubitBasis,
    log_callable: Callable[[str], None] = print,
) -> dict:
    """Compute, log, and return per-depth transpile stats for caching.

    Args:
        depth: RB depth (number of random Cliffords).
        circuits: Transpiled circuits at this depth.
        single_qubit_basis: ``XY`` or ``XZ``; decides which 1Q gates count as virtual.
        log_callable: Callable used for log output (e.g. ``node.log``).

    Returns:
        Summary dict suitable for ``log_depth_summary`` or ``depth_summaries`` cache.
    """
    total_cliffords = 0
    n_total_transpiled_gates = 0
    n_total_transpiled_physical_gates = 0
    total_1q_gates = 0
    total_cz_gates = 0

    for qc in circuits:
        stats = _count_transpiled_circuit_stats(qc, single_qubit_basis)
        total_cliffords += stats["n_cliffords"]
        n_total_transpiled_gates += stats["n_transpiled_gates"]
        n_total_transpiled_physical_gates += stats["n_transpiled_physical_gates"]
        total_1q_gates += stats["n_1q_gates"]
        total_cz_gates += stats["n_cz_gates"]

    avg_1q_per_clifford = total_1q_gates / total_cliffords if total_cliffords else 0.0
    avg_cz_per_clifford = total_cz_gates / total_cliffords if total_cliffords else 0.0
    avg_transpiled_gates_per_clifford = n_total_transpiled_gates / total_cliffords if total_cliffords else 0.0
    avg_physical_gates_per_clifford = n_total_transpiled_physical_gates / total_cliffords if total_cliffords else 0.0

    summary = {
        "depth": depth,
        "num_circuits": len(circuits),
        "total_cliffords": total_cliffords,
        "n_total_transpiled_gates": n_total_transpiled_gates,
        "n_total_transpiled_physical_gates": n_total_transpiled_physical_gates,
        "total_1q_gates": total_1q_gates,
        "total_cz_gates": total_cz_gates,
        "avg_transpiled_gates_per_clifford": avg_transpiled_gates_per_clifford,
        "avg_physical_gates_per_clifford": avg_physical_gates_per_clifford,
        "avg_1q_per_clifford": avg_1q_per_clifford,
        "avg_cz_per_clifford": avg_cz_per_clifford,
    }
    log_depth_summary(summary, log_callable=log_callable)
    return summary
