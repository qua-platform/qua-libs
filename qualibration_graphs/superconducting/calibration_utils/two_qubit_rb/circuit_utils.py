"""Circuit processing utilities for two-qubit randomized benchmarking.

This module provides functions for converting quantum circuits to integer
representations and processing circuit layers for RB experiments.
"""

from collections import Counter
from typing import Callable, List, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit

# Gate to integer mapping for single qubit gates
SINGLE_QUBIT_GATE_MAP = {"sx": 0, "x": 1, "rz(pi/2)": 2, "rz(pi)": 3, "rz(3pi/2)": 4, "idle": 5}

# Two-qubit gate mapping
TWO_QUBIT_GATE_MAP = {"cz": 36, "idle_2q": 37}

# Opcode appended by nodes after :func:`circuit_to_layer_ints` (``play_gate`` case 38).
READOUT_OPCODE = 38

IDLE_QUBIT_GATE = SINGLE_QUBIT_GATE_MAP["idle"]
EMPTY_LAYER = [[IDLE_QUBIT_GATE, IDLE_QUBIT_GATE]]


def get_gate_name(gate) -> str:
    """Extract the gate name and parameters in a standardized format."""
    name = gate.name.lower()
    if name == "rz":
        angle = gate.params[0]
        if np.isclose(angle, np.pi / 2):
            return name + "(pi/2)"
        if np.isclose(angle, np.pi) or np.isclose(angle, -np.pi):
            return name + "(pi)"
        if np.isclose(angle, 3 * np.pi / 2) or np.isclose(angle, -np.pi / 2):
            return name + "(3pi/2)"
        raise ValueError(f"Unsupported angle: {angle}")
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
        # For single-qubit gate pairs: g0 * |SINGLE_QUBIT_GATE_MAP| + g1 → [0, 35]
        return layer[0][0] * len(SINGLE_QUBIT_GATE_MAP) + layer[0][1]
    if isinstance(layer[0], str):
        return TWO_QUBIT_GATE_MAP[layer[0]]
    raise ValueError(f"Unsupported layer : {layer[0]}")


def _instructions_to_layer_int(instructions: QuantumCircuit) -> int:
    """Encode one parallel gate layer as a single ``play_gate`` integer.

    Each DAG layer represents gates that can execute simultaneously on the
    two-qubit pair. This function collects the gate on each qubit (or the
    single two-qubit gate) and packs them into one opcode consumed by
    :func:`~calibration_utils.two_qubit_rb.qua_utils.play_gate`.

    Single-qubit layers are encoded as ``g0 * len(SINGLE_QUBIT_GATE_MAP) + g1``
    where ``g0``/``g1`` index into :data:`SINGLE_QUBIT_GATE_MAP` (5 = idle).
    Two-qubit layers map to :data:`TWO_QUBIT_GATE_MAP` (36 = ``cz``,
    37 = ``idle_2q``).

    Args:
        instructions: One layer of the circuit, typically a small
            :class:`~qiskit.circuit.QuantumCircuit` from
            :func:`~qiskit.converters.dag_to_circuit` on a single
            :meth:`~qiskit.dagcircuit.DAGCircuit.layers` entry.

    Returns:
        Integer opcode in ``[0, 35]`` for parallel single-qubit layers, or
        ``36``/``37`` for two-qubit layers.

    Raises:
        ValueError: If a gate is unsupported, or if a two-qubit gate appears
            in the same layer as single-qubit gates.
    """
    current_layer = [row[:] for row in EMPTY_LAYER]
    for instruction in instructions:
        gate_name = get_gate_name(instruction.operation)

        if gate_name in ("cz", "idle_2q"):
            if current_layer != EMPTY_LAYER:
                raise ValueError(f"{gate_name} gate found with single qubit gates in the layer")
            current_layer = [gate_name]
        elif gate_name in SINGLE_QUBIT_GATE_MAP:
            qubit_index = instruction.qubits[0]._index
            current_layer[0][qubit_index] = SINGLE_QUBIT_GATE_MAP[gate_name]
        else:
            raise ValueError(f"Unsupported gate: {gate_name}")

    return get_layer_integer(current_layer)


def circuit_to_layer_ints(qc: QuantumCircuit) -> List[int]:
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
        qc: Transpiled two-qubit circuit in the RB basis gate set
            (``sx``, ``x``, ``rz``, ``cz``), possibly containing barriers.

    Returns:
        Ordered list of layer opcodes, one per parallel timestep. Does not
        append the readout marker (:data:`READOUT_OPCODE`, 38); callers add that
        when building the full sequence for the OPX.

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
        result.append(_instructions_to_layer_int(non_barrier_instructions))
    return result


def _count_transpiled_circuit_stats(qc: QuantumCircuit) -> dict:
    """Return Clifford, 1Q, and CZ gate counts for one barriered transpiled circuit."""
    gate_counts = Counter(instr.operation.name for instr in qc)
    n_cliffords = gate_counts.pop("barrier", 0)
    n_transpiled_gates = sum(gate_counts.values())
    two_qubit_names = {"cz", "idle_2q"}
    n_1q_gates = sum(count for name, count in gate_counts.items() if name not in two_qubit_names)
    n_cz_gates = gate_counts.get("cz", 0)
    return {
        "n_cliffords": n_cliffords,
        "n_transpiled_gates": n_transpiled_gates,
        "n_1q_gates": n_1q_gates,
        "n_cz_gates": n_cz_gates,
    }


def log_depth_summary(summary: dict, *, log_callable: Callable[[str], None] = print) -> None:
    """Log a one-line per-depth transpile summary (e.g. from cache)."""
    log_callable(
        f"depth={summary['depth']}: {summary['num_circuits']} circuits, "
        f"{summary['total_cliffords']} cliffords, {summary['total_transpiled_gates']} transpiled gates, "
        f"avg gates/Clifford={summary['avg_gates_per_clifford']:.2f}, "
        f"avg 1Q/Clifford={summary['avg_1q_per_clifford']:.2f}, "
        f"avg CZ/Clifford={summary['avg_cz_per_clifford']:.2f}"
    )


def summarize_transpiled_depth(
    depth: int,
    circuits: list[QuantumCircuit],
    *,
    log_callable: Callable[[str], None] = print,
) -> dict:
    """Compute, log, and return per-depth transpile stats for caching.

    Args:
        depth: RB depth (number of random Cliffords).
        circuits: Transpiled circuits at this depth.
        log_callable: Callable used for log output (e.g. ``node.log``).

    Returns:
        Summary dict suitable for ``log_depth_summary`` or ``depth_summaries`` cache.
    """
    total_cliffords = 0
    total_transpiled_gates = 0
    total_1q_gates = 0
    total_cz_gates = 0

    for qc in circuits:
        stats = _count_transpiled_circuit_stats(qc)
        total_cliffords += stats["n_cliffords"]
        total_transpiled_gates += stats["n_transpiled_gates"]
        total_1q_gates += stats["n_1q_gates"]
        total_cz_gates += stats["n_cz_gates"]

    avg_1q_per_clifford = total_1q_gates / total_cliffords if total_cliffords else 0.0
    avg_cz_per_clifford = total_cz_gates / total_cliffords if total_cliffords else 0.0
    avg_gates_per_clifford = total_transpiled_gates / total_cliffords if total_cliffords else 0.0

    summary = {
        "depth": depth,
        "num_circuits": len(circuits),
        "total_cliffords": total_cliffords,
        "total_transpiled_gates": total_transpiled_gates,
        "total_1q_gates": total_1q_gates,
        "total_cz_gates": total_cz_gates,
        "avg_gates_per_clifford": avg_gates_per_clifford,
        "avg_1q_per_clifford": avg_1q_per_clifford,
        "avg_cz_per_clifford": avg_cz_per_clifford,
    }
    log_depth_summary(summary, log_callable=log_callable)
    return summary
