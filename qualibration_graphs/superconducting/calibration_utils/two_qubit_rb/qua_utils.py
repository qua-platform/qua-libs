"""QUA program utilities for two-qubit randomized benchmarking.

This module provides functions and classes for generating and executing
QUA programs for randomized benchmarking experiments.
"""

from typing import Literal

import numpy as np
from more_itertools import flatten
from qm.qua import *
from qm.qua._expressions import QuaArrayVariable, QuaVariable
from qualang_tools.units import unit
from qualibrate import NodeParameters, QualibrationNode
from quam_config import Quam

# Padding value used to fill input-stream chunks to the declared array size.
# 65 maps to case_(65) -> idle_2q (wait(4) on both qubits in play_gate). It is
# defensive only: play_sequence is bounded by the actual chunk length so these
# slots are never executed in normal operation. Picking an in-range case_ value
# guarantees that an off-by-one would land on a known no-op rather than
# undefined behavior under switch_case(..., unsafe=True).
INPUT_STREAM_PAD_VALUE = 65

# OPX QUA variable budget is ~16000; leave headroom for streams, counters, sub_lens, etc.
OPX_QUA_VARIABLE_BUDGET = 16000


def compute_rb_circuit_memory_stats(
    circuits_as_ints: list[list[int]],
    circuit_depths: list[int],
    num_circuits_per_depth: int,
) -> dict:
    """Summarize encoded RB circuit sizes for memory validation and logging.

    The ints come from ``create_qua_program`` in nodes 37a/37b: each circuit is
    ``process_circuit_to_integers(layerize_quantum_circuit(qc)) + [66]``, i.e. one
    integer per transpiled parallel gate layer plus a trailing readout marker.

    ``circuits_as_ints`` is depth-major: all sequences for depth[0], then depth[1], ...

    Returns:
        Dict with keys ``num_circuits``, ``total_ints``, ``max_circuit_ints``,
        ``max_circuit_depth``, and ``per_depth`` (list of per-depth dicts).
    """
    expected = len(circuit_depths) * num_circuits_per_depth
    if len(circuits_as_ints) != expected:
        raise ValueError(
            "circuits_as_ints length does not match circuit_depths x num_circuits_per_depth: "
            f"got {len(circuits_as_ints)} circuits, expected {expected}."
        )

    lengths = [len(circuit) for circuit in circuits_as_ints]
    max_idx = int(np.argmax(lengths))
    max_depth_idx = max_idx // num_circuits_per_depth

    per_depth = []
    for depth_idx, depth in enumerate(circuit_depths):
        start = depth_idx * num_circuits_per_depth
        end = start + num_circuits_per_depth
        depth_lengths = lengths[start:end]
        per_depth.append(
            {
                "depth": depth,
                "num_circuits": len(depth_lengths),
                "min_ints": min(depth_lengths),
                "max_ints": max(depth_lengths),
                "mean_ints": float(np.mean(depth_lengths)),
            }
        )

    return {
        "num_circuits": len(circuits_as_ints),
        "total_ints": sum(lengths),
        "max_circuit_ints": lengths[max_idx],
        "max_circuit_depth": circuit_depths[max_depth_idx],
        "per_depth": per_depth,
    }


def format_per_depth_memory_summary(per_depth: list[dict]) -> str:
    """Format per-depth int-count stats for logs and error messages."""
    lines = ["Per depth (ints per circuit: min–max, mean):"]
    for entry in per_depth:
        lines.append(
            f"  depth={entry['depth']}: {entry['num_circuits']} circuits, "
            f"{entry['min_ints']}–{entry['max_ints']} ints "
            f"(mean {entry['mean_ints']:.1f})"
        )
    return "\n".join(lines)


def format_per_depth_chunk_summary(
    chunks_per_depth: list[list[list[int]]],
    circuit_depths: list[int],
) -> str:
    """Format per-depth input-stream sub-chunk sizes for logs."""
    lines = ["Per depth input-stream sub-chunks (ints per sub-chunk):"]
    total_sub_chunks = 0
    for depth, sub_chunks in zip(circuit_depths, chunks_per_depth):
        chunk_lengths = [len(sc) for sc in sub_chunks]
        total_sub_chunks += len(sub_chunks)
        if len(chunk_lengths) == 1:
            sizes = str(chunk_lengths[0])
        else:
            sizes = " + ".join(str(n) for n in chunk_lengths)
        lines.append(
            f"  depth={depth}: {len(sub_chunks)} sub-chunk(s), "
            f"{sizes} ints (depth total {sum(chunk_lengths)})"
        )
    lines.append(f"  → {total_sub_chunks} sub-chunk(s) total (one host push per sub-chunk per pair)")
    return "\n".join(lines)


def log_rb_circuit_memory_stats(
    stats: dict,
    *,
    use_input_stream: bool,
    max_chunk_ints: int,
    declared_size: int | None = None,
    chunks_per_depth: list[list[list[int]]] | None = None,
    circuit_depths: list[int] | None = None,
    verbose: bool = False,
    log_callable=print,
) -> None:
    """Log a one-line summary; optional per-depth breakdown when ``verbose``."""
    stream_extra = ""
    if use_input_stream and declared_size is not None:
        stream_extra = f", input_stream declared_size={declared_size}"
    log_callable(
        "RB circuit memory: "
        f"{stats['num_circuits']} circuits, "
        f"{stats['total_ints']} total ints, "
        f"largest circuit {stats['max_circuit_ints']} ints "
        f"(depth={stats['max_circuit_depth']} Cliffords), "
        f"use_input_stream={use_input_stream}, "
        f"budget={max_chunk_ints}"
        f"{stream_extra}"
    )
    if not verbose:
        return
    log_callable(format_per_depth_memory_summary(stats["per_depth"]))
    if use_input_stream and chunks_per_depth is not None and circuit_depths is not None:
        log_callable(format_per_depth_chunk_summary(chunks_per_depth, circuit_depths))


def validate_without_inputstream_path(stats: dict, max_chunk_ints: int) -> None:
    """Fail fast before QUA compile when the non-input-stream declare array would exceed budget."""
    total = stats["total_ints"]
    if total <= max_chunk_ints:
        return
    raise ValueError(
        "Flattened RB sequence exceeds the OPX QUA variable budget for the "
        f"non-input-stream path: {total} ints in "
        f"declare(int, value=job_sequence), limit is max_chunk_ints={max_chunk_ints} "
        f"(OPX budget ~{OPX_QUA_VARIABLE_BUDGET}). "
        "Enable use_input_stream=True, reduce circuit_depths, or reduce "
        "num_circuits_per_depth.\n"
        f"{format_per_depth_memory_summary(stats['per_depth'])}"
    )


def split_list_by_integer_count(lst: list, max_count: int) -> list[list]:
    """
    Split a list into batches where each batch contains at most max_count items.

    Args:
        lst: The list to split.
        max_count: Maximum number of items per batch.

    Returns:
        List of lists, where each sublist has at most max_count items.
    """
    return [lst[i : i + max_count] for i in range(0, len(lst), max_count)]


def build_single_depth_chunks(
    circuits_as_ints: list[list[int]],
    circuit_depths: list[int],
    num_circuits_per_depth: int,
    max_chunk_ints: int,
    per_depth: list[dict] | None = None,
) -> tuple[list[list[list[int]]], int]:
    """Pack circuits into per-depth sub-chunks for the input-stream QUA path.

    Circuits are grouped strictly by depth (never crossing depth boundaries):
    for each depth in ``circuit_depths``, the corresponding
    ``num_circuits_per_depth`` circuits are greedily packed into sub-chunks
    such that the total int count of each sub-chunk is <= ``max_chunk_ints``.

    Args:
        circuits_as_ints: Flat list of all circuits (each itself a list of ints
            terminated by a 66 readout marker), in the same order produced by
            ``create_qua_program``: depth-major, then circuit_index within depth.
        circuit_depths: Ordered list of depths (Cliffords) being benchmarked.
            Used only to slice ``circuits_as_ints`` into per-depth groups.
        num_circuits_per_depth: Number of circuits per depth.
        max_chunk_ints: Maximum number of ints permitted per sub-chunk
            (typically slightly below the OPX 16000 variable cap to leave
            headroom for other declared variables).

    Returns:
        chunks_per_depth: ``len(circuit_depths)`` entries, one per depth. Each
            entry is a list of sub-chunks; each sub-chunk is a flat list[int]
            of concatenated full circuits (including their trailing 66 markers)
            with total length <= max_chunk_ints. No padding is applied here;
            padding to the declared input-stream size happens at push time.
        declared_size: ``max`` over all sub-chunks of their int length. This is
            the size that the input stream is declared with; smaller sub-chunks
            are padded to this size before push.

    Raises:
        ValueError: If any single circuit's int count exceeds ``max_chunk_ints``
            (single-depth chunking cannot accommodate it). The error message
            includes the offending depth, circuit index, and lengths.
    """
    if len(circuits_as_ints) != len(circuit_depths) * num_circuits_per_depth:
        raise ValueError(
            "circuits_as_ints length does not match circuit_depths x num_circuits_per_depth: "
            f"got {len(circuits_as_ints)} circuits, expected "
            f"{len(circuit_depths)} x {num_circuits_per_depth} = "
            f"{len(circuit_depths) * num_circuits_per_depth}."
        )

    chunks_per_depth: list[list[list[int]]] = []
    declared_size = 0
    if per_depth is None:
        per_depth = compute_rb_circuit_memory_stats(
            circuits_as_ints, circuit_depths, num_circuits_per_depth
        )["per_depth"]

    for depth_idx, depth in enumerate(circuit_depths):
        start = depth_idx * num_circuits_per_depth
        end = start + num_circuits_per_depth
        circuits_for_this_depth = circuits_as_ints[start:end]

        sub_chunks: list[list[int]] = []
        current_chunk: list[int] = []

        for circ_idx, circuit in enumerate(circuits_for_this_depth):
            if len(circuit) > max_chunk_ints:
                raise ValueError(
                    f"Single circuit too large for input-stream chunk: depth={depth} "
                    f"Cliffords, circuit_index={circ_idx}, circuit_ints={len(circuit)}, "
                    f"max_chunk_ints={max_chunk_ints}. Reduce depth, raise "
                    "max_chunk_ints (must stay < 16000), or reduce "
                    f"num_circuits_per_depth.\n"
                    f"{format_per_depth_memory_summary(per_depth)}"
                )
            if len(current_chunk) + len(circuit) > max_chunk_ints:
                sub_chunks.append(current_chunk)
                current_chunk = []
            current_chunk.extend(circuit)

        if current_chunk:
            sub_chunks.append(current_chunk)

        for sc in sub_chunks:
            declared_size = max(declared_size, len(sc))

        chunks_per_depth.append(sub_chunks)

    return chunks_per_depth, declared_size


def reset_qubits(node, control: Quam.qubit_type, target: Quam.qubit_type, thermalization_time: float | None = None):
    """
    Reset both control and target qubits using the node's reset type.

    Args:
        node: The qualibration node containing reset parameters.
        control: The control qubit to reset.
        target: The target qubit to reset.
        thermalization_time: Optional thermalization time (currently unused).
    """
    control.reset(reset_type=node.parameters.reset_type)
    target.reset(reset_type=node.parameters.reset_type)


def play_gate(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-statements
    gate: QuaVariable,
    qubit_pair: Quam.qubit_pair_type,
    state: QuaVariable,
    state_control: QuaVariable,
    state_target: QuaVariable,
    state_st: "_ResultSource",
    reset_type: Literal["thermal", "active"],
    cz_operation: str = "cz_unipolar",
):
    """
    Play a single gate from the gate mapping based on the gate integer value.

    Args:
        gate: Integer variable representing the gate to play.
        qubit_pair: The qubit pair on which to apply the gate.
        state: Variable to store the combined 2-qubit state.
        state_control: Variable to store the control qubit state.
        state_target: Variable to store the target qubit state.
        state_st: Stream to save the state measurement.
        reset_type: Type of reset to use ("thermal" or "active").
        cz_operation: Name of the CZ operation macro to use.
    """
    with switch_(gate, unsafe=True):

        with case_(0):
            qubit_pair.qubit_control.xy.play("x90")
            qubit_pair.qubit_target.xy.play("x90")
        with case_(1):
            qubit_pair.qubit_control.xy.play("x90")
            qubit_pair.qubit_target.xy.play("x180")
        with case_(2):
            qubit_pair.qubit_control.xy.play("x90")
            qubit_pair.qubit_target.xy.play("y90")
        with case_(3):
            qubit_pair.qubit_control.xy.play("x90")
            qubit_pair.qubit_target.xy.play("y180")
        with case_(4):
            qubit_pair.qubit_control.xy.play("x90")
            qubit_pair.qubit_target.xy.frame_rotation(np.pi / 2)
        with case_(5):
            qubit_pair.qubit_control.xy.play("x90")
            qubit_pair.qubit_target.xy.frame_rotation(np.pi)
        with case_(6):
            qubit_pair.qubit_control.xy.play("x90")
            qubit_pair.qubit_target.xy.frame_rotation(3 * np.pi / 2)
        with case_(7):
            qubit_pair.qubit_control.xy.play("x90")
        with case_(8):
            qubit_pair.qubit_control.xy.play("x180")
            qubit_pair.qubit_target.xy.play("x90")
        with case_(9):
            qubit_pair.qubit_control.xy.play("x180")
            qubit_pair.qubit_target.xy.play("x180")
        with case_(10):
            qubit_pair.qubit_control.xy.play("x180")
            qubit_pair.qubit_target.xy.play("y90")
        with case_(11):
            qubit_pair.qubit_control.xy.play("x180")
            qubit_pair.qubit_target.xy.play("y180")
        with case_(12):
            qubit_pair.qubit_control.xy.play("x180")
            qubit_pair.qubit_target.xy.frame_rotation(np.pi / 2)
        with case_(13):
            qubit_pair.qubit_control.xy.play("x180")
            qubit_pair.qubit_target.xy.frame_rotation(np.pi)
        with case_(14):
            qubit_pair.qubit_control.xy.play("x180")
            qubit_pair.qubit_target.xy.frame_rotation(3 * np.pi / 2)
        with case_(15):
            qubit_pair.qubit_control.xy.play("x180")
        with case_(16):
            qubit_pair.qubit_control.xy.play("y90")
            qubit_pair.qubit_target.xy.play("x90")
        with case_(17):
            qubit_pair.qubit_control.xy.play("y90")
            qubit_pair.qubit_target.xy.play("x180")
        with case_(18):
            qubit_pair.qubit_control.xy.play("y90")
            qubit_pair.qubit_target.xy.play("y90")
        with case_(19):
            qubit_pair.qubit_control.xy.play("y90")
            qubit_pair.qubit_target.xy.play("y180")
        with case_(20):
            qubit_pair.qubit_control.xy.play("y90")
            qubit_pair.qubit_target.xy.frame_rotation(np.pi / 2)
        with case_(21):
            qubit_pair.qubit_control.xy.play("y90")
            qubit_pair.qubit_target.xy.frame_rotation(np.pi)
        with case_(22):
            qubit_pair.qubit_control.xy.play("y90")
            qubit_pair.qubit_target.xy.frame_rotation(3 * np.pi / 2)
        with case_(23):
            qubit_pair.qubit_control.xy.play("y90")
        with case_(24):
            qubit_pair.qubit_control.xy.play("y180")
            qubit_pair.qubit_target.xy.play("x90")
        with case_(25):
            qubit_pair.qubit_control.xy.play("y180")
            qubit_pair.qubit_target.xy.play("x180")
        with case_(26):
            qubit_pair.qubit_control.xy.play("y180")
            qubit_pair.qubit_target.xy.play("y90")
        with case_(27):
            qubit_pair.qubit_control.xy.play("y180")
            qubit_pair.qubit_target.xy.play("y180")
        with case_(28):
            qubit_pair.qubit_control.xy.play("y180")
            qubit_pair.qubit_target.xy.frame_rotation(np.pi / 2)
        with case_(29):
            qubit_pair.qubit_control.xy.play("y180")
            qubit_pair.qubit_target.xy.frame_rotation(np.pi)
        with case_(30):
            qubit_pair.qubit_control.xy.play("y180")
            qubit_pair.qubit_target.xy.frame_rotation(3 * np.pi / 2)
        with case_(31):
            qubit_pair.qubit_control.xy.play("y180")
        with case_(32):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi / 2)
            qubit_pair.qubit_target.xy.play("x90")
        with case_(33):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi / 2)
            qubit_pair.qubit_target.xy.play("x180")
        with case_(34):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi / 2)
            qubit_pair.qubit_target.xy.play("y90")
        with case_(35):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi / 2)
            qubit_pair.qubit_target.xy.play("y180")
        with case_(36):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi / 2)
            qubit_pair.qubit_target.xy.frame_rotation(np.pi / 2)
        with case_(37):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi / 2)
            qubit_pair.qubit_target.xy.frame_rotation(np.pi)
        with case_(38):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi / 2)
            qubit_pair.qubit_target.xy.frame_rotation(3 * np.pi / 2)
        with case_(39):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi / 2)
        with case_(40):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi)
            qubit_pair.qubit_target.xy.play("x90")
        with case_(41):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi)
            qubit_pair.qubit_target.xy.play("x180")
        with case_(42):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi)
            qubit_pair.qubit_target.xy.play("y90")
        with case_(43):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi)
            qubit_pair.qubit_target.xy.play("y180")
        with case_(44):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi)
            qubit_pair.qubit_target.xy.frame_rotation(np.pi / 2)
        with case_(45):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi)
            qubit_pair.qubit_target.xy.frame_rotation(np.pi)
        with case_(46):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi)
            qubit_pair.qubit_target.xy.frame_rotation(3 * np.pi / 2)
        with case_(47):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi)
        with case_(48):
            qubit_pair.qubit_control.xy.frame_rotation(3 * np.pi / 2)
            qubit_pair.qubit_target.xy.play("x90")
        with case_(49):
            qubit_pair.qubit_control.xy.frame_rotation(3 * np.pi / 2)
            qubit_pair.qubit_target.xy.play("x180")
        with case_(50):
            qubit_pair.qubit_control.xy.frame_rotation(3 * np.pi / 2)
            qubit_pair.qubit_target.xy.play("y90")
        with case_(51):
            qubit_pair.qubit_control.xy.frame_rotation(3 * np.pi / 2)
            qubit_pair.qubit_target.xy.play("y180")
        with case_(52):
            qubit_pair.qubit_control.xy.frame_rotation(3 * np.pi / 2)
            qubit_pair.qubit_target.xy.frame_rotation(np.pi / 2)
        with case_(53):
            qubit_pair.qubit_control.xy.frame_rotation(3 * np.pi / 2)
            qubit_pair.qubit_target.xy.frame_rotation(np.pi)
        with case_(54):
            qubit_pair.qubit_control.xy.frame_rotation(3 * np.pi / 2)
            qubit_pair.qubit_target.xy.frame_rotation(3 * np.pi / 2)
        with case_(55):
            qubit_pair.qubit_control.xy.frame_rotation(3 * np.pi / 2)
        with case_(56):
            qubit_pair.qubit_target.xy.play("x90")
        with case_(57):
            qubit_pair.qubit_target.xy.play("x180")
        with case_(58):
            qubit_pair.qubit_target.xy.play("y90")
        with case_(59):
            qubit_pair.qubit_target.xy.play("y180")
        with case_(60):
            qubit_pair.qubit_target.xy.frame_rotation(np.pi / 2)
        with case_(61):
            qubit_pair.qubit_target.xy.frame_rotation(np.pi)
        with case_(62):
            qubit_pair.qubit_target.xy.frame_rotation(3 * np.pi / 2)
        with case_(63):
            qubit_pair.qubit_control.wait(20)
            qubit_pair.qubit_target.wait(20)
        with case_(64):  # CZ
            qubit_pair.macros[cz_operation].apply()
        with case_(65):  # idle_2q
            qubit_pair.qubit_control.wait(4)
            qubit_pair.qubit_target.wait(4)

        with case_(66):

            align()
            qubit_pair.qubit_control.readout_state(state_control)
            qubit_pair.qubit_target.readout_state(state_target)

            assign(state, state_control * 2 + state_target)
            save(state, state_st)

            # Initialize the qubits
            qubit_pair.qubit_control.reset(reset_type=reset_type)
            qubit_pair.qubit_target.reset(reset_type=reset_type)

            # Reset the frame of the qubits in order not to accumulate rotations
            reset_frame(qubit_pair.qubit_control.xy.name, qubit_pair.qubit_target.xy.name)

            align()


def play_sequence(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    sequence: QuaArrayVariable,
    depth: int,
    qubit_pair: Quam.qubit_pair_type,
    state: list[QuaVariable],
    state_control: QuaVariable,
    state_target: QuaVariable,
    state_st,
    reset_type: Literal["thermal", "active"],
    cz_operation: str = "cz_unipolar",
):
    """
    Play a sequence of gates up to the specified depth.

    Args:
        sequence: Array variable containing the gate sequence.
        depth: Number of gates to play from the sequence.
        qubit_pair: The qubit pair on which to apply the gates.
        state: List of variables to store the combined 2-qubit state.
        state_control: Variable to store the control qubit state.
        state_target: Variable to store the target qubit state.
        state_st: Stream to save the state measurement.
        reset_type: Type of reset to use ("thermal" or "active").
        cz_operation: Name of the CZ operation macro to use.
    """

    i = declare(int)
    with for_(i, 0, i < depth, i + 1):
        play_gate(sequence[i], qubit_pair, state, state_control, state_target, state_st, reset_type, cz_operation)


class QuaProgramHandler:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """Handler for generating QUA programs for randomized benchmarking experiments."""

    def __init__(  # pylint: disable=too-many-positional-arguments,too-many-arguments
        self,
        node: QualibrationNode,
        num_pairs: int,
        circuits_as_ints: list[list[int]],
        machine: Quam,
        qubit_pairs: list[Quam.qubit_pair_type],
    ):
        """
        Initialize the QUA program handler.

        Args:
            node: The qualibration node containing experiment parameters.
            num_pairs: Number of qubit pairs in the experiment.
            circuits_as_ints: List of circuits, each a list of ints terminated
                by a ``66`` readout marker. Order is depth-major, then
                circuit_index within depth (as produced by ``create_qua_program``).
            machine: The QUAM machine configuration.
            qubit_pairs: List of qubit pairs to benchmark.
        """

        self.u = unit(coerce_to_integer=True)
        self.node = node
        self.num_pairs = num_pairs
        self.circuits_as_ints = circuits_as_ints
        self.machine = machine
        self.qubit_pairs = qubit_pairs

        circuit_depths = list(self.node.parameters.circuit_depths)
        num_circuits_per_depth = self.node.parameters.num_circuits_per_depth
        max_chunk_ints = self.node.parameters.max_chunk_ints
        memory_stats = compute_rb_circuit_memory_stats(
            circuits_as_ints, circuit_depths, num_circuits_per_depth
        )

        self.declared_size = None
        if self.node.parameters.use_input_stream:
            self.chunks_per_depth, self.declared_size = build_single_depth_chunks(
                circuits_as_ints=self.circuits_as_ints,
                circuit_depths=circuit_depths,
                num_circuits_per_depth=num_circuits_per_depth,
                max_chunk_ints=max_chunk_ints,
                per_depth=memory_stats["per_depth"],
            )
        else:
            validate_without_inputstream_path(memory_stats, max_chunk_ints)

        log_rb_circuit_memory_stats(
            memory_stats,
            use_input_stream=self.node.parameters.use_input_stream,
            max_chunk_ints=max_chunk_ints,
            declared_size=self.declared_size,
            chunks_per_depth=self.chunks_per_depth if self.node.parameters.use_input_stream else None,
            circuit_depths=circuit_depths if self.node.parameters.use_input_stream else None,
            verbose=self.node.parameters.verbose_memory_log,
            log_callable=self.node.log,
        )

    def _get_qua_program_with_input_stream(self):
        # Flatten chunks_per_depth into a single ordered list of sub-chunks.
        # Order: depth-major, then sub_chunk_index within depth -- same order
        # the QUA program consumes them and the host pushes them in.
        flat_sub_chunks = [sc for sub_chunks in self.chunks_per_depth for sc in sub_chunks]
        sub_chunk_lengths = [len(sc) for sc in flat_sub_chunks]
        n_sub_chunks = len(flat_sub_chunks)

        with program() as rb:

            n = declare(int)
            n_st = declare_stream()
            # Replace the Python-unrolled per-sub-chunk loop with a QUA for_
            # loop so the (huge) play_gate switch_case body appears in the
            # compiled program ONCE per qubit pair, not once per sub-chunk.
            # Sub-chunk lengths live in a tiny QUA array (one int per
            # sub-chunk, typically <= a few tens of entries).
            j = declare(int)
            i = declare(int)
            sub_lens = declare(int, value=sub_chunk_lengths)

            sequence = declare_input_stream(int, name="sequence", size=self.declared_size)

            # The relevant streams
            state_control = declare(int)
            state_target = declare(int)
            state = declare(int)
            state_st = [declare_stream() for _ in range(self.num_pairs)]

            for qp in self.qubit_pairs:
                self.node.machine.initialize_qpu(target=qp.qubit_control)
                self.node.machine.initialize_qpu(target=qp.qubit_target)

            for k, qubit_pair in enumerate(self.qubit_pairs):

                # Initialize the qubits
                qubit_pair.qubit_control.reset(reset_type=self.node.parameters.reset_type)
                qubit_pair.qubit_target.reset(reset_type=self.node.parameters.reset_type)

                # Align the two elements to play the sequence after qubit initialization
                align()

                # Chunk outer, shot inner: advance once per sub-chunk, replay all
                # shots from the buffered chunk (no per-shot re-push on the host).
                with for_(j, 0, j < n_sub_chunks, j + 1):
                    advance_input_stream(sequence)
                    with for_(n, 0, n < self.node.parameters.num_shots, n + 1):
                        with for_(i, 0, i < sub_lens[j], i + 1):
                            play_gate(
                                sequence[i],
                                qubit_pair,
                                state,
                                state_control,
                                state_target,
                                state_st[k],
                                self.node.parameters.reset_type,
                                self.node.parameters.operation,
                            )
                        with if_(j + 1 == n_sub_chunks):
                            save(n, n_st)

            with stream_processing():
                n_st.save("n")
                for k in range(len(self.qubit_pairs)):
                    state_st[k].buffer(self.node.parameters.num_circuits_per_depth).buffer(
                        len(self.node.parameters.circuit_depths)
                    ).buffer(self.node.parameters.num_shots).save(f"state{k + 1}")
        return rb

    def _padded_chunks(self) -> list[list[int]]:
        """Return the flat depth-major list of sub-chunks, each padded to
        ``self.declared_size`` ints with :data:`INPUT_STREAM_PAD_VALUE`.

        This is the canonical "what to push, in what order, for one pass over
        the QUA program's sub-chunks" representation. ``push_all_chunks``
        iterates over this list once per qubit pair.
        """
        declared = self.declared_size
        return [
            sub_chunk + [INPUT_STREAM_PAD_VALUE] * (declared - len(sub_chunk))
            for sub_chunks in self.chunks_per_depth
            for sub_chunk in sub_chunks
        ]

    def push_all_chunks(self, job) -> None:
        """Push every input-stream chunk in the order the QUA program consumes them.

        Order mirrors ``_get_qua_program_with_input_stream``:
        ``for pair: for sub_chunk: advance``. Shots replay each chunk on the
        OPX without additional host pushes.

        Each push is padded out to ``self.declared_size`` ints with
        :data:`INPUT_STREAM_PAD_VALUE`. Padding bytes are never executed by the
        OPX (the QUA inner loop is bounded by the real sub-chunk length, stored
        in a small QUA array).

        Args:
            job: The running ``QmJob`` returned by ``qm.execute(...)``.
        """
        if not self.node.parameters.use_input_stream:
            raise RuntimeError(
                "push_all_chunks called but use_input_stream is False; "
                "the QUA program does not declare an input stream."
            )

        padded = self._padded_chunks()
        for _ in self.qubit_pairs:
            for chunk in padded:
                job.push_to_input_stream("sequence", chunk)

    def _get_qua_program_without_input_stream(self):

        job_sequence = list(flatten(self.circuits_as_ints))
        sequence_length = len(job_sequence)

        with program() as rb:

            n = declare(int)
            n_st = declare_stream()

            job_sequence_qua = declare(int, value=job_sequence)

            # The relevant streams
            state_control = declare(int)
            state_target = declare(int)
            state = declare(int)
            state_st = [declare_stream() for _ in range(self.num_pairs)]

            # Bring the active qubits to the desired frequency point
            for qp in self.qubit_pairs:
                self.node.machine.initialize_qpu(target=qp.qubit_control)
                self.node.machine.initialize_qpu(target=qp.qubit_target)

            for i, qubit_pair in enumerate(self.qubit_pairs):

                # Initialize the qubits
                qubit_pair.qubit_control.reset(reset_type=self.node.parameters.reset_type)
                qubit_pair.qubit_target.reset(reset_type=self.node.parameters.reset_type)

                # Align the two elements to play the sequence after qubit initialization
                align()

                with for_(n, 0, n < self.node.parameters.num_shots, n + 1):

                    play_sequence(
                        job_sequence_qua,
                        sequence_length,
                        qubit_pair,
                        state,
                        state_control,
                        state_target,
                        state_st[i],
                        self.node.parameters.reset_type,
                        self.node.parameters.operation,
                    )

                    save(n, n_st)

            with stream_processing():
                n_st.save("n")
                for i in range(len(self.qubit_pairs)):
                    state_st[i].buffer(self.node.parameters.num_circuits_per_depth).buffer(
                        len(self.node.parameters.circuit_depths)
                    ).buffer(self.node.parameters.num_shots).save(f"state{i + 1}")
        return rb

    def get_qua_program(self):
        """
        Get the appropriate QUA program based on input stream configuration.

        Returns:
            The QUA program for execution.
        """
        if self.node.parameters.use_input_stream:
            return self._get_qua_program_with_input_stream()
        return self._get_qua_program_without_input_stream()
