"""QUA program utilities for two-qubit randomized benchmarking.

Gate playback uses an unsafe switch over opcodes 0–37 only. Readout, science
save, reset, and frame cleanup run once per circuit outside that switch.
"""

from __future__ import annotations

import inspect
from typing import Literal

import numpy as np
from qm.qua import *
from qm.qua._expressions import QuaArrayVariable, QuaVariable
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from quam.components.pulses import SquarePulse
from quam_config import Quam

from .packing import (
    build_packed_depth_chunks,
    flatten_padded_chunks,
    pack_circuits,
    packed_packet_words,
    validate_circuit_list,
)

# Padding value used to fill input-stream chunks to the declared array size.
# 37 maps to case_(37) -> idle_2q (wait(4) on both qubits). It is defensive only:
# packed circuit ranges never include trailing pad words, so these slots are not
# played. Picking an in-range case_ value guarantees that an off-by-one would
# land on a known no-op rather than undefined behavior under switch_(..., unsafe=True).
INPUT_STREAM_PAD_VALUE = 37

# OPX QUA variable budget is ~16000; leave headroom for streams, counters, etc.
OPX_QUA_VARIABLE_BUDGET = 16000

_ALIGN_ELEMENTS_KW = "align_elements"


class CzAlignElementsUnavailable(RuntimeError):
    """Installed CZ macro cannot opt out of explicit aligns (would be a silent no-op)."""


def _iter_qubit_pairs(qubit_pairs):
    """Yield pair objects from a list, dict, or QUAM pair collection."""
    if isinstance(qubit_pairs, dict):
        yield from qubit_pairs.values()
        return
    values = getattr(qubit_pairs, "values", None)
    if callable(values):
        yielded = list(values())
        if yielded and not isinstance(yielded[0], (int, str)):
            yield from yielded
            return
    yield from qubit_pairs


def require_cz_align_elements(cz_macro, *, cz_operation: str, pair_label: str) -> None:
    """Fail if ``apply`` would swallow ``align_elements=False`` via ``**kwargs``."""
    apply = getattr(cz_macro, "apply", None)
    if apply is None:
        raise CzAlignElementsUnavailable(
            f"CZ operation {cz_operation!r} on {pair_label} has no apply() method."
        )
    try:
        params = inspect.signature(apply).parameters
    except (TypeError, ValueError) as exc:
        raise CzAlignElementsUnavailable(
            f"Cannot inspect {cz_operation!r}.apply on {pair_label}: {exc}."
        ) from exc
    if _ALIGN_ELEMENTS_KW not in params:
        raise CzAlignElementsUnavailable(
            f"CZ operation {cz_operation!r} on {pair_label} does not accept "
            f"apply({_ALIGN_ELEMENTS_KW}=False). The installed quam-builder revision "
            "likely swallows unknown keywords via **kwargs, so the RB dispatcher would "
            "still emit explicit CZ aligns. Install a quam-builder revision that adds "
            "the explicit align_elements keyword (hotfix/cz-align-elements-opt-in)."
        )


def preflight_cz_align_elements(qubit_pairs, cz_operation: str) -> None:
    """Require the CZ opt-in on every pair before tracing the unsafe switch."""
    for qp in _iter_qubit_pairs(qubit_pairs):
        pair_label = str(getattr(qp, "name", qp))
        macros = getattr(qp, "macros", None)
        if macros is None or cz_operation not in macros:
            raise CzAlignElementsUnavailable(
                f"Qubit pair {pair_label} has no macro {cz_operation!r}; "
                "cannot verify align_elements support."
            )
        require_cz_align_elements(macros[cz_operation], cz_operation=cz_operation, pair_label=pair_label)


_XY_ZERO_OP = "zero"
_XY_ZERO_LEN_NS = 4


def _xy_channels_for_pair(qp):
    """Control and target XY lines of a qubit pair (spectators are not occupied)."""
    for attr in ("qubit_control", "qubit_target"):
        qubit = getattr(qp, attr, None)
        if qubit is None:
            continue
        xy = getattr(qubit, "xy", None)
        if xy is not None:
            yield xy


def ensure_xy_zero_pulse(qubit_pairs) -> None:
    """Attach a 4 ns amp-0 XY ``"zero"`` pulse when a pair's XY ops lack it.

    Trace-time only. Needed so ``play("zero")`` in the CZ unsafe-switch case
    compiles on machines that do not ship the sanitized occupancy pulse.
    Idempotent: existing ``"zero"`` operations are left unchanged.
    """
    for qp in _iter_qubit_pairs(qubit_pairs):
        for xy in _xy_channels_for_pair(qp):
            operations = getattr(xy, "operations", None)
            if operations is None or _XY_ZERO_OP in operations:
                continue
            operations[_XY_ZERO_OP] = SquarePulse(length=_XY_ZERO_LEN_NS, amplitude=0)


def compute_rb_circuit_memory_stats(
    circuits_as_ints: list[list[int]],
    circuit_depths: list[int],
    num_circuits_per_depth: int,
) -> dict:
    """Summarize encoded RB circuit sizes for memory validation and logging.

    ``circuits_as_ints`` is gate-only (opcodes 0–37), depth-major: all sequences
    for depth[0], then depth[1], ... Headers are not stored in these lists.

    Returns:
        Dict with keys ``num_circuits``, ``total_ints`` (packed packet words),
        ``total_gates``, ``max_circuit_ints``, ``max_circuit_depth``, and
        ``per_depth``.
    """
    gates = validate_circuit_list(
        circuits_as_ints,
        circuit_depths=circuit_depths,
        num_circuits_per_depth=num_circuits_per_depth,
    )

    lengths = [len(circuit) for circuit in gates]
    max_idx = int(np.argmax(lengths)) if lengths else 0
    max_depth_idx = max_idx // num_circuits_per_depth if num_circuits_per_depth else 0

    per_depth = []
    for depth_idx, depth in enumerate(circuit_depths):
        start = depth_idx * num_circuits_per_depth
        end = start + num_circuits_per_depth
        depth_circuits = gates[start:end]
        depth_lengths = lengths[start:end]
        per_depth.append(
            {
                "depth": depth,
                "num_circuits": len(depth_lengths),
                "min_ints": min(depth_lengths) if depth_lengths else 0,
                "max_ints": max(depth_lengths) if depth_lengths else 0,
                "mean_ints": float(np.mean(depth_lengths)) if depth_lengths else 0.0,
                "packed_words": packed_packet_words(depth_circuits) if depth_circuits else 0,
            }
        )

    return {
        "num_circuits": len(gates),
        "total_ints": packed_packet_words(gates) if gates else 1,
        "total_gates": sum(lengths),
        "max_circuit_ints": lengths[max_idx] if lengths else 0,
        "max_circuit_depth": circuit_depths[max_depth_idx] if circuit_depths else 0,
        "per_depth": per_depth,
    }


def format_per_depth_memory_summary(per_depth: list[dict]) -> str:
    """Format per-depth gate-count stats for logs and error messages."""
    lines = ["Per depth (gates per circuit: min–max, mean):"]
    for entry in per_depth:
        lines.append(
            f"  depth={entry['depth']}: {entry['num_circuits']} circuits, "
            f"{entry['min_ints']}–{entry['max_ints']} gates "
            f"(mean {entry['mean_ints']:.1f})"
        )
    return "\n".join(lines)


def format_per_depth_chunk_summary(
    chunks_per_depth: list[list[list[int]]],
    circuit_depths: list[int],
) -> str:
    """Format per-depth packed input-stream sub-chunk sizes for logs."""
    lines = ["Per depth packed input-stream sub-chunks (words per sub-chunk):"]
    total_sub_chunks = 0
    for depth, sub_chunks in zip(circuit_depths, chunks_per_depth):
        chunk_lengths = [len(sc) for sc in sub_chunks]
        total_sub_chunks += len(sub_chunks)
        if len(chunk_lengths) == 1:
            sizes = str(chunk_lengths[0])
        else:
            sizes = " + ".join(str(n) for n in chunk_lengths)
        lines.append(
            f"  depth={depth}: {len(sub_chunks)} sub-chunk(s), " f"{sizes} words (depth total {sum(chunk_lengths)})"
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
    """Log RB circuit memory summary; optional per-depth breakdown when ``verbose``."""
    stream_extra = ""
    if use_input_stream and declared_size is not None:
        stream_extra = f", input_stream declared_size={declared_size}"
    log_callable(
        "RB circuit OPX memory summary stats: "
        f"{stats['num_circuits']} circuits, "
        f"{stats['total_gates']} gates, "
        f"{stats['total_ints']} packed words, "
        f"largest circuit {stats['max_circuit_ints']} gates "
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
    """Fail fast before QUA compile when the non-input-stream packed array would exceed budget."""
    total = stats["total_ints"]
    if total <= max_chunk_ints:
        return
    raise ValueError(
        "Packed RB sequence exceeds the OPX QUA variable budget for the "
        f"non-input-stream path: {total} packed words in "
        f"declare(int, value=packed_sequence), limit is max_chunk_ints={max_chunk_ints} "
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
    """Pack gate-only circuits into per-depth length-delimited input-stream chunks.

    Wrapper around :func:`build_packed_depth_chunks`. ``per_depth`` is accepted
    for call-site compatibility and is unused (packing validates on its own).
    """
    del per_depth
    return build_packed_depth_chunks(
        circuits_as_ints,
        circuit_depths,
        num_circuits_per_depth,
        max_chunk_ints,
    )


def play_gate(
    gate: QuaVariable,
    qubit_pair: Quam.qubit_pair_type,
    cz_operation: str = "cz_unipolar",
):
    """Play one coherent RB layer (opcodes 0–37) with no readout or explicit align.

    1Q layers are analog XY only: ``x90``/``x180``/``y90``/``y180``/``-y90``.
    CZ uses ``apply(align_elements=False)`` after :func:`preflight_cz_align_elements`,
    then ``play("zero")`` on both XY so the compiler occupies those elements for
    the flux case (frame-only compensation has no analog envelope).
    Circuit-boundary align/readout/reset lives in :func:`readout_save_and_reset`.
    """
    preflight_cz_align_elements(qubit_pair, cz_operation)

    with switch_(gate, unsafe=True):

        with case_(0):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("x90")
                qp.qubit_target.xy.play("x90")
        with case_(1):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("x90")
                qp.qubit_target.xy.play("x180")
        with case_(2):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("x90")
                qp.qubit_target.xy.play("y90")
        with case_(3):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("x90")
                qp.qubit_target.xy.play("y180")
        with case_(4):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("x90")
                qp.qubit_target.xy.play("-y90")
        with case_(5):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("x90")
        with case_(6):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("x180")
                qp.qubit_target.xy.play("x90")
        with case_(7):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("x180")
                qp.qubit_target.xy.play("x180")
        with case_(8):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("x180")
                qp.qubit_target.xy.play("y90")
        with case_(9):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("x180")
                qp.qubit_target.xy.play("y180")
        with case_(10):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("x180")
                qp.qubit_target.xy.play("-y90")
        with case_(11):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("x180")
        with case_(12):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("y90")
                qp.qubit_target.xy.play("x90")
        with case_(13):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("y90")
                qp.qubit_target.xy.play("x180")
        with case_(14):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("y90")
                qp.qubit_target.xy.play("y90")
        with case_(15):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("y90")
                qp.qubit_target.xy.play("y180")
        with case_(16):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("y90")
                qp.qubit_target.xy.play("-y90")
        with case_(17):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("y90")
        with case_(18):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("y180")
                qp.qubit_target.xy.play("x90")
        with case_(19):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("y180")
                qp.qubit_target.xy.play("x180")
        with case_(20):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("y180")
                qp.qubit_target.xy.play("y90")
        with case_(21):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("y180")
                qp.qubit_target.xy.play("y180")
        with case_(22):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("y180")
                qp.qubit_target.xy.play("-y90")
        with case_(23):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("y180")
        with case_(24):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("-y90")
                qp.qubit_target.xy.play("x90")
        with case_(25):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("-y90")
                qp.qubit_target.xy.play("x180")
        with case_(26):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("-y90")
                qp.qubit_target.xy.play("y90")
        with case_(27):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("-y90")
                qp.qubit_target.xy.play("y180")
        with case_(28):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("-y90")
                qp.qubit_target.xy.play("-y90")
        with case_(29):
            for qp in qubit_pair.values():
                qp.qubit_control.xy.play("-y90")
        with case_(30):
            for qp in qubit_pair.values():
                qp.qubit_target.xy.play("x90")
        with case_(31):
            for qp in qubit_pair.values():
                qp.qubit_target.xy.play("x180")
        with case_(32):
            for qp in qubit_pair.values():
                qp.qubit_target.xy.play("y90")
        with case_(33):
            for qp in qubit_pair.values():
                qp.qubit_target.xy.play("y180")
        with case_(34):
            for qp in qubit_pair.values():
                qp.qubit_target.xy.play("-y90")
        with case_(35):  # idle gate
            for qp in qubit_pair.values():
                qp.qubit_control.wait(4)
                qp.qubit_target.wait(4)
        with case_(36):  # CZ
            for qp in qubit_pair.values():
                qp.macros[cz_operation].apply(align_elements=False)
                qp.qubit_control.xy.play("zero")
                qp.qubit_target.xy.play("zero")
        with case_(37):  # idle_2q
            for qp in qubit_pair.values():
                qp.qubit_control.wait(4)
                qp.qubit_target.wait(4)


def readout_save_and_reset(
    qubit_pair: Quam.qubit_pair_type,
    state: QuaVariable,
    state_control: QuaVariable,
    state_target: QuaVariable,
    state_st,
    reset_type: Literal["thermal", "active"],
    simulate: bool = False,
):
    """Science readout, save, reset, and frame cleanup for one circuit repetition.

    Aligns all participating resources after the last coherent pulse (XY-only
    align is not enough when CZ flux / spectators may still be active). Empty
    circuits still call this helper once.
    """
    align()
    for i, qp in qubit_pair.items():
        qp.qubit_control.readout_state(state_control)
        qp.qubit_target.readout_state(state_target)
        assign(state, state_control * 2 + state_target)
        save(state, state_st[i])
    align()

    for qp in qubit_pair.values():
        qp.qubit_control.reset(reset_type, simulate)
        qp.qubit_target.reset(reset_type, simulate)
        reset_frame(qp.qubit_control.xy.name, qp.qubit_target.xy.name)
    align()


def play_sequence(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    sequence: QuaArrayVariable,
    circuit_start,
    circuit_stop,
    qubit_pair: Quam.qubit_pair_type,
    state: QuaVariable,
    state_control: QuaVariable,
    state_target: QuaVariable,
    state_st,
    reset_type: Literal["thermal", "active"],
    cz_operation: str = "cz_unipolar",
    simulate: bool = False,
):
    """Play one packed circuit's gates, then perform the circuit-boundary helper.

    ``circuit_start`` / ``circuit_stop`` are absolute indices of gate fields
    (headers never enter the switch). When they are equal the gate loop is
    skipped and readout still runs.
    """
    gate_index = declare(int)
    with for_(gate_index, circuit_start, gate_index < circuit_stop, gate_index + 1):
        play_gate(sequence[gate_index], qubit_pair, cz_operation)
    readout_save_and_reset(
        qubit_pair,
        state,
        state_control,
        state_target,
        state_st,
        reset_type,
        simulate,
    )


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
            circuits_as_ints: Gate-only circuits (opcodes 0–37), depth-major then
                sequence within depth. Do not append a readout marker.
            machine: The QUAM machine configuration.
            qubit_pairs: List of qubit pairs to benchmark.
        """

        self.u = unit(coerce_to_integer=True)
        self.node = node
        self.num_pairs = num_pairs
        self.machine = machine
        self.qubit_pairs = qubit_pairs

        preflight_cz_align_elements(qubit_pairs, self.node.parameters.operation)
        ensure_xy_zero_pulse(qubit_pairs)

        circuit_depths = list(self.node.namespace["circuit_depths"])
        num_circuits_per_depth = self.node.parameters.num_circuits_per_depth
        max_chunk_ints = self.node.parameters.max_chunk_ints

        self.circuits_as_ints = validate_circuit_list(
            circuits_as_ints,
            circuit_depths=circuit_depths,
            num_circuits_per_depth=num_circuits_per_depth,
        )
        memory_stats = compute_rb_circuit_memory_stats(
            self.circuits_as_ints, circuit_depths, num_circuits_per_depth
        )

        self.declared_size = None
        self.chunks_per_depth = None
        if self.node.parameters.use_input_stream:
            self.chunks_per_depth, self.declared_size = build_single_depth_chunks(
                circuits_as_ints=self.circuits_as_ints,
                circuit_depths=circuit_depths,
                num_circuits_per_depth=num_circuits_per_depth,
                max_chunk_ints=max_chunk_ints,
                per_depth=memory_stats["per_depth"],
            )
        else:
            self.packed_sequence = pack_circuits(self.circuits_as_ints, validate=False)
            validate_without_inputstream_path(memory_stats, max_chunk_ints)

        if self.node.parameters.verbose_memory_log:
            log_rb_circuit_memory_stats(
                memory_stats,
                use_input_stream=self.node.parameters.use_input_stream,
                max_chunk_ints=max_chunk_ints,
                declared_size=self.declared_size,
                chunks_per_depth=self.chunks_per_depth if self.node.parameters.use_input_stream else None,
                circuit_depths=circuit_depths if self.node.parameters.use_input_stream else None,
                verbose=True,
                log_callable=self.node.log,
            )

    def _get_qua_program_with_input_stream(self):
        # Flatten chunks_per_depth into a single ordered list of packed packets.
        # Order: depth-major, then sub_chunk_index within depth — same order
        # the QUA program consumes them and the host pushes them in.
        flat_sub_chunks = [sc for sub_chunks in self.chunks_per_depth for sc in sub_chunks]
        n_sub_chunks = len(flat_sub_chunks)
        num_shots = self.node.parameters.num_shots
        num_depths = len(self.node.namespace["circuit_depths"])
        num_circuits_per_depth = self.node.parameters.num_circuits_per_depth

        with program() as rb:

            n = declare(int)
            n_done = declare(int)
            n_st = declare_stream()
            j = declare(int)
            c = declare(int)
            cursor = declare(int)
            n_in_chunk = declare(int)
            length = declare(int)
            circuit_start = declare(int)
            circuit_stop = declare(int)

            sequence = declare_input_stream("client", stream_id="sequence", dtype=int, size=self.declared_size)

            state_st = [declare_stream() for _ in range(self.num_pairs)]

            for multiplexed_qubit_pairs in self.qubit_pairs.batch():
                state_control = declare(int)
                state_target = declare(int)
                state = declare(int)

                for qp in multiplexed_qubit_pairs.values():
                    self.node.machine.initialize_qpu(target=qp.qubit_control)
                    self.node.machine.initialize_qpu(target=qp.qubit_target)
                align()

                for qp in multiplexed_qubit_pairs.values():
                    qp.qubit_control.reset(
                        reset_type=self.node.parameters.reset_type,
                        simulate=self.node.parameters.simulate,
                    )
                    qp.qubit_target.reset(
                        reset_type=self.node.parameters.reset_type,
                        simulate=self.node.parameters.simulate,
                    )
                align()

                assign(n_done, 0)
                # multiplex → chunk → circuit → shot → gate. One advance per
                # packed packet; shots replay the same circuit range on the OPX.
                with for_(j, 0, j < n_sub_chunks, j + 1):
                    advance_input_stream(sequence)
                    assign(n_in_chunk, sequence[0])
                    assign(cursor, 1)
                    with for_(c, 0, c < n_in_chunk, c + 1):
                        assign(length, sequence[cursor])
                        assign(circuit_start, cursor + 1)
                        assign(circuit_stop, circuit_start + length)
                        assign(cursor, circuit_stop)
                        with for_(n, 0, n < num_shots, n + 1):
                            play_sequence(
                                sequence,
                                circuit_start,
                                circuit_stop,
                                multiplexed_qubit_pairs,
                                state,
                                state_control,
                                state_target,
                                state_st,
                                self.node.parameters.reset_type,
                                self.node.parameters.operation,
                                self.node.parameters.simulate,
                            )
                            assign(n_done, n_done + 1)
                            save(n_done, n_st)

            with stream_processing():
                n_st.save("n")
                for k in range(len(self.qubit_pairs)):
                    state_st[k].buffer(num_shots).buffer(num_circuits_per_depth).buffer(num_depths).save(
                        f"state{k + 1}"
                    )
        return rb

    def _padded_chunks(self) -> list[list[int]]:
        """Return the flat depth-major list of packed sub-chunks, each padded to
        ``self.declared_size`` ints with :data:`INPUT_STREAM_PAD_VALUE`.

        Pad words sit after the last parsed circuit and are never passed to
        ``play_gate``. ``push_all_chunks`` pushes this list once per multiplex
        batch (one host push per sub-chunk; shots replay each circuit on the OPX).
        """
        return flatten_padded_chunks(
            self.chunks_per_depth,
            self.declared_size,
            pad_value=INPUT_STREAM_PAD_VALUE,
        )

    def push_all_chunks(self, job) -> None:
        """Push input-stream chunks in the order the QUA program consumes them.

        Order mirrors ``_get_qua_program_with_input_stream``:
        ``for batch: for sub_chunk: advance; for circuit: for shot: replay``.
        Each chunk is pushed once per multiplex batch; shots replay on the OPX.

        Args:
            job: The running ``QmJob`` returned by ``qm.execute(...)``.
        """
        if not self.node.parameters.use_input_stream:
            raise RuntimeError(
                "push_all_chunks called but use_input_stream is False; "
                "the QUA program does not declare an input stream."
            )

        padded = self._padded_chunks()
        for _ in self.qubit_pairs.batch():
            for chunk in padded:
                job.push_to_input_stream("sequence", chunk)

    def _get_qua_program_without_input_stream(self):
        packed = self.packed_sequence
        n_circuits = packed[0]
        num_shots = self.node.parameters.num_shots
        num_depths = len(self.node.namespace["circuit_depths"])
        num_circuits_per_depth = self.node.parameters.num_circuits_per_depth

        with program() as rb:

            n = declare(int)
            n_st = declare_stream()
            c = declare(int)
            cursor = declare(int)
            length = declare(int)
            circuit_start = declare(int)
            circuit_stop = declare(int)

            job_sequence_qua = declare(int, value=packed)

            state_st = [declare_stream() for _ in range(self.num_pairs)]

            for multiplexed_qubit_pairs in self.qubit_pairs.batch():
                state_control = declare(int)
                state_target = declare(int)
                state = declare(int)

                for qp in multiplexed_qubit_pairs.values():
                    self.node.machine.initialize_qpu(target=qp.qubit_control)
                    self.node.machine.initialize_qpu(target=qp.qubit_target)
                align()

                # multiplex → shot → circuit (depth-major) → gate
                with for_(n, 0, n < num_shots, n + 1):
                    for qp in multiplexed_qubit_pairs.values():
                        qp.qubit_control.reset(
                            self.node.parameters.reset_type,
                            self.node.parameters.simulate,
                        )
                        qp.qubit_target.reset(
                            self.node.parameters.reset_type,
                            self.node.parameters.simulate,
                        )
                    align()

                    assign(cursor, 1)
                    with for_(c, 0, c < n_circuits, c + 1):
                        assign(length, job_sequence_qua[cursor])
                        assign(circuit_start, cursor + 1)
                        assign(circuit_stop, circuit_start + length)
                        play_sequence(
                            job_sequence_qua,
                            circuit_start,
                            circuit_stop,
                            multiplexed_qubit_pairs,
                            state,
                            state_control,
                            state_target,
                            state_st,
                            self.node.parameters.reset_type,
                            self.node.parameters.operation,
                            self.node.parameters.simulate,
                        )
                        assign(cursor, circuit_stop)

                    save(n, n_st)

            with stream_processing():
                n_st.save("n")
                for i in range(len(self.qubit_pairs)):
                    state_st[i].buffer(num_circuits_per_depth).buffer(num_depths).buffer(num_shots).save(
                        f"state{i + 1}"
                    )
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
