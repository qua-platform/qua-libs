"""Host-side tests for two-qubit RB packing, chunks, cache, and data identity.

No CS_3. No customer support scripts. These checks cover circuit boundaries and
the fetch/buffer/normalize contract after the gate-only executor change.
"""

from __future__ import annotations

import inspect
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate, RYGate, SXGate, XGate, YGate
from qiskit.quantum_info import Operator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import calibration_utils.two_qubit_rb.qua_utils as qua_utils  # noqa: E402
from calibration_utils.two_qubit_rb import fit_utils  # noqa: E402
from calibration_utils.two_qubit_rb.analysis import (  # noqa: E402
    RB_EXECUTION_FORMAT_VERSION,
    process_raw_dataset,
    stamp_execution_format,
)
from calibration_utils.two_qubit_rb.circuit_utils import (  # noqa: E402
    circuit_to_layer_ints,
    get_gate_name,
    summarize_transpiled_depth,
)
from calibration_utils.two_qubit_rb.fit_utils import REFERENCE_BASIS_SKIPPED_ISSUE, fit_irb_pair  # noqa: E402
from calibration_utils.two_qubit_rb.packing import (  # noqa: E402
    INPUT_STREAM_PAD_VALUE,
    MIN_CHUNK_CAPACITY,
    RB_ENCODING_VERSION,
    RB_ENCODING_VERSIONS,
    READOUT_MARKER_OPCODE,
    CircuitPackingError,
    build_packed_depth_chunks,
    circuit_gate_ranges,
    flatten_padded_chunks,
    pack_circuits,
    packed_packet_words,
    pad_packet,
    science_readout_count,
    unpack_circuits,
    validate_chunk_capacity,
    validate_gate_opcode,
)
from calibration_utils.two_qubit_rb.parameters import (  # noqa: E402
    CANONICAL_ANALYSIS_DIMS,
    DECLARED_RAW_DIMS,
    STREAMED_RAW_DIMS,
    build_sweep_axes,
    rb_progress_total,
)
from calibration_utils.two_qubit_rb.rb_cache import (  # noqa: E402
    DEFAULT_RB_BASIS_GATES,
    RB_BASIS_GATES,
    cache_key,
    legacy_cache_key,
    save,
    try_load,
    try_load_legacy_statistics,
    try_load_statistics,
)
from calibration_utils.two_qubit_rb.rb_utils import StandardRB  # noqa: E402
from calibration_utils.two_qubit_rb.qua_utils import play_gate, play_sequence  # noqa: E402

# Sanitized gate counts matching the support reproduction's circuit lengths
# (readout markers stripped). Do not import the customer script.
SUPPORT_LIKE_GATE_COUNTS = [0, 0, 113, 102, 195, 185, 300, 288, 391, 393]


def _gates(n: int, *, seed: int = 0) -> list[int]:
    return [((seed + i) % (READOUT_MARKER_OPCODE)) for i in range(n)]


def _circuits_from_counts(counts: list[int]) -> list[list[int]]:
    return [_gates(n, seed=i) for i, n in enumerate(counts)]


def _node(*, use_input_stream: bool, single_qubit_basis: str = "XY") -> SimpleNamespace:
    return SimpleNamespace(
        parameters=SimpleNamespace(use_input_stream=use_input_stream, single_qubit_basis=single_qubit_basis)
    )


class _NamedPairs:
    def __init__(self, names: list[str]):
        self._names = names

    def get_names(self) -> list[str]:
        return list(self._names)


def _event(pair: int, depth: int, sequence: int, shot: int) -> int:
    """Distinct science label so aggregate means cannot hide an axis swap."""
    return pair * 1_000_000 + depth * 10_000 + sequence * 100 + shot


def _pair_batches(n_pairs: int, batch_size: int) -> list[list[int]]:
    return [list(range(i, min(i + batch_size, n_pairs))) for i in range(0, n_pairs, batch_size)]


def _reshape_declared(events: list[int], n_shots: int, n_depths: int, n_seq: int) -> np.ndarray:
    """Mirror ``.buffer(sequence).buffer(circuit_depth).buffer(shots)``."""
    arr = np.asarray(events, dtype=int)
    assert arr.size == n_shots * n_depths * n_seq
    return arr.reshape(n_shots, n_depths, n_seq)


def _reshape_streamed(events: list[int], n_shots: int, n_depths: int, n_seq: int) -> np.ndarray:
    """Mirror ``.buffer(num_shots).buffer(num_circuits_per_depth).buffer(num_depths)``."""
    arr = np.asarray(events, dtype=int)
    assert arr.size == n_shots * n_depths * n_seq
    return arr.reshape(n_depths, n_seq, n_shots)


def _simulate_declared(
    n_pairs: int,
    batch_size: int,
    n_depths: int,
    n_seq: int,
    n_shots: int,
) -> list[list[int]]:
    streams = [[] for _ in range(n_pairs)]
    for batch in _pair_batches(n_pairs, batch_size):
        for shot in range(n_shots):
            for depth in range(n_depths):
                for seq in range(n_seq):
                    for pair in batch:
                        streams[pair].append(_event(pair, depth, seq, shot))
    return streams


def _simulate_streamed(
    n_pairs: int,
    batch_size: int,
    n_shots: int,
    chunks_per_depth: list[list[list[int]]],
) -> list[list[int]]:
    """Host-side replica of multiplex → chunk → circuit → shot (shots replay a circuit)."""
    streams = [[] for _ in range(n_pairs)]
    flat_chunks: list[tuple[int, list[int]]] = []
    for depth, sub_chunks in enumerate(chunks_per_depth):
        for circuit_seqs in sub_chunks:
            flat_chunks.append((depth, list(circuit_seqs)))
    for batch in _pair_batches(n_pairs, batch_size):
        for depth, circuit_seqs in flat_chunks:
            for seq in circuit_seqs:
                for shot in range(n_shots):
                    for pair in batch:
                        streams[pair].append(_event(pair, depth, seq, shot))
    return streams


def _dataset_from_raw(
    stacked: np.ndarray,
    pair_names: list[str],
    depths: list[int],
    n_seq: int,
    n_shots: int,
    *,
    use_input_stream: bool,
) -> xr.Dataset:
    axes = build_sweep_axes(
        _NamedPairs(pair_names),
        n_shots,
        depths,
        n_seq,
        use_input_stream=use_input_stream,
    )
    dim_order = list(axes)
    expected_dims = list(STREAMED_RAW_DIMS if use_input_stream else DECLARED_RAW_DIMS)
    assert dim_order == expected_dims
    coords = {name: np.asarray(da.values) for name, da in axes.items()}
    ds = xr.Dataset({"state": (dim_order, stacked)}, coords=coords)
    return stamp_execution_format(ds, _node(use_input_stream=use_input_stream))


# ------------------------------------------------------------------- round trip


def test_pack_unpack_round_trip_empty_mixed_and_consecutive_empty():
    circuits = [[], [36], [36, 14, 0], [], [], [1, 17, 36]]
    packet = pack_circuits(circuits)
    assert packet == [6, 0, 1, 36, 3, 36, 14, 0, 0, 0, 3, 1, 17, 36]
    assert unpack_circuits(packet) == circuits
    assert science_readout_count(circuits) == len(circuits)


def test_pack_example_from_encoding_docs():
    circuits = [[], [36, 14, 0], []]
    packet = pack_circuits(circuits)
    assert packet == [3, 0, 3, 36, 14, 0, 0]
    assert circuit_gate_ranges(packet) == [(2, 2), (3, 6), (7, 7)]
    header_indices = {0, 1, 2, 6}
    for start, stop in circuit_gate_ranges(packet):
        assert header_indices.isdisjoint(range(start, stop))
    assert packet[3:6] == [36, 14, 0]


def test_length_header_may_equal_38_without_being_readout():
    circuit = _gates(READOUT_MARKER_OPCODE, seed=7)
    packet = pack_circuits([circuit])
    assert packet[0] == 1
    assert packet[1] == READOUT_MARKER_OPCODE
    assert unpack_circuits(packet) == [circuit]


# ------------------------------------------------------- support-like lengths


def test_support_like_gate_counts_round_trip_and_science_readouts():
    circuits = _circuits_from_counts(SUPPORT_LIKE_GATE_COUNTS)
    depths = [0, 8]
    n_per = 5
    packet = pack_circuits(circuits)
    restored = unpack_circuits(packet)
    assert restored == circuits
    assert [len(c) for c in restored] == SUPPORT_LIKE_GATE_COUNTS
    assert restored[0] == [] and restored[1] == []
    assert science_readout_count(circuits) == 10

    chunks, declared = build_packed_depth_chunks(circuits, depths, n_per, max_chunk_ints=15000)
    assert len(chunks) == 2
    recovered = [circ for depth_chunks in chunks for packet in depth_chunks for circ in unpack_circuits(packet)]
    assert recovered == circuits
    assert science_readout_count(recovered) == 10
    padded = flatten_padded_chunks(chunks, declared)
    for packet in padded:
        unpacked = unpack_circuits(packet, allow_padding=True)
        assert science_readout_count(unpacked) == packet[0]
        last_stop = circuit_gate_ranges(packet)[-1][1]
        assert all(w == INPUT_STREAM_PAD_VALUE for w in packet[last_stop:])


# ------------------------------------------------------------------- chunk edges


def test_chunk_exact_fit_then_one_word_too_large_starts_new_chunk():
    circuits = [_gates(2, seed=0), _gates(2, seed=1), _gates(2, seed=2)]
    two_fit = packed_packet_words(circuits[:2])
    three = packed_packet_words(circuits)
    assert three == two_fit + 1 + len(circuits[2])
    chunks, _ = build_packed_depth_chunks(circuits, [4], 3, max_chunk_ints=two_fit)
    assert len(chunks) == 1
    assert len(chunks[0]) == 2
    assert unpack_circuits(chunks[0][0]) == circuits[:2]
    assert unpack_circuits(chunks[0][1]) == [circuits[2]]
    recovered = [c for packet in chunks[0] for c in unpack_circuits(packet)]
    assert recovered == circuits
    assert science_readout_count(recovered) == 3


def test_several_unequal_chunks_within_depth_never_split_a_circuit():
    circuits = [_gates(1, seed=0), _gates(8, seed=1), _gates(1, seed=2), _gates(8, seed=3)]
    # Capacity fits one 8-gate circuit (1+1+8=10) plus maybe a 1-gate ( +2), not two 8-gates.
    capacity = packed_packet_words([_gates(8), _gates(1)])
    chunks, declared = build_packed_depth_chunks(circuits, [2], 4, max_chunk_ints=capacity)
    sub = chunks[0]
    assert len(sub) >= 2
    sizes = [len(p) for p in sub]
    assert len(set(sizes)) >= 1
    recovered = [c for p in sub for c in unpack_circuits(p)]
    assert recovered == circuits
    for packet in sub:
        for circ in unpack_circuits(packet):
            assert circ in circuits
    padded = flatten_padded_chunks(chunks, declared)
    for packet in padded:
        ranges = circuit_gate_ranges(packet)
        last_stop = ranges[-1][1]
        pad_words = packet[last_stop:]
        assert all(w == INPUT_STREAM_PAD_VALUE for w in pad_words)
        unpack_circuits(packet, allow_padding=True)


def test_all_empty_chunk_and_consecutive_empty_circuits():
    circuits = [[], [], []]
    chunks, declared = build_packed_depth_chunks(circuits, [0], 3, max_chunk_ints=MIN_CHUNK_CAPACITY)
    assert all(unpack_circuits(p) == [[]] for depth in chunks for p in depth)
    recovered = [c for depth in chunks for p in depth for c in unpack_circuits(p)]
    assert recovered == circuits
    assert science_readout_count(recovered) == 3
    padded = flatten_padded_chunks(chunks, declared)
    for packet in padded:
        unpack_circuits(packet, allow_padding=True)
        assert packet[0] == 1
        assert packet[1] == 0


def test_chunks_do_not_cross_depth_boundaries():
    a = [_gates(3, seed=0), _gates(3, seed=1)]
    b = [_gates(4, seed=2), _gates(4, seed=3)]
    circuits = a + b
    capacity = packed_packet_words([_gates(4)])
    chunks, _ = build_packed_depth_chunks(circuits, [1, 2], 2, max_chunk_ints=capacity)
    assert unpack_circuits(chunks[0][0]) == [a[0]]
    assert unpack_circuits(chunks[0][1]) == [a[1]]
    assert unpack_circuits(chunks[1][0]) == [b[0]]
    assert unpack_circuits(chunks[1][1]) == [b[1]]


def test_oversized_circuit_names_depth_sequence_and_capacity():
    circuits = [_gates(5, seed=0), _gates(20, seed=1)]
    capacity = packed_packet_words([_gates(5)])
    with pytest.raises(CircuitPackingError, match="sequence_index=1") as exc:
        build_packed_depth_chunks(circuits, [7], 2, max_chunk_ints=capacity)
    msg = str(exc.value)
    assert "depth=7" in msg
    assert "required_words=" in msg
    assert f"available_capacity={capacity}" in msg
    assert "cannot split" in msg.lower() or "cannot split" in msg or "Streaming cannot split" in msg


# -------------------------------------------------------------- validation


def test_validate_rejects_opcode_38_as_a_gate():
    with pytest.raises(CircuitPackingError, match="not a gate"):
        validate_gate_opcode(READOUT_MARKER_OPCODE)
    with pytest.raises(CircuitPackingError, match="not a gate"):
        pack_circuits([[0, READOUT_MARKER_OPCODE]])


def test_validate_rejects_negative_float_and_unknown_opcodes():
    with pytest.raises(CircuitPackingError, match="outside"):
        validate_gate_opcode(-1)
    with pytest.raises(CircuitPackingError, match="integer"):
        validate_gate_opcode(1.5)
    with pytest.raises(CircuitPackingError, match="outside"):
        pack_circuits([[99]])


def test_validate_rejects_malformed_packet_and_tiny_capacity():
    with pytest.raises(CircuitPackingError, match="overrun"):
        unpack_circuits([1, 3, 0])
    with pytest.raises(CircuitPackingError, match="unused word"):
        unpack_circuits([1, 0, 7])
    with pytest.raises(CircuitPackingError, match="expected pad_value"):
        unpack_circuits([1, 0, 7], allow_padding=True)
    with pytest.raises(CircuitPackingError, match="cannot hold"):
        validate_chunk_capacity(1)
    padded = pad_packet(pack_circuits([[36]]), 6)
    assert padded[-3:] == [INPUT_STREAM_PAD_VALUE] * 3
    assert unpack_circuits(padded, allow_padding=True) == [[36]]
    start, stop = circuit_gate_ranges(padded)[0]
    assert padded[start:stop] == [36]


# -------------------------------------------------------------- data identity


def test_declared_and_streamed_synthetic_identity_two_chunks_partial_batch():
    pair_names = ["q0-q1", "q2-q3", "q4-q5"]
    n_pairs = 3
    batch_size = 2
    assert _pair_batches(n_pairs, batch_size)[-1] == [2]

    depths = [0, 4]
    n_seq = 3
    n_shots = 2
    # Force two chunks in each depth: first two circuits fit, third does not.
    per_depth_circuits = [_gates(2, seed=d * 10 + s) for d in depths for s in range(n_seq)]
    two_fit = packed_packet_words(per_depth_circuits[:2])
    chunks, declared = build_packed_depth_chunks(per_depth_circuits, depths, n_seq, max_chunk_ints=two_fit)
    assert len(chunks[0]) == 2
    seq_chunks = []
    for depth_packets in chunks:
        seq_at = 0
        depth_seq = []
        for packet in depth_packets:
            n_here = packet[0]
            depth_seq.append(list(range(seq_at, seq_at + n_here)))
            seq_at += n_here
        assert seq_at == n_seq
        seq_chunks.append(depth_seq)
    assert seq_chunks[0] == [[0, 1], [2]]

    padded = flatten_padded_chunks(chunks, declared)
    n_batches = len(_pair_batches(n_pairs, batch_size))
    host_pushes = n_batches * len(padded)
    assert host_pushes == 2 * len(padded)
    for packet in padded:
        unpack_circuits(packet, allow_padding=True)

    streamed_events = _simulate_streamed(n_pairs, batch_size, n_shots, seq_chunks)
    declared_events = _simulate_declared(n_pairs, batch_size, len(depths), n_seq, n_shots)

    streamed_stack = np.stack([_reshape_streamed(ev, n_shots, len(depths), n_seq) for ev in streamed_events])
    declared_stack = np.stack([_reshape_declared(ev, n_shots, len(depths), n_seq) for ev in declared_events])

    ds_stream = _dataset_from_raw(streamed_stack, pair_names, depths, n_seq, n_shots, use_input_stream=True)
    ds_decl = _dataset_from_raw(declared_stack, pair_names, depths, n_seq, n_shots, use_input_stream=False)
    assert tuple(ds_stream["state"].dims) == STREAMED_RAW_DIMS
    assert tuple(ds_decl["state"].dims) == DECLARED_RAW_DIMS
    assert ds_stream.attrs["rb_execution_format_version"] == RB_EXECUTION_FORMAT_VERSION
    assert ds_stream.attrs["rb_encoding_version"] == RB_ENCODING_VERSION
    assert ds_stream.attrs["rb_raw_acquisition_order"] == list(STREAMED_RAW_DIMS)
    assert ds_decl.attrs["rb_raw_acquisition_order"] == list(DECLARED_RAW_DIMS)

    proc_s = process_raw_dataset(ds_stream, _node(use_input_stream=True))
    proc_d = process_raw_dataset(ds_decl, _node(use_input_stream=False))
    assert tuple(proc_s["state"].dims) == CANONICAL_ANALYSIS_DIMS
    assert tuple(proc_d["state"].dims) == CANONICAL_ANALYSIS_DIMS
    xr.testing.assert_equal(proc_s["state"], proc_d["state"])

    for p, name in enumerate(pair_names):
        for d_i, depth in enumerate(depths):
            for seq in range(n_seq):
                for shot in range(n_shots):
                    expected = _event(p, d_i, seq, shot)
                    got = int(
                        proc_s["state"].sel(qubit_pair=name, circuit_depth=depth, sequence=seq, shots=shot).item()
                    )
                    assert got == expected

    assert rb_progress_total(n_shots, n_seq, len(depths), use_input_stream=True) == n_shots * n_seq * len(depths)
    assert rb_progress_total(n_shots, n_seq, len(depths), use_input_stream=False) == n_shots


# ---------------------------------------------------------------------- cache


def test_cache_v2_hit_old_format_miss_and_statistics_only(tmp_path: Path):
    seed, depths, n_per = 0, [0, 4], 2
    circuits = [[], [36, 14, 0], [1], [17, 36]]
    stats = {
        "average_gates_per_clifford": 1.5,
        "avg_1q_per_clifford": 1.0,
        "avg_cz_per_clifford": 0.5,
    }
    key_v2 = cache_key(seed, depths, n_per)
    save(tmp_path, key_v2, {"circuits_as_ints": circuits, **stats})
    loaded = try_load(tmp_path, key_v2)
    assert loaded is not None
    assert loaded["encoding_version"] == RB_ENCODING_VERSION
    assert loaded["circuits_as_ints"] == circuits
    # Repack after load with a different capacity (cache stores lists, not packets).
    chunks_a, _ = build_packed_depth_chunks(loaded["circuits_as_ints"], depths, n_per, 50)
    chunks_b, _ = build_packed_depth_chunks(loaded["circuits_as_ints"], depths, n_per, 8)
    recovered_a = [c for d in chunks_a for p in d for c in unpack_circuits(p)]
    recovered_b = [c for d in chunks_b for p in d for c in unpack_circuits(p)]
    assert recovered_a == recovered_b == circuits

    legacy_key = legacy_cache_key(seed, depths, n_per)
    assert legacy_key != key_v2
    legacy_payload = {
        "circuits_as_ints": [c + [READOUT_MARKER_OPCODE] for c in circuits],
        **stats,
    }
    (tmp_path / f"{legacy_key}.json").write_text(json.dumps(legacy_payload), encoding="utf-8")
    assert try_load(tmp_path, legacy_key) is None
    assert (tmp_path / f"{legacy_key}.json").exists()
    legacy_stats = try_load_legacy_statistics(tmp_path, seed, depths, n_per)
    assert legacy_stats is not None
    assert "circuits_as_ints" not in legacy_stats
    assert legacy_stats["average_gates_per_clifford"] == 1.5
    assert try_load_statistics(tmp_path, key_v2)["avg_cz_per_clifford"] == 0.5

    bad_key = cache_key(1, depths, n_per)
    (tmp_path / f"{bad_key}.json").write_text(
        json.dumps({"encoding_version": RB_ENCODING_VERSION, "circuits_as_ints": [[READOUT_MARKER_OPCODE]]}),
        encoding="utf-8",
    )
    assert try_load(tmp_path, bad_key) is None
    (tmp_path / f"{bad_key}.json").write_text("{not-json", encoding="utf-8")
    assert try_load(tmp_path, bad_key) is None


# ------------------------------------------------ analog XY encoding (task-m)


_ALLOWED_TRANSPILE_OPS = frozenset({"sx", "x", "ry", "y", "cz", "barrier"})
_SLOT_GATES = {
    "XY": (SXGate(), XGate(), RYGate(np.pi / 2), YGate(), RYGate(3.0 * np.pi / 2), None),
    "XZ": (SXGate(), XGate(), RZGate(np.pi / 2), RZGate(np.pi), RZGate(3.0 * np.pi / 2), None),
}


def _apply_opcode(qc: QuantumCircuit, opcode: int, basis: str) -> None:
    """Replay one encoded layer in *basis* (idle slot 5 is a no-op); CZ is 36, idle_2q is 37."""
    if opcode == 36:
        qc.cz(0, 1)
        return
    if opcode == 37:
        return
    if not 0 <= opcode <= 35:
        raise AssertionError(f"unexpected opcode {opcode}")
    g_c, g_t = divmod(opcode, 6)
    for qubit, slot in ((0, g_c), (1, g_t)):
        gate = _SLOT_GATES[basis][slot]
        if gate is not None:
            qc.append(gate, [qubit])


def test_get_gate_name_and_layer_ints_reject_rz_in_xy():
    with pytest.raises(ValueError, match="rz is not in the analog-XY"):
        get_gate_name(RZGate(np.pi / 2), "XY")
    with pytest.raises(ValueError, match="rz is not in the analog-XY"):
        get_gate_name(SimpleNamespace(name="rz", params=[np.pi]), "XY")

    assert get_gate_name(SXGate(), "XY") == "sx"
    assert get_gate_name(XGate(), "XY") == "x"
    assert get_gate_name(YGate(), "XY") == "y"
    assert get_gate_name(RYGate(np.pi / 2), "XY") == "ry(pi/2)"
    assert get_gate_name(RYGate(np.pi), "XY") == "y"
    assert get_gate_name(RYGate(-np.pi), "XY") == "y"
    assert get_gate_name(RYGate(3.0 * np.pi / 2), "XY") == "ry(3pi/2)"
    assert get_gate_name(RYGate(-np.pi / 2), "XY") == "ry(3pi/2)"
    with pytest.raises(ValueError, match="Unsupported angle"):
        get_gate_name(RYGate(0.3), "XY")

    rz_circuit = QuantumCircuit(2)
    rz_circuit.rz(np.pi / 2, 0)
    rz_circuit.sx(1)
    with pytest.raises(ValueError, match="rz is not in the analog-XY"):
        circuit_to_layer_ints(rz_circuit, "XY")

    mixed = QuantumCircuit(2)
    mixed.sx(0)
    mixed.ry(np.pi / 2, 1)
    assert circuit_to_layer_ints(mixed, "XY") == [2]


def test_rz_is_accepted_in_xz_and_rejected_in_xy():
    assert get_gate_name(RZGate(np.pi / 2), "XZ") == "rz(pi/2)"
    assert get_gate_name(RZGate(-np.pi), "XZ") == "rz(pi)"
    assert get_gate_name(RZGate(-np.pi / 2), "XZ") == "rz(3pi/2)"
    with pytest.raises(ValueError, match="Unsupported angle"):
        get_gate_name(RZGate(0.3), "XZ")
    with pytest.raises(ValueError, match="not in the XZ RB basis"):
        get_gate_name(YGate(), "XZ")
    with pytest.raises(ValueError, match="not in the XZ RB basis"):
        get_gate_name(RYGate(np.pi / 2), "XZ")

    xz_circuit = QuantumCircuit(2)
    xz_circuit.rz(np.pi / 2, 0)
    xz_circuit.sx(1)
    assert circuit_to_layer_ints(xz_circuit, "XZ") == [2 * 6 + 0]
    with pytest.raises(ValueError, match="rz is not in the analog-XY"):
        circuit_to_layer_ints(xz_circuit, "XY")


def test_seed_stable_transpile_contains_only_analog_xy_ops():
    depths = [1, 2, 4]
    n_per = 3
    seed = 7
    first = StandardRB(depths, n_per, seed=seed, show_progress=False)
    second = StandardRB(depths, n_per, seed=seed, show_progress=False)
    assert first.basis_gates == ["cz", "sx", "x", "ry", "y"]

    for depth in depths:
        assert len(first.transpiled_circuits[depth]) == n_per
        for qc_a, qc_b in zip(first.transpiled_circuits[depth], second.transpiled_circuits[depth]):
            names = {instr.operation.name for instr in qc_a}
            assert names <= _ALLOWED_TRANSPILE_OPS
            assert "rz" not in names
            encoded_a = circuit_to_layer_ints(qc_a, "XY")
            encoded_b = circuit_to_layer_ints(qc_b, "XY")
            assert encoded_a == encoded_b
            assert all(0 <= op <= 37 for op in encoded_a)
            assert 38 not in encoded_a


@pytest.mark.parametrize("basis", ["XY", "XZ"])
def test_encoding_round_trip_recovers_identity_up_to_phase_per_basis(basis):
    depths = [0, 1, 3]
    rb = StandardRB(depths, 2, basis_gates=list(RB_BASIS_GATES[basis]), seed=11, show_progress=False)
    identity = Operator(np.eye(4, dtype=complex))
    for depth in depths:
        for qc in rb.transpiled_circuits[depth]:
            names = {instr.operation.name for instr in qc}
            assert names <= set(RB_BASIS_GATES[basis]) | {"barrier"}
            replay = QuantumCircuit(2)
            for opcode in circuit_to_layer_ints(qc, basis):
                _apply_opcode(replay, opcode, basis)
            assert Operator(replay).equiv(identity)


def test_cache_v2_payload_is_acquisition_miss_and_stats_survive(tmp_path: Path):
    seed, depths, n_per = 0, [0, 4], 2
    circuits = [[], [36, 14, 0], [1], [17, 36]]
    stats = {"average_gates_per_clifford": 1.5, "avg_1q_per_clifford": 1.0, "avg_cz_per_clifford": 0.5}

    key_xy = cache_key(seed, depths, n_per)
    assert "ry" in DEFAULT_RB_BASIS_GATES and "rz" not in DEFAULT_RB_BASIS_GATES

    v2_payload = {"encoding_version": 2, "circuits_as_ints": circuits, **stats}
    (tmp_path / f"{key_xy}.json").write_text(json.dumps(v2_payload), encoding="utf-8")
    assert try_load(tmp_path, key_xy) is None
    v2_stats = try_load_statistics(tmp_path, key_xy)
    assert v2_stats is not None
    assert "circuits_as_ints" not in v2_stats
    assert v2_stats["average_gates_per_clifford"] == 1.5

    save(tmp_path, key_xy, {"circuits_as_ints": circuits, **stats})
    loaded = try_load(tmp_path, key_xy)
    assert loaded is not None
    assert loaded["encoding_version"] == RB_ENCODING_VERSION == 3
    assert loaded["circuits_as_ints"] == circuits


def test_cache_keys_and_payloads_split_by_basis(tmp_path: Path):
    seed, depths, n_per = 0, [0, 4], 2
    key_xy = cache_key(seed, depths, n_per, single_qubit_basis="XY")
    key_xz = cache_key(seed, depths, n_per, single_qubit_basis="XZ")
    assert key_xy == cache_key(seed, depths, n_per)
    assert key_xz != key_xy
    assert cache_key(seed, depths, n_per, single_qubit_basis="XZ", target_gate="cz") != key_xz
    assert RB_ENCODING_VERSIONS == {"XY": 3, "XZ": 4}

    circuits = [[], [12, 0]]
    save(tmp_path, key_xz, {"circuits_as_ints": circuits}, single_qubit_basis="XZ")
    loaded = try_load(tmp_path, key_xz, single_qubit_basis="XZ")
    assert loaded is not None
    assert loaded["single_qubit_basis"] == "XZ"
    assert loaded["encoding_version"] == 4
    assert loaded["circuits_as_ints"] == circuits
    assert try_load(tmp_path, key_xz, single_qubit_basis="XY") is None

    xz_payload_text = (tmp_path / f"{key_xz}.json").read_text(encoding="utf-8")
    (tmp_path / f"{key_xy}.json").write_text(xz_payload_text, encoding="utf-8")
    assert try_load(tmp_path, key_xy, single_qubit_basis="XY") is None


# --------------------------------------- CZ XY occupy (task-r)


class _Trace:
    """Records each play / frame / wait / apply call with the switch case that issued it."""

    def __init__(self):
        self.case = None
        self.calls: list[tuple] = []

    def case_(self, value):
        trace = self

        class _Case:
            def __enter__(self):
                trace.case = value

            def __exit__(self, *exc):
                trace.case = None
                return False

        return _Case()

    def ops(self, case) -> list[tuple]:
        return [call[1:] for call in self.calls if call[0] == case]


class _Element:
    def __init__(self, name: str, trace: _Trace):
        self.name, self.trace = name, trace

    def play(self, pulse):
        self.trace.calls.append((self.trace.case, self.name, "play", pulse))

    def frame_rotation(self, angle):
        self.trace.calls.append((self.trace.case, self.name, "frame_rotation", angle))

    def wait(self, duration):
        self.trace.calls.append((self.trace.case, self.name, "wait", duration))


class _CZMacro:
    def __init__(self, trace: _Trace):
        self.trace = trace

    def apply(self, align_elements=True):
        self.trace.calls.append((self.trace.case, "cz", "apply", align_elements))


def _qubit(name: str, trace: _Trace) -> SimpleNamespace:
    return SimpleNamespace(xy=_Element(f"{name}.xy", trace), wait=_Element(name, trace).wait)


@pytest.fixture
def traced_pair(monkeypatch):
    trace = _Trace()
    monkeypatch.setattr(qua_utils, "switch_", lambda gate, unsafe: nullcontext())
    monkeypatch.setattr(qua_utils, "case_", trace.case_)
    pair = SimpleNamespace(
        qubit_control=_qubit("qA2", trace),
        qubit_target=_qubit("qA1", trace),
        macros={"cz_unipolar": _CZMacro(trace)},
    )
    return trace, {"qA2-qA1": pair}


@pytest.mark.parametrize("basis", ["XY", "XZ"])
def test_cz_case_plays_no_xy_pulse_and_occupies_both_xy(traced_pair, basis):
    trace, pairs = traced_pair
    qua_utils.play_gate(None, pairs, basis)
    assert sorted(trace.ops(36), key=str) == sorted(
        [("cz", "apply", False), ("qA2.xy", "wait", 4), ("qA1.xy", "wait", 4)], key=str
    )


@pytest.mark.parametrize("basis", ["XY", "XZ"])
def test_idle_cases_wait_on_elements_without_xy_ops(traced_pair, basis):
    trace, pairs = traced_pair
    qua_utils.play_gate(None, pairs, basis)
    for case in (35, 37):
        assert trace.ops(case) == [("qA2", "wait", 4), ("qA1", "wait", 4)]


@pytest.mark.parametrize(
    "opcode, basis, expected",
    [
        (2, "XY", [("qA2.xy", "play", "x90"), ("qA1.xy", "play", "y90")]),
        (2, "XZ", [("qA2.xy", "play", "x90"), ("qA1.xy", "frame_rotation", np.pi / 2)]),
        (4, "XY", [("qA2.xy", "play", "x90"), ("qA1.xy", "play", "-y90")]),
        (4, "XZ", [("qA2.xy", "play", "x90"), ("qA1.xy", "frame_rotation", 3 * np.pi / 2)]),
        (17, "XY", [("qA2.xy", "play", "y90")]),
        (17, "XZ", [("qA2.xy", "frame_rotation", np.pi / 2)]),
        (30, "XZ", [("qA1.xy", "play", "x90")]),
    ],
)
def test_single_qubit_slots_follow_basis(traced_pair, opcode, basis, expected):
    trace, pairs = traced_pair
    qua_utils.play_gate(None, pairs, basis)
    assert trace.ops(opcode) == expected


def test_play_gate_body_has_no_align_and_play_sequence_gate_loop_is_clean():
    impl = inspect.getsource(play_gate).split('"""', 2)[-1]
    assert "align(" not in impl

    seq_src = inspect.getsource(play_sequence)
    loop_start = seq_src.index("with for_(gate_index")
    loop_end = seq_src.index("readout_save_and_reset")
    gate_for = seq_src[loop_start:loop_end]
    assert "play_gate(" in gate_for
    assert "align(" not in gate_for


# ------------------------------------------------ single-qubit basis (XZ / XY)


@pytest.mark.parametrize("basis", ["XY", "XZ"])
def test_stamp_records_single_qubit_basis_and_encoding(basis):
    stamped = stamp_execution_format(xr.Dataset(), _node(use_input_stream=False, single_qubit_basis=basis))
    assert stamped.attrs["rb_single_qubit_basis"] == basis
    assert stamped.attrs["rb_encoding_version"] == RB_ENCODING_VERSIONS[basis]
    assert stamped.attrs["rb_basis_gates"] == list(RB_BASIS_GATES[basis])


def test_summary_counts_virtual_rz_as_non_physical_only_in_xz():
    qc = QuantumCircuit(2)
    qc.sx(0)
    qc.rz(np.pi / 2, 0)
    qc.cz(0, 1)
    qc.barrier()
    silent = lambda message: None  # noqa: E731
    xz = summarize_transpiled_depth(1, [qc], single_qubit_basis="XZ", log_callable=silent)
    assert xz["avg_transpiled_gates_per_clifford"] == 3.0
    assert xz["avg_physical_gates_per_clifford"] == 2.0
    assert xz["avg_1q_per_clifford"] == 1.0
    xy = summarize_transpiled_depth(1, [qc], single_qubit_basis="XY", log_callable=silent)
    assert xy["avg_physical_gates_per_clifford"] == 3.0


def _survival_dataset(seed: int = 0, shots: int = 2000, sequences: int = 4) -> xr.Dataset:
    depths = np.array([0, 2, 4, 8, 16, 32, 64])
    survival = 0.75 * 0.97**depths + 0.25
    rng = np.random.default_rng(seed)
    success = rng.random((shots, depths.size, sequences)) < survival[None, :, None]
    return xr.Dataset(
        {"state": (("shots", "circuit_depth", "sequence"), np.where(success, 0, 1))},
        coords={"shots": np.arange(shots), "circuit_depth": depths, "sequence": np.arange(sequences)},
    )


def _irb_node(fidelity: dict, single_qubit_basis: str):
    logs: list[str] = []
    macro = SimpleNamespace(fidelity=fidelity, extras={})
    node = SimpleNamespace(
        parameters=SimpleNamespace(operation="cz_unipolar", single_qubit_basis=single_qubit_basis),
        machine=SimpleNamespace(qubit_pairs={"qA2-qA1": SimpleNamespace(macros={"cz_unipolar": macro})}),
        log=logs.append,
    )
    return node, logs


def test_irb_skips_a_reference_in_the_other_basis():
    reference = {"StandardRB_alpha": 0.97, "StandardRB": 0.977, "StandardRB_single_qubit_basis": "XZ"}
    node, logs = _irb_node(reference, "XY")
    ds_fit = fit_irb_pair(_survival_dataset(), node, "qA2-qA1")
    assert np.isnan(float(ds_fit.fidelity))
    assert np.isnan(float(ds_fit.standard_rb_alpha))
    assert not bool(ds_fit.success)
    assert REFERENCE_BASIS_SKIPPED_ISSUE in str(ds_fit.fit_issues.values)
    assert len(logs) == 1 and "XZ" in logs[0] and "skipping" in logs[0]


def test_irb_uses_a_reference_in_the_same_basis_silently():
    reference = {"StandardRB_alpha": 0.97, "StandardRB": 0.977, "StandardRB_single_qubit_basis": "XY"}
    node, logs = _irb_node(reference, "XY")
    ds_fit = fit_irb_pair(_survival_dataset(), node, "qA2-qA1")
    assert np.isfinite(float(ds_fit.fidelity))
    assert float(ds_fit.standard_rb_alpha) == pytest.approx(0.97)
    assert logs == []


def test_irb_warns_but_uses_a_legacy_reference_without_basis():
    node, logs = _irb_node({"StandardRB_alpha": 0.97, "StandardRB": 0.977}, "XZ")
    ds_fit = fit_irb_pair(_survival_dataset(), node, "qA2-qA1")
    assert np.isfinite(float(ds_fit.fidelity))
    assert len(logs) == 1 and "legacy" in logs[0]


def test_irb_overlay_basis_from_saved_ds_fit_takes_precedence(monkeypatch):
    depths = np.array([0, 2, 4, 8, 16, 32, 64])
    saved = xr.Dataset(
        {
            "survival_probability": (("qubit_pair", "circuit_depth"), (0.75 * 0.97**depths + 0.25)[None, :]),
            "fitted_curve": (("qubit_pair", "circuit_depth"), (0.75 * 0.97**depths + 0.25)[None, :]),
            "fit_alpha": ("qubit_pair", [0.97]),
        },
        coords={"qubit_pair": ["qA2-qA1"], "circuit_depth": depths},
        attrs={"rb_single_qubit_basis": "XZ"},
    )
    monkeypatch.setattr(fit_utils, "load_saved_ds_fit", lambda load_id: saved)
    reference = {"StandardRB_alpha": 0.97, "StandardRB_load_id": 7, "StandardRB_single_qubit_basis": "XY"}
    node, logs = _irb_node(reference, "XY")
    ds_fit = fit_irb_pair(_survival_dataset(), node, "qA2-qA1")
    assert np.isnan(float(ds_fit.fidelity))
    assert len(logs) == 1 and "XZ" in logs[0] and "skipping" in logs[0]
