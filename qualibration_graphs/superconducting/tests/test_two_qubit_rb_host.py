"""Host-side tests for two-qubit RB packing, chunks, cache, and data identity.

No CS_3. No customer support scripts. These checks cover circuit boundaries and
the fetch/buffer/normalize contract after the gate-only executor change.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate, RYGate, SXGate, XGate, YGate
from qiskit.quantum_info import Operator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_utils.two_qubit_rb.analysis import (  # noqa: E402
    RB_EXECUTION_FORMAT_VERSION,
    process_raw_dataset,
    stamp_execution_format,
)
from calibration_utils.two_qubit_rb.circuit_utils import (  # noqa: E402
    circuit_to_layer_ints,
    get_gate_name,
)
from calibration_utils.two_qubit_rb.packing import (  # noqa: E402
    INPUT_STREAM_PAD_VALUE,
    MIN_CHUNK_CAPACITY,
    RB_ENCODING_VERSION,
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
    cache_key,
    save,
    try_load,
    try_load_legacy_statistics,
    try_load_statistics,
)
from calibration_utils.two_qubit_rb.rb_utils import StandardRB  # noqa: E402
from calibration_utils.two_qubit_rb.qua_utils import (  # noqa: E402
    ensure_xy_zero_pulse,
    play_gate,
    play_sequence,
)
from quam.components.pulses import SquarePulse  # noqa: E402

# Sanitized gate counts matching the support reproduction's circuit lengths
# (readout markers stripped). Do not import the customer script.
SUPPORT_LIKE_GATE_COUNTS = [0, 0, 113, 102, 195, 185, 300, 288, 391, 393]


def _gates(n: int, *, seed: int = 0) -> list[int]:
    return [((seed + i) % (READOUT_MARKER_OPCODE)) for i in range(n)]


def _circuits_from_counts(counts: list[int]) -> list[list[int]]:
    return [_gates(n, seed=i) for i, n in enumerate(counts)]


def _node(*, use_input_stream: bool) -> SimpleNamespace:
    return SimpleNamespace(parameters=SimpleNamespace(use_input_stream=use_input_stream))


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

    streamed_stack = np.stack(
        [_reshape_streamed(ev, n_shots, len(depths), n_seq) for ev in streamed_events]
    )
    declared_stack = np.stack(
        [_reshape_declared(ev, n_shots, len(depths), n_seq) for ev in declared_events]
    )

    ds_stream = _dataset_from_raw(
        streamed_stack, pair_names, depths, n_seq, n_shots, use_input_stream=True
    )
    ds_decl = _dataset_from_raw(
        declared_stack, pair_names, depths, n_seq, n_shots, use_input_stream=False
    )
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
                        proc_s["state"].sel(
                            qubit_pair=name, circuit_depth=depth, sequence=seq, shots=shot
                        ).item()
                    )
                    assert got == expected

    assert rb_progress_total(n_shots, n_seq, len(depths), use_input_stream=True) == n_shots * n_seq * len(
        depths
    )
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

    legacy_key = cache_key(seed, depths, n_per, encoding_version=None)
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
_XY_SLOT_GATES = (SXGate(), XGate(), RYGate(np.pi / 2), YGate(), RYGate(3.0 * np.pi / 2), None)


def _apply_xy_slot(qc: QuantumCircuit, slot: int, qubit: int) -> None:
    """Software analog of play_gate 1Q slots 0–4 (idle slot 5 is a no-op)."""
    gate = _XY_SLOT_GATES[slot]
    if gate is not None:
        qc.append(gate, [qubit])


def _apply_opcode(qc: QuantumCircuit, opcode: int) -> None:
    """Replay X90/X180/Y90/Y180/−Y90 and CZ without frame_rotation."""
    if opcode == 36:
        qc.cz(0, 1)
        return
    if opcode == 37:
        return
    if not 0 <= opcode <= 35:
        raise AssertionError(f"unexpected opcode {opcode}")
    g_c, g_t = divmod(opcode, 6)
    _apply_xy_slot(qc, g_c, 0)
    _apply_xy_slot(qc, g_t, 1)


def test_get_gate_name_and_layer_ints_reject_rz():
    with pytest.raises(ValueError, match="rz is not in the analog-XY"):
        get_gate_name(RZGate(np.pi / 2))
    with pytest.raises(ValueError, match="rz is not in the analog-XY"):
        get_gate_name(SimpleNamespace(name="rz", params=[np.pi]))

    assert get_gate_name(SXGate()) == "sx"
    assert get_gate_name(XGate()) == "x"
    assert get_gate_name(YGate()) == "y"
    assert get_gate_name(RYGate(np.pi / 2)) == "ry(pi/2)"
    assert get_gate_name(RYGate(np.pi)) == "y"
    assert get_gate_name(RYGate(-np.pi)) == "y"
    assert get_gate_name(RYGate(3.0 * np.pi / 2)) == "ry(3pi/2)"
    assert get_gate_name(RYGate(-np.pi / 2)) == "ry(3pi/2)"
    with pytest.raises(ValueError, match="Unsupported angle"):
        get_gate_name(RYGate(0.3))

    rz_circuit = QuantumCircuit(2)
    rz_circuit.rz(np.pi / 2, 0)
    rz_circuit.sx(1)
    with pytest.raises(ValueError, match="rz is not in the analog-XY"):
        circuit_to_layer_ints(rz_circuit)

    mixed = QuantumCircuit(2)
    mixed.sx(0)
    mixed.ry(np.pi / 2, 1)
    assert circuit_to_layer_ints(mixed) == [2]


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
            encoded_a = circuit_to_layer_ints(qc_a)
            encoded_b = circuit_to_layer_ints(qc_b)
            assert encoded_a == encoded_b
            assert all(0 <= op <= 37 for op in encoded_a)
            assert 38 not in encoded_a


def test_opcode_software_apply_recovers_identity_clifford():
    depths = [0, 1, 3]
    rb = StandardRB(depths, num_circuits_per_length=2, seed=11, show_progress=False)
    identity = Operator(np.eye(4, dtype=complex))
    for depth in depths:
        for qc in rb.transpiled_circuits[depth]:
            replay = QuantumCircuit(2)
            for opcode in circuit_to_layer_ints(qc):
                _apply_opcode(replay, opcode)
            assert Operator(replay).equiv(identity)


def test_cache_v2_payload_is_acquisition_miss_and_v3_key_includes_basis(tmp_path: Path):
    seed, depths, n_per = 0, [0, 4], 2
    circuits = [[], [36, 14, 0], [1], [17, 36]]
    stats = {"average_gates_per_clifford": 1.5, "avg_1q_per_clifford": 1.0, "avg_cz_per_clifford": 0.5}

    key_v3 = cache_key(seed, depths, n_per)
    key_v3_explicit = cache_key(seed, depths, n_per, encoding_version=3, basis_gates=DEFAULT_RB_BASIS_GATES)
    key_v3_zx_basis = cache_key(seed, depths, n_per, encoding_version=3, basis_gates=["cz", "sx", "x", "rz"])
    key_v2 = cache_key(seed, depths, n_per, encoding_version=2, basis_gates=None)
    assert key_v3 == key_v3_explicit
    assert key_v3 != key_v2
    assert key_v3 != key_v3_zx_basis
    assert "ry" in DEFAULT_RB_BASIS_GATES and "rz" not in DEFAULT_RB_BASIS_GATES

    v2_payload = {"encoding_version": 2, "circuits_as_ints": circuits, **stats}
    (tmp_path / f"{key_v2}.json").write_text(json.dumps(v2_payload), encoding="utf-8")
    (tmp_path / f"{key_v3}.json").write_text(json.dumps(v2_payload), encoding="utf-8")
    assert try_load(tmp_path, key_v2) is None
    assert try_load(tmp_path, key_v3) is None
    assert (tmp_path / f"{key_v2}.json").exists()
    v2_stats = try_load_statistics(tmp_path, key_v2)
    assert v2_stats is not None
    assert "circuits_as_ints" not in v2_stats
    assert v2_stats["average_gates_per_clifford"] == 1.5

    save(tmp_path, key_v3, {"circuits_as_ints": circuits, **stats})
    loaded = try_load(tmp_path, key_v3)
    assert loaded is not None
    assert loaded["encoding_version"] == 3
    assert loaded["encoding_version"] == RB_ENCODING_VERSION
    assert loaded["circuits_as_ints"] == circuits


# --------------------------------------- CZ XY occupy (task-r)


def _xy_channel(operations: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(operations={} if operations is None else operations)


def _fake_pair(
    *,
    control_ops: dict | None = None,
    target_ops: dict | None = None,
    spectator_ops: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        qubit_control=SimpleNamespace(xy=_xy_channel(control_ops)),
        qubit_target=SimpleNamespace(xy=_xy_channel(target_ops)),
        spectator=SimpleNamespace(xy=_xy_channel(spectator_ops)),
    )


def test_ensure_xy_zero_pulse_writes_4ns_amp0_and_is_idempotent():
    existing = SquarePulse(length=16, amplitude=0.0)
    pair = _fake_pair(target_ops={"zero": existing})
    spectator_ops = pair.spectator.xy.operations

    ensure_xy_zero_pulse([pair])
    control_zero = pair.qubit_control.xy.operations["zero"]
    assert isinstance(control_zero, SquarePulse)
    assert control_zero.length == 4
    assert control_zero.amplitude == 0
    assert pair.qubit_target.xy.operations["zero"] is existing
    assert existing.length == 16
    assert "zero" not in spectator_ops

    ensure_xy_zero_pulse({"p": pair})
    assert pair.qubit_control.xy.operations["zero"] is control_zero
    assert pair.qubit_target.xy.operations["zero"] is existing
    assert "zero" not in spectator_ops


def test_play_gate_source_zero_occupy_only_in_cz_case():
    src = inspect.getsource(play_gate)
    impl = src.split('"""', 2)[-1]
    assert "align(" not in impl

    cz_start = impl.index("with case_(36)")
    cz_end = impl.index("with case_(37)")
    cz_body = impl[cz_start:cz_end]
    rest = impl[:cz_start] + impl[cz_end:]

    assert cz_body.count('play("zero")') == 2
    assert "qubit_control.xy.play" in cz_body and '"zero"' in cz_body
    assert "qubit_target.xy.play" in cz_body
    assert "align_elements=False" in cz_body
    assert 'play("zero")' not in rest
    assert "with case_(0)" in rest
    assert "with case_(35)" in rest
    assert "with case_(37)" in rest

    seq_src = inspect.getsource(play_sequence)
    loop_start = seq_src.index("with for_(gate_index")
    loop_end = seq_src.index("readout_save_and_reset")
    gate_for = seq_src[loop_start:loop_end]
    assert "play_gate(" in gate_for
    assert "align(" not in gate_for
