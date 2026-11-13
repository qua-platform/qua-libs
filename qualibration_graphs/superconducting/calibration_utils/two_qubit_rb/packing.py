"""Host-side packing of gate-only two-qubit RB circuits.

Packed layout (headers are never passed to the gate switch)::

    [n_circuits, len0, gate0_0, ..., len1, gate1_0, ..., ...]

Example: ``[[], [36, 14, 0], []]`` becomes ``[3, 0, 3, 36, 14, 0, 0]``.
A length header may equal 38 without meaning readout; opcode 38 as a *gate*
is rejected.

Chunk capacity counts the circuit-count header and every length word:
``1 + sum(1 + len(circuit))``. Coherent circuits are never split.
Padding, if any, sits after the last parsed circuit and is not a pulse.
"""

from __future__ import annotations

from numbers import Integral
from typing import Iterable, Sequence

# Encoding written to the RB cache key and payload. v2 is ZX (virtual-Z) gate-only
# lists; those files are an acquisition miss and must not be replayed as analog XY.
# Unversioned / v1 marker-terminated files remain a miss. Keep statistics-only
# lookup for old averages.
RB_ENCODING_VERSION = 3

MIN_GATE_OPCODE = 0
MAX_GATE_OPCODE = 37
# Former play_gate case 38. Valid as a *length header*, never as a gate field.
READOUT_MARKER_OPCODE = 38

# Same defensive idle opcode as qua_utils.INPUT_STREAM_PAD_VALUE (case 37).
# Padding must remain outside circuit ranges and must never emit a pulse.
INPUT_STREAM_PAD_VALUE = 37

# Smallest legal packet: one empty circuit → [1, 0].
MIN_CHUNK_CAPACITY = 2


class CircuitPackingError(ValueError):
    """Invalid gate-only circuits, packed packet, or chunk capacity."""


def _require_int(value: object, *, what: str, context: str = "") -> int:
    suffix = f" ({context})" if context else ""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise CircuitPackingError(f"{what} must be an integer, got {value!r} ({type(value).__name__}){suffix}.")
    return int(value)


def packed_circuit_words(circuit: Sequence[int]) -> int:
    """Words for one circuit: length header plus gate fields."""
    return 1 + len(circuit)


def packed_packet_words(circuits: Sequence[Sequence[int]]) -> int:
    """Words for a length-delimited packet, including the circuit-count header."""
    return 1 + sum(packed_circuit_words(circuit) for circuit in circuits)


def validate_gate_opcode(opcode: object, *, context: str = "") -> int:
    """Return ``opcode`` if it is an integer in ``[0, 37]``.

    Rejects 38, negatives, floats, and unknown values. Length headers are not
    validated here: a gate-count of 38 is a 38-gate circuit, not a readout.
    """
    value = _require_int(opcode, what="Gate opcode", context=context)
    if value < MIN_GATE_OPCODE or value > MAX_GATE_OPCODE:
        if value == READOUT_MARKER_OPCODE:
            raise CircuitPackingError(
                f"Opcode {READOUT_MARKER_OPCODE} is not a gate; readout runs outside "
                f"the unsafe switch{f' ({context})' if context else ''}."
            )
        raise CircuitPackingError(
            f"Gate opcode {value} is outside [{MIN_GATE_OPCODE}, {MAX_GATE_OPCODE}]"
            f"{f' ({context})' if context else ''}."
        )
    return value


def validate_circuit(
    circuit: object,
    *,
    depth: int | None = None,
    sequence_index: int | None = None,
) -> list[int]:
    """Return a copy of *circuit* after validating every gate field."""
    loc = _circuit_context(depth, sequence_index)
    if not isinstance(circuit, (list, tuple)):
        raise CircuitPackingError(f"Circuit must be a list of gate opcodes{loc}, got {type(circuit).__name__}.")
    return [validate_gate_opcode(opcode, context=f"{loc} gate_index={i}".strip()) for i, opcode in enumerate(circuit)]


def _circuit_context(depth: int | None, sequence_index: int | None) -> str:
    parts = []
    if depth is not None:
        parts.append(f"depth={depth}")
    if sequence_index is not None:
        parts.append(f"sequence_index={sequence_index}")
    return ", ".join(parts)


def validate_circuit_list(
    circuits: Sequence[object],
    *,
    circuit_depths: Sequence[int] | None = None,
    num_circuits_per_depth: int | None = None,
) -> list[list[int]]:
    """Validate gate-only circuits and optional depth-major cardinality."""
    if circuit_depths is not None or num_circuits_per_depth is not None:
        if circuit_depths is None or num_circuits_per_depth is None:
            raise CircuitPackingError("Provide both circuit_depths and num_circuits_per_depth, or neither.")
        n_per = _require_int(num_circuits_per_depth, what="num_circuits_per_depth")
        if n_per < 0:
            raise CircuitPackingError(f"num_circuits_per_depth must be >= 0, got {n_per}.")
        depths = [_require_int(d, what="circuit depth", context=f"index={i}") for i, d in enumerate(circuit_depths)]
        expected = len(depths) * n_per
        if len(circuits) != expected:
            raise CircuitPackingError(
                "Number of circuits must equal len(circuit_depths) * num_circuits_per_depth: "
                f"got {len(circuits)} circuits, expected {len(depths)} x {n_per} = {expected}."
            )
        validated: list[list[int]] = []
        for idx, circuit in enumerate(circuits):
            depth = depths[idx // n_per] if n_per else None
            seq = idx % n_per if n_per else idx
            validated.append(validate_circuit(circuit, depth=depth, sequence_index=seq))
        return validated

    return [validate_circuit(circuit, sequence_index=i) for i, circuit in enumerate(circuits)]


def validate_chunk_capacity(max_chunk_ints: object) -> int:
    """Reject non-positive or too-small capacities before packing."""
    capacity = _require_int(max_chunk_ints, what="max_chunk_ints")
    if capacity < MIN_CHUNK_CAPACITY:
        raise CircuitPackingError(
            f"max_chunk_ints={capacity} cannot hold a packed circuit "
            f"(need at least {MIN_CHUNK_CAPACITY} words: circuit-count header + length)."
        )
    return capacity


def pack_circuits(circuits: Sequence[Sequence[int]], *, validate: bool = True) -> list[int]:
    """Pack gate-only circuits into one length-delimited packet.

    Headers never enter the gate switch. Empty circuits are ``len=0`` with no
    following gate words.
    """
    if validate:
        gates = validate_circuit_list(circuits)
    else:
        gates = [list(circuit) for circuit in circuits]
    packet = [len(gates)]
    for circuit in gates:
        packet.append(len(circuit))
        packet.extend(circuit)
    return packet


def unpack_circuits(
    packet: Sequence[object],
    *,
    allow_padding: bool = False,
    pad_value: int = INPUT_STREAM_PAD_VALUE,
    validate_gates: bool = True,
) -> list[list[int]]:
    """Parse a packed packet into ordered gate lists.

    The packet must contain exactly the declared number of complete circuits.
    Unused trailing words are allowed only when ``allow_padding`` is true and
    every leftover word equals ``pad_value``.
    """
    if not isinstance(packet, (list, tuple)):
        raise CircuitPackingError(f"Packed packet must be a sequence, got {type(packet).__name__}.")
    if len(packet) < 1:
        raise CircuitPackingError("Packed packet is empty (missing circuit-count header).")

    n_circuits = _require_int(packet[0], what="n_circuits header")
    if n_circuits < 0:
        raise CircuitPackingError(f"n_circuits header must be >= 0, got {n_circuits}.")

    cursor = 1
    circuits: list[list[int]] = []
    for circ_idx in range(n_circuits):
        if cursor >= len(packet):
            raise CircuitPackingError(
                f"Packed packet overrun: expected length header for circuit {circ_idx} "
                f"of {n_circuits}, packet length {len(packet)}."
            )
        length = _require_int(packet[cursor], what="circuit length header", context=f"circuit_index={circ_idx}")
        if length < 0:
            raise CircuitPackingError(f"Circuit length header must be >= 0, got {length} (circuit_index={circ_idx}).")
        cursor += 1
        end = cursor + length
        if end > len(packet):
            raise CircuitPackingError(
                f"Packed packet overrun: circuit_index={circ_idx} declares {length} gates "
                f"starting at index {cursor}, packet length {len(packet)}."
            )
        raw = list(packet[cursor:end])
        if validate_gates:
            circuit = validate_circuit(raw, sequence_index=circ_idx)
        else:
            circuit = [_require_int(g, what="Gate opcode", context=f"circuit_index={circ_idx}") for g in raw]
        circuits.append(circuit)
        cursor = end

    leftover = packet[cursor:]
    if leftover:
        if not allow_padding:
            raise CircuitPackingError(
                f"Packed packet has {len(leftover)} unused word(s) after {n_circuits} circuit(s)."
            )
        for offset, word in enumerate(leftover):
            value = _require_int(word, what="padding word", context=f"index={cursor + offset}")
            if value != pad_value:
                raise CircuitPackingError(
                    f"Unused non-padding data at index {cursor + offset}: {value} "
                    f"(expected pad_value={pad_value})."
                )
    return circuits


def circuit_gate_ranges(packet: Sequence[object]) -> list[tuple[int, int]]:
    """Absolute ``[start, stop)`` indices of gate fields in *packet*.

    Empty circuits have ``start == stop`` immediately after their length header.
    Headers are excluded so a QUA gate loop never sees ``n_circuits`` or lengths.
    """
    circuits_meta = _parse_headers(packet)
    return [(start, stop) for _, start, stop in circuits_meta]


def _parse_headers(packet: Sequence[object]) -> list[tuple[int, int, int]]:
    """Return ``(length_index, gate_start, gate_stop)`` for each circuit."""
    if len(packet) < 1:
        raise CircuitPackingError("Packed packet is empty (missing circuit-count header).")
    n_circuits = _require_int(packet[0], what="n_circuits header")
    if n_circuits < 0:
        raise CircuitPackingError(f"n_circuits header must be >= 0, got {n_circuits}.")
    cursor = 1
    meta: list[tuple[int, int, int]] = []
    for circ_idx in range(n_circuits):
        if cursor >= len(packet):
            raise CircuitPackingError(
                f"Packed packet overrun: expected length header for circuit {circ_idx} of {n_circuits}."
            )
        length_index = cursor
        length = _require_int(packet[cursor], what="circuit length header", context=f"circuit_index={circ_idx}")
        if length < 0:
            raise CircuitPackingError(f"Circuit length header must be >= 0, got {length} (circuit_index={circ_idx}).")
        start = cursor + 1
        stop = start + length
        if stop > len(packet):
            raise CircuitPackingError(
                f"Packed packet overrun: circuit_index={circ_idx} length={length} "
                f"needs indices [{start}, {stop}), packet length {len(packet)}."
            )
        meta.append((length_index, start, stop))
        cursor = stop
    return meta


def pad_packet(
    packet: Sequence[int],
    declared_size: int,
    *,
    pad_value: int = INPUT_STREAM_PAD_VALUE,
) -> list[int]:
    """Pad *packet* to *declared_size* with *pad_value* after the last circuit."""
    size = _require_int(declared_size, what="declared_size")
    if size < len(packet):
        raise CircuitPackingError(f"declared_size={size} is smaller than packet length {len(packet)}.")
    padded = list(packet)
    padded.extend([pad_value] * (size - len(packet)))
    return padded


def _single_circuit_too_large_message(
    *,
    depth: int,
    sequence_index: int,
    circuit_len: int,
    required_words: int,
    max_chunk_ints: int,
) -> str:
    return (
        f"Single circuit too large for one packed input-stream chunk: depth={depth} "
        f"Cliffords, sequence_index={sequence_index}, gate_count={circuit_len}, "
        f"required_words={required_words} (1 circuit-count header + 1 length + "
        f"{circuit_len} gates), available_capacity={max_chunk_ints}. "
        "Streaming cannot split a coherent circuit; reduce depth, raise "
        "max_chunk_ints (must stay below the OPX QUA variable budget), or reduce "
        "num_circuits_per_depth."
    )


def build_packed_depth_chunks(
    circuits: Sequence[Sequence[int]],
    circuit_depths: Sequence[int],
    num_circuits_per_depth: int,
    max_chunk_ints: int,
) -> tuple[list[list[list[int]]], int]:
    """Greedily pack complete circuits into per-depth length-delimited chunks.

    Depth boundaries are never crossed. A circuit that does not fit the current
    chunk starts a new chunk. Returns ``(chunks_per_depth, declared_size)``
    where each sub-chunk is an unpadded packed packet and ``declared_size`` is
    the max packet length (padding is applied later).
    """
    capacity = validate_chunk_capacity(max_chunk_ints)
    n_per = _require_int(num_circuits_per_depth, what="num_circuits_per_depth")
    depths = [_require_int(d, what="circuit depth", context=f"index={i}") for i, d in enumerate(circuit_depths)]
    gates = validate_circuit_list(circuits, circuit_depths=depths, num_circuits_per_depth=n_per)

    chunks_per_depth: list[list[list[int]]] = []
    declared_size = 0

    for depth_idx, depth in enumerate(depths):
        start = depth_idx * n_per
        group = gates[start : start + n_per]
        sub_chunks_circuits: list[list[list[int]]] = []
        current: list[list[int]] = []

        for seq_idx, circuit in enumerate(group):
            required = packed_packet_words([circuit])
            if required > capacity:
                raise CircuitPackingError(
                    _single_circuit_too_large_message(
                        depth=depth,
                        sequence_index=seq_idx,
                        circuit_len=len(circuit),
                        required_words=required,
                        max_chunk_ints=capacity,
                    )
                )
            if current and packed_packet_words(current + [circuit]) > capacity:
                sub_chunks_circuits.append(current)
                current = []
            current.append(circuit)

        if current:
            sub_chunks_circuits.append(current)

        packed_sub = [pack_circuits(chunk_circuits, validate=False) for chunk_circuits in sub_chunks_circuits]
        for packet in packed_sub:
            declared_size = max(declared_size, len(packet))
        chunks_per_depth.append(packed_sub)

    return chunks_per_depth, declared_size


def flatten_padded_chunks(
    chunks_per_depth: Sequence[Sequence[Sequence[int]]],
    declared_size: int,
    *,
    pad_value: int = INPUT_STREAM_PAD_VALUE,
) -> list[list[int]]:
    """Depth-major list of sub-chunks, each padded to *declared_size*."""
    return [
        pad_packet(sub_chunk, declared_size, pad_value=pad_value)
        for sub_chunks in chunks_per_depth
        for sub_chunk in sub_chunks
    ]


def science_readout_count(circuits: Iterable[Sequence[int]]) -> int:
    """One science readout per circuit, including empty circuits."""
    return sum(1 for _ in circuits)
