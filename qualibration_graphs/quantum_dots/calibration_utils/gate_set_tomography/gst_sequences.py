"""Build GST sequences for gate set tomography.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import pygsti

# PREP_FIDUCIAL_MAP: dict[str, int] = {
#     "{}": 0,                    # no gate applied
#     "Gxpi2:0": 1,               # X90
#     "Gypi2:0": 2,               # Y90
#     "Gxpi2:0Gxpi2": 3,          # X90X90
#     "Gxpi2:0Gxpi2:0Gxpi2:0": 4, # X90X90X90
#     "Gypi2:0Gypi2:0Gypi2:0": 5, # Y90Y90Y90
# }

# MEAS_FIDUCIAL_MAP: dict[str, int] = {
#     "{}": 0,                    # no gate applied
#     "Gxpi2:0": 1,               # X90
#     "Gypi2:0": 2,               # Y90
#     "Gxpi2:0Gxpi2": 3,          # X90X90
#     "Gxpi2:0Gxpi2:0Gxpi2:0": 4, # X90X90X90
#     "Gypi2:0Gypi2:0Gypi2:0": 5, # Y90Y90Y90
# }

# GERM_MAP: dict[str, int] = {
#     "{}": 0,                    # no gate applied
#     "Gxpi2:0": 1,               # X90
#     "Gypi2:0": 2,               # Y90
#     "Gxpi2:0Gypi2:0": 3,        # X90Y90
#     "Gxpi2:0Gxpi2:0Gypi2:0": 4, # X90X90Y90
#     "[]": 5,                    # Identity gate (must be different from {})
# }

GST_SEQUENCE_COUNT_LIMIT = 2000  # max circuits returned by build_gst_sequences


def _load_pygsti_model_pack(model_name: str):
    """Return the ``pygsti.modelpacks.<model_name>`` module, or raise ``ValueError`` if missing."""
    if not model_name.isidentifier():
        raise ValueError(
            f"Invalid GST model name {model_name!r}; expected a model pack identifier (e.g. 'smq1Q_XY')."
        )
    import pygsti.modelpacks as modelpacks_pkg  # noqa: PLC0415

    try:
        return importlib.import_module(f"pygsti.modelpacks.{model_name}")
    except ModuleNotFoundError as e:
        available = sorted(
            m.name for m in pkgutil.iter_modules(modelpacks_pkg.__path__)
        )
        raise ValueError(
            f"Unknown pyGSTi model pack {model_name!r} (no submodule pygsti.modelpacks.{model_name}). "
            f"Available model packs: {', '.join(available)}."
        ) from e


# def get_gst_components(model_name: str) -> tuple[list[str], list[str], list[str]]:
#     """Get the GST components from a model pack.

#     Args:
#         model_name: Name of a ``pygsti.modelpacks`` module (e.g. ``\"smq1Q_XY\"``, ``\"smq1Q_XYI\"``).

#     Returns:
#         Lists of preparation fiducials, measurement fiducials, germs, and target model.
#     """
    
#     pack = _load_pygsti_model_pack(model_name)
#     try:
#         prep_fiducials = pack.prep_fiducials()
#         meas_fiducials = pack.meas_fiducials()
#         germs = pack.germs()
#         target_model = pack.target_model()
#     except AttributeError as e:
#         raise ValueError(
#             f"Model pack {model_name!r} does not exist."
#         ) from e
#     return prep_fiducials, meas_fiducials, germs, target_model


def build_gst_sequences(target_model, prep_fiducials, meas_fiducials, germs, max_lengths) -> list[str]:
    """Build GST sequences for gate set tomography.

    Args:
        target_model: Target model for GST.
        prep_fiducials: Preparation fiducials.
        meas_fiducials: Measurement fiducials.
        germs: Germs that will be repeated in the GST sequences.
        max_lengths: Maximum lengths of the germ repetitions in the GST sequences.

    Returns:
        List of GST sequences converted to strings.

    Raises:
        ValueError: If ``model_name`` is not a loadable model pack, if the pack lacks
            the usual GST helpers, or if the number of sequences exceeds
            :data:`GST_SEQUENCE_COUNT_LIMIT`.
    """
    import pygsti  # noqa: PLC0415

    lsgst_lists = pygsti.circuits.create_lsgst_circuit_lists(
        target_model, prep_fiducials, meas_fiducials, germs, max_lengths
    )

    sequences = [circuit.str for circuit in lsgst_lists[-1]]
    if len(sequences) > GST_SEQUENCE_COUNT_LIMIT:
        raise ValueError(
            f"GST sequence count ({len(sequences)}) exceeds the limit ({GST_SEQUENCE_COUNT_LIMIT}). "
            "Reduce max_lengths, the fiducial set, or the germ set."
        )
    return sequences


def strip_pygsti_line_labels(sequence: str) -> str:
    """Remove trailing ``@(...)`` circuit line labels from a pyGSTi circuit string."""
    return re.sub(r"@\([^)]*\)$", "", sequence.strip())


def _normalize_empty_for_map(segment: str) -> str:
    return "{}" if segment == "" else segment


def build_gate_map(circuit_objects: list[pygsti.Circuit]) -> dict[str, int]:
    """Build a gate map from a list of circuit objects.
    Used to construct the PREP_FIDUCIAL_MAP, MEAS_FIDUCIAL_MAP and GERM_MAP dictionaries.
    
    Args:
        circuit_objects: List of circuit objects.
    """
    gate_map = {"{}": 0}
    idx = 1
    for circuit in circuit_objects:
        circuit_string = strip_pygsti_line_labels(circuit.str)
        if not circuit_string in gate_map.keys():
            gate_map[circuit_string] = idx
            idx += 1
    return gate_map


def split_lsgst_circstring(sequence: str) -> tuple[str, str, int, str]:
    """Split a pyGSTi LSGST circuit string into prep, germ body, repetition count, and meas.

    Expected format::

        <prep>(<germ>)^N<meas>[@(line)]

    Uses the first ``(`` and the first ``)`` after it as the germ delimiters.
    If ``^N`` is omitted, ``N`` is 1. Prep, germ, and meas are normalized so that empty
    strings become ``\"{}\"`` (for map lookup). If the germ is ``\"{}\"`` after normalization,
    repetition is reported as ``0`` (identity germ).
    """
    s = strip_pygsti_line_labels(sequence)
    open_idx = s.find("(")
    close_idx = s.find(")", open_idx)

    prep = s[:open_idx]
    germ = s[open_idx + 1 : close_idx]
    tail = s[close_idx + 1 :]

    m = re.match(r"\^(\d+)(.*)$", tail)
    if m:
        repetition = int(m.group(1))
        meas = m.group(2)
    else:
        repetition = 1
        meas = tail

    prep = _normalize_empty_for_map(prep)
    germ = _normalize_empty_for_map(germ)
    meas = _normalize_empty_for_map(meas)
    if germ == "{}":
        repetition = 0

    return prep, germ, repetition, meas


def gst_sequence_to_indices(sequence: str, prep_fiducial_map: dict[str, int], meas_fiducial_map: dict[str, int], germ_map: dict[str, int]) -> list[int]:
    """Map a pyGSTi GST circuit string to ``[prep_idx, meas_idx, germ_idx, repetition]``.

    Args:
        sequence: GST circuit string in pyGSTi line format.
        prep_fiducial_map: Map of preparation fiducials to indices.
        meas_fiducial_map: Map of measurement fiducials to indices.
        germ_map: Map of germs to indices.

    Returns:
        List of indices: [prep_idx, meas_idx, germ_idx, repetition].

    Repetition is ``0`` when the germ is ``\"{}\"``; see :func:`split_lsgst_circstring`.
    """
    prep_s, germ_s, repetition, meas_s = split_lsgst_circstring(sequence)

    def miss(name: str, key: str, map: dict[str, int]) -> int:
        try:
            return map[key]
        except KeyError as e:
            raise KeyError(
                f"Unknown GST {name} segment {key!r}; keys are {list(map.keys())}"
            ) from e

    return [
        miss("prep", prep_s, prep_fiducial_map),
        miss("meas", meas_s, meas_fiducial_map),
        miss("germ", germ_s, germ_map),
        repetition,
    ]


def gst_sequences_to_index_lists(
    sequences: list[str],
    prep_fiducial_map: dict[str, int],
    meas_fiducial_map: dict[str, int],
    germ_map: dict[str, int],
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Convert a list of GST circuit strings to four parallel index lists.

    For each string, applies :func:`gst_sequence_to_indices` and collects
    preparation, measurement, germ indices, and repetition counts into
    separate lists. Each list has length ``n = len(sequences)`` (all empty
    if ``sequences`` is empty).

    Args:
        sequences: GST circuit strings in pyGSTi line format.
        prep_fiducial_map: Map of preparation fiducials to indices.
        meas_fiducial_map: Map of measurement fiducials to indices.
        germ_map: Map of germs to indices.
    Returns:
        ``(prep_indices, meas_indices, germ_indices, repetitions)``, each a
        ``list[int]`` of length ``n``.
    """
    if not sequences:
        return [], [], [], []
    rows = [gst_sequence_to_indices(s, prep_fiducial_map, meas_fiducial_map, germ_map) for s in sequences]
    prep_indices, meas_indices, germ_indices, repetitions = map(list, zip(*rows))
    return prep_indices, meas_indices, germ_indices, repetitions


def _normalize_pygsti_segment(segment: str) -> str:
    """Strip line labels and whitespace from a pyGSTi fiducial/germ/meas segment string."""
    s = strip_pygsti_line_labels(segment.strip())
    return re.sub(r"\s+", "", s)


def tokenize_pygsti_segment(segment: str, basic_gates_map: dict[str, str]) -> list[str]:
    """Split a pyGSTi layer string into primitive labels using longest-prefix matching.

    ``basic_gates_map`` keys (except the empty-circuit sentinel ``\"{}\"``) define the
    vocabulary. At each position we pick the **longest** key that matches from that
    position. That avoids ambiguity when one label is a prefix of another (if both
    ``Gxpi2`` and ``Gxpi2:0`` were in the map, shorter-first would wrongly split
    ``Gxpi2:0`` into ``Gxpi2`` plus leftover ``:0``). Keys are sorted by length only
    to implement that longest-match rule, not for alphabetical ordering.

    Parameters
    ----------
    segment
        A single fiducial, germ, or meas substring as stored in :func:`build_gate_map`
        (e.g. ``Gxpi2:0Gypi2:0``).
    basic_gates_map
        Maps pyGSTi gate labels to QUAM macro attribute names; keys supply the tokenizer
        vocabulary.

    Returns
    -------
    list[str]
        Ordered pyGSTi labels (keys into ``basic_gates_map``).

    Raises
    ------
    ValueError
        If the segment cannot be fully tokenized.
    """
    s = _normalize_pygsti_segment(segment)
    if s == "" or s == "{}":
        return []
    # Same keys as basic_gates_map; sorted by length only so try-order is longest-first
    # (dict iteration order is insertion order and does not imply longest-prefix match).
    vocab = sorted(
        (k for k in basic_gates_map if k not in ("{}",)),
        key=len,
        reverse=True,
    )
    if not vocab:
        raise ValueError("basic_gates_map must contain at least one gate label besides '{}'.")
    tokens: list[str] = []
    pos = 0
    while pos < len(s):
        for key in vocab:
            if s.startswith(key, pos):
                tokens.append(key)
                pos += len(key)
                break
        else:
            raise ValueError(
                f"Cannot tokenize pyGSTi segment {segment!r} at position {pos}: "
                f"remainder {s[pos:]!r}. Vocabulary (from basic_gates_map keys): {vocab}"
            )
    return tokens


def circuit_segment_to_macro_names(segment: str, basic_gates_map: dict[str, str]) -> list[str]:
    """Translate a pyGSTi segment string to QUAM macro attribute names (e.g. ``\"x90\"``).

    The empty circuit ``\"{}\"`` (after normalization) yields an empty list (no pulses).
    """
    s = _normalize_pygsti_segment(segment)
    if s == "" or s == "{}":
        return []
    macro_names: list[str] = []
    for label in tokenize_pygsti_segment(segment, basic_gates_map):
        try:
            macro = basic_gates_map[label]
        except KeyError as e:
            raise KeyError(
                f"No basic_gates_map entry for pyGSTi label {label!r} (from segment {segment!r})."
            ) from e
        if macro:
            macro_names.append(macro)
    return macro_names


def build_gst_playback_macro_lists(
    gate_map: dict[str, int],
    basic_gates_map: dict[str, str],
    qubit: Any,
) -> list[list[Callable[..., None]]]:
    """Build per-``case_(i)`` lists of bound qubit gate callables for QUA ``switch_`` playback.

    Index *i* is the integer stored in ``gate_map`` for some circuit string (same dense
    indexing as :func:`build_gate_map`: contiguous ``0 .. max``).

    Parameters
    ----------
    gate_map
        Maps segment strings (e.g. ``Gxpi2:0Gxpi2:0Gypi2:0``) to case indices.
    basic_gates_map
        Maps each pyGSTi primitive label to a qubit attribute name (``\"x90\"``, …).
    qubit
        QUAM qubit (or any object) exposing those attributes as callable macros.

    Returns
    -------
    list[list[Callable[..., None]]]
        ``result[i]`` is the ordered list of callables for ``with case_(i):``.

    Raises
    ------
    AttributeError
        If a macro name is missing on ``qubit``.
    ValueError
        If ``gate_map`` indices are not contiguous from 0.
    """
    if not gate_map:
        return []
    max_idx = max(gate_map.values())
    circuits_by_idx: list[str | None] = [None] * (max_idx + 1)
    for circuit_str, idx in gate_map.items():
        circuits_by_idx[idx] = circuit_str
    missing_idx = [i for i, c in enumerate(circuits_by_idx) if c is None]
    if missing_idx:
        raise ValueError(
            "gate_map must use contiguous indices 0..N-1; missing circuit for indices: "
            f"{missing_idx}"
        )
    out: list[list[Callable[..., None]]] = []
    for circuit_str in circuits_by_idx:
        assert circuit_str is not None
        names = circuit_segment_to_macro_names(circuit_str, basic_gates_map)
        macros: list[Callable[..., None]] = []
        for name in names:
            try:
                fn = getattr(qubit, name)
            except AttributeError as e:
                raise AttributeError(
                    f"Qubit type {type(qubit).__name__!r} has no gate macro or method {name!r} "
                    f"(from circuit {circuit_str!r})."
                ) from e
            macros.append(fn)
        out.append(macros)
    return out

