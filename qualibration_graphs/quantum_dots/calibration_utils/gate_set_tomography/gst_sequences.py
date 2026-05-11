"""Build GST sequences for gate set tomography.
"""

from __future__ import annotations

import importlib
import pkgutil
import re

PREP_FIDUCIAL_MAP: dict[str, int] = {
    "{}": 0,                    # no gate applied
    "Gxpi2:0": 1,               # X90
    "Gypi2:0": 2,               # Y90
    "Gxpi2:0Gxpi2": 3,          # X90X90
    "Gxpi2:0Gxpi2:0Gxpi2:0": 4, # X90X90X90
    "Gypi2:0Gypi2:0Gypi2:0": 5, # Y90Y90Y90
}

MEAS_FIDUCIAL_MAP: dict[str, int] = {
    "{}": 0,                    # no gate applied
    "Gxpi2:0": 1,               # X90
    "Gypi2:0": 2,               # Y90
    "Gxpi2:0Gxpi2": 3,          # X90X90
    "Gxpi2:0Gxpi2:0Gxpi2:0": 4, # X90X90X90
    "Gypi2:0Gypi2:0Gypi2:0": 5, # Y90Y90Y90
}

GERM_MAP: dict[str, int] = {
    "{}": 0,                    # no gate applied
    "Gxpi2:0": 1,               # X90
    "Gypi2:0": 2,               # Y90
    "Gxpi2:0Gypi2:0": 3,        # X90Y90
    "Gxpi2:0Gxpi2:0Gypi2:0": 4, # X90X90Y90
    "[]": 5,                    # Identity gate (must be different from {})
}

GST_SEQUENCE_COUNT_LIMIT = 2000  # max circuits returned by build_gst_sequences


def strip_pygsti_line_labels(sequence: str) -> str:
    """Remove trailing ``@(...)`` circuit line labels from a pyGSTi circuit string."""
    return re.sub(r"@\([^)]*\)$", "", sequence.strip())


def _normalize_empty_for_map(segment: str) -> str:
    return "{}" if segment == "" else segment


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


def gst_sequence_to_indices(sequence: str) -> list[int]:
    """Map a pyGSTi GST circuit string to ``[prep_idx, meas_idx, germ_idx, repetition]``.

    Uses :data:`PREP_FIDUCIAL_MAP`, :data:`MEAS_FIDUCIAL_MAP`, and :data:`GERM_MAP`.
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
        miss("prep", prep_s, PREP_FIDUCIAL_MAP),
        miss("meas", meas_s, MEAS_FIDUCIAL_MAP),
        miss("germ", germ_s, GERM_MAP),
        repetition,
    ]


def gst_sequences_to_index_lists(
    sequences: list[str],
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Convert a list of GST circuit strings to four parallel index lists.

    For each string, applies :func:`gst_sequence_to_indices` and collects
    preparation, measurement, germ indices, and repetition counts into
    separate lists. Each list has length ``n = len(sequences)`` (all empty
    if ``sequences`` is empty).

    Args:
        sequences: GST circuit strings in pyGSTi line format.

    Returns:
        ``(prep_indices, meas_indices, germ_indices, repetitions)``, each a
        ``list[int]`` of length ``n``.
    """
    if not sequences:
        return [], [], [], []
    rows = [gst_sequence_to_indices(s) for s in sequences]
    prep_indices, meas_indices, germ_indices, repetitions = map(list, zip(*rows))
    return prep_indices, meas_indices, germ_indices, repetitions


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


def build_gst_sequences(model_name: str, max_lengths: list[int]) -> list[str]:
    """Build GST sequences for gate set tomography.

    Args:
        model_name: Name of a ``pygsti.modelpacks`` module (e.g. ``\"smq1Q_XY\"``, ``\"smq1Q_XYI\"``).
        max_lengths: Maximum lengths of the GST sequences.

    Returns:
        List of GST sequences.

    Raises:
        ValueError: If ``model_name`` is not a loadable model pack, if the pack lacks
            the usual GST helpers, or if the number of sequences exceeds
            :data:`GST_SEQUENCE_COUNT_LIMIT`.
    """
    import pygsti  # noqa: PLC0415

    pack = _load_pygsti_model_pack(model_name)
    try:
        prep_fiducials = pack.prep_fiducials()
        meas_fiducials = pack.meas_fiducials()
        germs = pack.germs()
        target_model = pack.target_model()
    except AttributeError as e:
        raise ValueError(
            f"Model pack {model_name!r} does not exist."
        ) from e

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