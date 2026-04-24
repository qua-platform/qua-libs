"""Build GST sequences for gate set tomography.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pygsti

PREP_FIDUCIAL_MAP: dict[str, int] = {
    "{}": 0,
    "Gxpi2:0": 1,
    "Gypi2:0": 2,
    "Gxpi2:0Gxpi2": 3,
    "Gxpi2:0Gxpi2:0Gxpi2:0": 4,
    "Gypi2:0Gypi2:0Gypi2:0": 5,
}

MEAS_FIDUCIAL_MAP: dict[str, int] = {
    "{}": 0,
    "Gxpi2:0": 1,
    "Gypi2:0": 2,
    "Gxpi2:0Gxpi2": 3,
    "Gxpi2:0Gxpi2:0Gxpi2:0": 4,
    "Gypi2:0Gypi2:0Gypi2:0": 5,
}

GERM_MAP: dict[str, int] = {
    "{}": 0,
    "Gxpi2:0": 1,
    "Gypi2:0": 2,
    "Gxpi2:0Gypi2:0": 3,
    "Gxpi2:0Gxpi2:0Gypi2:0": 4,
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


def build_gst_sequences(model: pygsti.Model, max_lengths: list[int]) -> list[str]:
    """Build GST sequences for gate set tomography.

    Args:
        model: Pygsti model.
        max_lengths: Maximum lengths of the GST sequences.

    Returns:
        List of GST sequences.

    Raises:
        ValueError: If the number of sequences exceeds :data:`GST_SEQUENCE_COUNT_LIMIT`.
    """
    import pygsti  # noqa: PLC0415

    prep_fiducials = model.prep_fiducials()
    meas_fiducials = model.meas_fiducials()
    germs = model.germs()

    lsgst_lists = pygsti.circuits.create_lsgst_circuit_lists(model.target_model(), prep_fiducials, meas_fiducials, germs, max_lengths)

    sequences = [circuit.str for circuit in lsgst_lists[-1]]
    if len(sequences) > GST_SEQUENCE_COUNT_LIMIT:
        raise ValueError(
            f"GST sequence count ({len(sequences)}) exceeds the limit ({GST_SEQUENCE_COUNT_LIMIT}). "
            "Reduce max_lengths, the fiducial set, or the germ set."
        )
    return sequences