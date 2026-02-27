"""Single-qubit Clifford group algebra for PPU-based randomized benchmarking.

This module provides hardcoded lookup tables that the QUA program uses to
generate, compose, and invert random Clifford circuits entirely on the PPU.
The tables are computed once in Python and loaded as QUA
``declare(int, value=...)`` arrays.

Native gate set
---------------
The 24 single-qubit Cliffords are decomposed into physical and virtual gates:

    Physical (Gaussian pulses on the XY channel):
        x90, x180, xm90, y90, y180, ym90

    Virtual (frame rotations, zero duration):
        z90   (R_z(+pi/2))
        z180  (R_z(pi))
        zm90  (R_z(-pi/2))

All physical gates derive from a single calibrated X180 pulse:
  - X rotations use amplitude_scale = theta / 180
  - Y rotations apply a +90 deg frame shift, play the X equivalent,
    then undo the frame shift

Gate integer encoding
---------------------
Each native gate is assigned a unique integer for efficient QUA lookup:

    0: x90     1: x180    2: xm90
    3: y90     4: y180    5: ym90
    6: z90     7: z180    8: zm90

Sequence convention
-------------------
Gates are listed in chronological pulse order (left = first applied).
E.g. ['ym90', 'z180'] means: apply ym90 first, then the virtual z180.

Cayley table convention
-----------------------
CAYLEY[i][j] = k  means  C_i . C_j = C_k
i.e. C_j is applied first (to the state), then C_i.

Inversion / RB recovery
------------------------
INVERSES[i] = j  such that  C_i . C_j = I  (C_j undoes C_i).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Gate integer map
# ---------------------------------------------------------------------------

NATIVE_GATE_MAP: dict[str, int] = {
    "x90": 0,
    "x180": 1,
    "xm90": 2,
    "y90": 3,
    "y180": 4,
    "ym90": 5,
    "z90": 6,
    "z180": 7,
    "zm90": 8,
}

NUM_CLIFFORDS = 24

# ---------------------------------------------------------------------------
# Decomposition table
# ---------------------------------------------------------------------------
# Maps each Clifford index to its native-gate pulse sequence (chronological).
# Every Clifford decomposes into at most 1 physical pulse + 0-1 virtual Z.

_DECOMPOSITION_SEQUENCES: list[list[str]] = [
    [],  # 0:  I
    ["ym90", "z180"],  # 1:  H
    ["z90"],  # 2:  S
    ["ym90", "zm90"],  # 3:  SH
    ["xm90", "zm90"],  # 4:  HS
    ["z180"],  # 5:  Z
    ["x90"],  # 6:  X90
    ["ym90"],  # 7:  Ym90
    ["xm90"],  # 8:  Xm90
    ["y90"],  # 9:  Y90
    ["zm90"],  # 10: S†
    ["x90", "z90"],  # 11: X90·S
    ["x180"],  # 12: X
    ["ym90", "z90"],  # 13: Ym90·S
    ["xm90", "z90"],  # 14: Xm90·S
    ["y90", "z90"],  # 15: Y90·S
    ["x90", "z180"],  # 16: X90·Z
    ["x180", "z90"],  # 17: X·S
    ["x180", "zm90"],  # 18: X·S†
    ["xm90", "z180"],  # 19: Xm90·Z
    ["y90", "z180"],  # 20: Y90·Z
    ["y90", "zm90"],  # 21: Y90·S†
    ["x90", "zm90"],  # 22: X90·S†
    ["y180"],  # 23: Y
]

# ---------------------------------------------------------------------------
# Inverse table
# ---------------------------------------------------------------------------
# INVERSES[i] = j  such that  CAYLEY[i][j] = 0  (identity)

INVERSES: list[int] = [
    0,  # I        → I
    1,  # H        → H         (self-inverse)
    10,  # S        → S†
    11,  # SH       → X90·S
    13,  # HS       → Ym90·S
    5,  # Z        → Z         (self-inverse)
    8,  # X90      → Xm90
    9,  # Ym90     → Y90
    6,  # Xm90     → X90
    7,  # Y90      → Ym90
    2,  # S†       → S
    3,  # X90·S    → SH
    12,  # X        → X         (self-inverse)
    4,  # Ym90·S   → HS
    21,  # Xm90·S   → Y90·S†
    22,  # Y90·S    → X90·S†
    16,  # X90·Z    → X90·Z     (self-inverse)
    17,  # X·S      → X·S       (self-inverse)
    18,  # X·S†     → X·S†      (self-inverse)
    19,  # Xm90·Z   → Xm90·Z   (self-inverse)
    20,  # Y90·Z    → Y90·Z     (self-inverse)
    14,  # Y90·S†   → Xm90·S
    15,  # X90·S†   → Y90·S
    23,  # Y        → Y         (self-inverse)
]

# ---------------------------------------------------------------------------
# Cayley table
# ---------------------------------------------------------------------------
# CAYLEY[i][j] = k  means  C_i · C_j = C_k  (C_j first, then C_i)

# fmt: off
CAYLEY: list[list[int]] = [
    [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    [ 1,  0,  4,  6,  2,  9,  3, 12, 13,  5, 11, 10,  7,  8, 18, 19, 21, 22, 14, 15, 23, 16, 17, 20],
    [ 2,  3,  5,  7,  8, 10, 11, 13, 14, 15,  0, 16, 17,  1, 19, 20, 22, 23, 12,  4, 21,  9,  6, 18],
    [ 3,  2,  8, 11,  5, 15,  7, 17,  1, 10, 16,  0, 13, 14, 12,  4,  9,  6, 19, 20, 18, 22, 23, 21],
    [ 4,  6,  9, 12, 13, 11, 10,  8, 18, 19,  1, 21, 22,  0, 15, 23, 17, 20,  7,  2, 16,  5,  3, 14],
    [ 5,  7, 10, 13, 14,  0, 16,  1, 19, 20,  2, 22, 23,  3,  4, 21,  6, 18, 17,  8,  9, 15, 11, 12],
    [ 6,  4, 13, 10,  9, 19, 12, 22,  0, 11, 21,  1,  8, 18,  7,  2,  5,  3, 15, 23, 14, 17, 20, 16],
    [ 7,  5, 14, 16, 10, 20, 13, 23,  3,  0, 22,  2,  1, 19, 17,  8, 15, 11,  4, 21, 12,  6, 18,  9],
    [ 8, 11, 15, 17,  1, 16,  0, 14, 12,  4,  3,  9,  6,  2, 20, 18, 23, 21, 13,  5, 22, 10,  7, 19],
    [ 9, 12, 11,  8, 18,  1, 21,  0, 15, 23,  4, 17, 20,  6,  2, 16,  3, 14, 22, 13,  5, 19, 10,  7],
    [10, 13,  0,  1, 19,  2, 22,  3,  4, 21,  5,  6, 18,  7,  8,  9, 11, 12, 23, 14, 15, 20, 16, 17],
    [11,  8,  1,  0, 15,  4, 17,  6,  2, 16,  9,  3, 14, 12, 13,  5, 10,  7, 20, 18, 19, 23, 21, 22],
    [12,  9, 18, 21, 11, 23,  8, 20,  6,  1, 17,  4,  0, 15, 22, 13, 19, 10,  2, 16,  7,  3, 14,  5],
    [13, 10, 19, 22,  0, 21,  1, 18,  7,  2,  6,  5,  3,  4, 23, 14, 20, 16,  8,  9, 17, 11, 12, 15],
    [14, 16, 20, 23,  3, 22,  2, 19, 17,  8,  7, 15, 11,  5, 21, 12, 18,  9,  1, 10,  6,  0, 13,  4],
    [15, 17, 16, 14, 12,  3,  9,  2, 20, 18,  8, 23, 21, 11,  5, 22,  7, 19,  6,  1, 10,  4,  0, 13],
    [16, 14,  3,  2, 20,  8, 23, 11,  5, 22, 15,  7, 19, 17,  1, 10,  0, 13, 21, 12,  4, 18,  9,  6],
    [17, 15, 12,  9, 16, 18, 14, 21, 11,  3, 23,  8,  2, 20,  6,  1,  4,  0,  5, 22, 13,  7, 19, 10],
    [18, 21, 23, 20,  6, 17,  4, 15, 22, 13, 12, 19, 10,  9, 16,  7, 14,  5,  0, 11,  3,  1,  8,  2],
    [19, 22, 21, 18,  7,  6,  5,  4, 23, 14, 13, 20, 16, 10,  9, 17, 12, 15,  3,  0, 11,  2,  1,  8],
    [20, 23, 22, 19, 17,  7, 15,  5, 21, 12, 14, 18,  9, 16, 10,  6, 13,  4, 11,  3,  0,  8,  2,  1],
    [21, 18,  6,  4, 23, 13, 20, 10,  9, 17, 19, 12, 15, 22,  0, 11,  1,  8, 16,  7,  2, 14,  5,  3],
    [22, 19,  7,  5, 21, 14, 18, 16, 10,  6, 20, 13,  4, 23,  3,  0,  2,  1,  9, 17,  8, 12, 15, 11],
    [23, 20, 17, 15, 22, 12, 19,  9, 16,  7, 18, 14,  5, 21, 11,  3,  8,  2, 10,  6,  1, 13,  4,  0],
]
# fmt: on


# ---------------------------------------------------------------------------
# Public API — build all tables for QUA
# ---------------------------------------------------------------------------


def build_single_qubit_clifford_tables() -> dict[str, list[int] | int]:
    """Build PPU lookup tables for single-qubit randomized benchmarking.

    Returns hardcoded, pre-verified Clifford group tables ready to be
    loaded into QUA ``declare(int, value=...)`` arrays.

    Returns
    -------
    dict
        Keys:

        - ``num_cliffords`` (int): Always 24.
        - ``compose`` (list[int]): Flattened 24x24 Cayley table.
          ``compose[i * 24 + j]`` = index of ``C_i . C_j``.
        - ``inverse`` (list[int]): ``inverse[i]`` = index of C_i^{-1}.
        - ``decomp_flat`` (list[int]): Concatenated gate-integer sequences
          for all 24 Cliffords.
        - ``decomp_offsets`` (list[int]): Start offset into ``decomp_flat``
          for each Clifford.
        - ``decomp_lengths`` (list[int]): Length of each decomposition.
        - ``max_decomp_length`` (int): Longest decomposition (always 2).
    """
    compose_flat: list[int] = []
    for row in CAYLEY:
        compose_flat.extend(row)

    decomp_flat: list[int] = []
    decomp_offsets: list[int] = []
    decomp_lengths: list[int] = []

    for seq in _DECOMPOSITION_SEQUENCES:
        decomp_offsets.append(len(decomp_flat))
        gate_ints = [NATIVE_GATE_MAP[g] for g in seq]
        decomp_lengths.append(len(gate_ints))
        decomp_flat.extend(gate_ints)

    return {
        "num_cliffords": NUM_CLIFFORDS,
        "compose": compose_flat,
        "inverse": list(INVERSES),
        "decomp_flat": decomp_flat,
        "decomp_offsets": decomp_offsets,
        "decomp_lengths": decomp_lengths,
        "max_decomp_length": max(decomp_lengths) if decomp_lengths else 0,
    }


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def compose(i: int, j: int) -> int:
    """Return index of C_i . C_j  (C_j applied first, then C_i)."""
    return CAYLEY[i][j]


def compose_sequence(indices) -> int:
    """Return the net Clifford index for a sequence applied left-to-right."""
    net = 0
    for c in indices:
        net = CAYLEY[c][net]
    return net


def invert(i: int) -> int:
    """Return the index of the inverse of C_i."""
    return INVERSES[i]


def decomposition(i: int) -> list[str]:
    """Return the native-gate pulse sequence for Clifford i."""
    return list(_DECOMPOSITION_SEQUENCES[i])
