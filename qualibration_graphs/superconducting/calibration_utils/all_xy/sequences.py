"""Canonical All-XY gate-pair sequences and derived labels."""

import numpy as np

# 21 pairs of single-qubit gates (Reed's thesis / Phys. Rev. A 82).
# "I" = identity (wait of x90 duration). Other names must match qubit.xy.operations.
ALL_XY_SEQUENCES = [
    ("I", "I"),
    ("x180", "x180"),
    ("y180", "y180"),
    ("x180", "y180"),
    ("y180", "x180"),
    ("x90", "I"),
    ("y90", "I"),
    ("x90", "y90"),
    ("y90", "x90"),
    ("x90", "y180"),
    ("y90", "x180"),
    ("x180", "y90"),
    ("y180", "x90"),
    ("x90", "x180"),
    ("x180", "x90"),
    ("y90", "y180"),
    ("y180", "y90"),
    ("x180", "I"),
    ("y180", "I"),
    ("x90", "x90"),
    ("y90", "y90"),
]

ALL_XY_LABELS = [f"{g1},{g2}" for g1, g2 in ALL_XY_SEQUENCES]
N_ALL_XY = len(ALL_XY_SEQUENCES)

# Ideal population staircase: ground, superposition, excited.
N_GROUND, N_SUPERPOSITION, N_EXCITED = 5, 12, 4
IDEAL_ALL_XY = np.array(
    [0.0] * N_GROUND + [0.5] * N_SUPERPOSITION + [1.0] * N_EXCITED,
    dtype=float,
)
