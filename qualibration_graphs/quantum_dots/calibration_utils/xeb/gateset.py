"""XEB gate set definitions and unitary matrices.

Defines the single-qubit gates used in cross-entropy benchmarking circuits.
Two gate sets are available:

    "sw": {SX (√X), SY (√Y), SW (√W)} — matches the Google/superconducting XEB convention
    "t":  {SX (√X), SY (√Y), T (R_z(π/4))} — lighter alternative with a virtual gate

Gate matrices are used for classical ideal probability computation.
The QUA playback of these gates is handled by qua_macros.play_xeb_gate().

Spin qubit native decompositions:
    SX → qubit.x90()
    SY → qubit.y90()
    SW → qubit.z_neg90() then qubit.x90()
    T  → qubit.xy.frame_rotation_2pi(0.125)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

_s = 1.0 / np.sqrt(2)

NUM_XEB_GATES: int = 3

SX_MATRIX: np.ndarray = np.array(
    [[_s, -1j * _s], [-1j * _s, _s]], dtype=complex
)

SY_MATRIX: np.ndarray = np.array(
    [[_s, -_s], [_s, _s]], dtype=complex
)

SW_MATRIX: np.ndarray = np.array(
    [[1, -(1j**0.5)], [(-1j) ** 0.5, 1]], dtype=complex
) / np.sqrt(2)

T_MATRIX: np.ndarray = np.array(
    [[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex
)


def get_gate_matrices(gate_set: Literal["sw", "t"]) -> list[np.ndarray]:
    """Return the list of 2x2 unitary matrices for the chosen gate set.

    Parameters
    ----------
    gate_set : {"sw", "t"}
        Gate set identifier.

    Returns
    -------
    list of ndarray
        Three 2x2 complex unitary matrices indexed by gate integer (0, 1, 2).
    """
    if gate_set == "sw":
        return [SX_MATRIX, SY_MATRIX, SW_MATRIX]
    elif gate_set == "t":
        return [SX_MATRIX, SY_MATRIX, T_MATRIX]
    else:
        raise ValueError(f"Unknown gate set: {gate_set!r}. Use 'sw' or 't'.")
