"""NumPy-based ideal output probability computation for XEB.

Computes the ideal (noiseless) probability distribution for each random
circuit by multiplying gate unitaries layer-by-layer. Supports both
single-qubit and two-qubit (with fSim/CZ gate) modes.

The 2Q gate is parameterized as:

    U_2q = RZ1(φ1) ⊗ RZ2(φ2) · fSim(θ, φ)

where fSim(θ, φ) is:
    [[1,           0,            0,         0       ],
     [0,      cos(θ),    -i·sin(θ),         0       ],
     [0,   -i·sin(θ),       cos(θ),         0       ],
     [0,           0,            0,    e^{-iφ}      ]]

Default for CZ: θ=0, φ=π.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .gateset import get_gate_matrices


def fSim(theta_iswap: float = 0.0, phi_cphase: float = np.pi) -> np.ndarray:
    """Build the 4x4 fSim gate unitary."""
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(theta_iswap), -1j * np.sin(theta_iswap), 0],
            [0, -1j * np.sin(theta_iswap), np.cos(theta_iswap), 0],
            [0, 0, 0, np.exp(-1j * phi_cphase)],
        ],
        dtype=complex,
    )


def _rz(phi: float) -> np.ndarray:
    """Single-qubit Z rotation."""
    return np.array([[1, 0], [0, np.exp(-1j * phi)]], dtype=complex)


def build_2q_gate(
    theta_iswap: float = 0.0,
    phi_cphase: float = np.pi,
    phi_rz1: float = 0.0,
    phi_rz2: float = 0.0,
) -> np.ndarray:
    """Build the full 2Q gate: RZ1⊗RZ2 · fSim."""
    rz_layer = np.kron(_rz(phi_rz2), _rz(phi_rz1))
    return rz_layer @ fSim(theta_iswap, phi_cphase)


def build_1q_lut(
    gate_set: Literal["sw", "t"] = "sw",
) -> list[np.ndarray]:
    """Return list of 1Q gate matrices indexed by gate integer."""
    return get_gate_matrices(gate_set)


def build_2q_lut(
    gate_set: Literal["sw", "t"] = "sw",
    theta_iswap: float = 0.0,
    phi_cphase: float = np.pi,
    phi_rz1: float = 0.0,
    phi_rz2: float = 0.0,
    insert_2q_gate: bool = True,
) -> list[list[np.ndarray]]:
    """Build a 3x3 lookup table of 4x4 unitaries.

    LUT[i][j] is the 4x4 unitary for applying gate i on qubit 1 and
    gate j on qubit 2, followed by the 2Q gate (if insert_2q_gate=True).

    Returns
    -------
    list of list of ndarray
        LUT[i_q1][i_q2] — 4x4 unitary.
    """
    sq_gates = get_gate_matrices(gate_set)
    gate_2q = build_2q_gate(theta_iswap, phi_cphase, phi_rz1, phi_rz2)

    num_gates = len(sq_gates)
    lut = []
    for i in range(num_gates):
        row = []
        for j in range(num_gates):
            sq_layer = np.kron(sq_gates[j], sq_gates[i])
            if insert_2q_gate:
                row.append(gate_2q @ sq_layer)
            else:
                row.append(sq_layer)
        lut.append(row)
    return lut


def calc_ideal_probs_1q(
    gate_indices: np.ndarray,
    depths: np.ndarray,
    gate_set: Literal["sw", "t"] = "sw",
) -> np.ndarray:
    """Compute ideal output probabilities for single-qubit XEB.

    Parameters
    ----------
    gate_indices : ndarray, shape (n_sequences, max_depth)
        Random gate indices for one qubit.
    depths : ndarray of int
        Depth checkpoints at which to record probabilities.
    gate_set : {"sw", "t"}
        Which gate set was used.

    Returns
    -------
    ndarray, shape (n_sequences, len(depths), 2)
        Ideal probability [P(|0⟩), P(|1⟩)] at each (sequence, depth).
    """
    sq_gates = get_gate_matrices(gate_set)
    n_sequences = gate_indices.shape[0]
    n_depths = len(depths)
    probs = np.zeros((n_sequences, n_depths, 2))

    for s in range(n_sequences):
        state = np.array([1, 0], dtype=complex)  # |0⟩
        depth_ptr = 0
        for d in range(gate_indices.shape[1]):
            state = sq_gates[gate_indices[s, d]] @ state
            if depth_ptr < n_depths and (d + 1) == depths[depth_ptr]:
                probs[s, depth_ptr] = np.abs(state) ** 2
                depth_ptr += 1
            if depth_ptr >= n_depths:
                break

    return probs


def calc_ideal_probs_2q(
    gate_indices: np.ndarray,
    depths: np.ndarray,
    gate_set: Literal["sw", "t"] = "sw",
    theta_iswap: float = 0.0,
    phi_cphase: float = np.pi,
    phi_rz1: float = 0.0,
    phi_rz2: float = 0.0,
    insert_2q_gate: bool = True,
) -> np.ndarray:
    """Compute ideal output probabilities for two-qubit XEB.

    Parameters
    ----------
    gate_indices : ndarray, shape (n_sequences, 2, max_depth)
        Random gate indices for both qubits.
    depths : ndarray of int
        Depth checkpoints.
    gate_set : {"sw", "t"}
        Which gate set was used.
    theta_iswap, phi_cphase, phi_rz1, phi_rz2 : float
        Parameters of the 2Q gate (fSim + single-qubit Z rotations).
    insert_2q_gate : bool
        Whether the 2Q gate is applied in the circuit.

    Returns
    -------
    ndarray, shape (n_sequences, len(depths), 4)
        Ideal probability [P(|00⟩), P(|01⟩), P(|10⟩), P(|11⟩)].
    """
    lut = build_2q_lut(
        gate_set, theta_iswap, phi_cphase, phi_rz1, phi_rz2, insert_2q_gate
    )
    n_sequences = gate_indices.shape[0]
    n_depths = len(depths)
    probs = np.zeros((n_sequences, n_depths, 4))

    for s in range(n_sequences):
        state = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
        depth_ptr = 0
        for d in range(gate_indices.shape[2]):
            i_q1 = gate_indices[s, 0, d]
            i_q2 = gate_indices[s, 1, d]
            state = lut[i_q1][i_q2] @ state
            if depth_ptr < n_depths and (d + 1) == depths[depth_ptr]:
                probs[s, depth_ptr] = np.abs(state) ** 2
                depth_ptr += 1
            if depth_ptr >= n_depths:
                break

    return probs
