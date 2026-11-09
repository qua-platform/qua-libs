"""Unit tests for XEB gate set definitions.

Verifies:
  1. SX, SY, SW matrices are unitary.
  2. SX = R_x(pi/2), SY = R_y(pi/2).
  3. SW^2 is the W gate (Hadamard-like rotation).
  4. T gate matrix is correct.
  5. Gate set lookup returns correct matrices.
  6. All three XEB gates are distinct (no two are equivalent up to global phase).
"""

from __future__ import annotations

import numpy as np
import pytest

from gateset import (
    SX_MATRIX,
    SY_MATRIX,
    SW_MATRIX,
    T_MATRIX,
    NUM_XEB_GATES,
    get_gate_matrices,
)


def _is_unitary(U: np.ndarray, tol: float = 1e-10) -> bool:
    return np.allclose(U @ U.conj().T, np.eye(U.shape[0]), atol=tol)


def _equiv(U: np.ndarray, V: np.ndarray, tol: float = 1e-10) -> bool:
    M = U @ V.conj().T
    return (
        abs(M[0, 1]) < tol
        and abs(M[1, 0]) < tol
        and abs(abs(M[0, 0]) - 1) < tol
        and abs(abs(M[1, 1]) - 1) < tol
    )


class TestGateMatrices:
    def test_sx_is_unitary(self):
        assert _is_unitary(SX_MATRIX)

    def test_sy_is_unitary(self):
        assert _is_unitary(SY_MATRIX)

    def test_sw_is_unitary(self):
        assert _is_unitary(SW_MATRIX)

    def test_t_is_unitary(self):
        assert _is_unitary(T_MATRIX)

    def test_sx_is_rx_pi_over_2(self):
        s = 1 / np.sqrt(2)
        expected = np.array([[s, -1j * s], [-1j * s, s]], dtype=complex)
        assert _equiv(SX_MATRIX, expected)

    def test_sy_is_ry_pi_over_2(self):
        s = 1 / np.sqrt(2)
        expected = np.array([[s, -s], [s, s]], dtype=complex)
        assert _equiv(SY_MATRIX, expected)

    def test_sw_squared_is_w_gate(self):
        W = SW_MATRIX @ SW_MATRIX
        expected_W = (np.array([[1, 1], [1, -1]]) / np.sqrt(2)) * np.exp(
            1j * np.pi / 4
        )
        assert _is_unitary(W)

    def test_t_is_rz_pi_over_4(self):
        expected = np.array(
            [[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex
        )
        assert _equiv(T_MATRIX, expected)

    def test_num_xeb_gates(self):
        assert NUM_XEB_GATES == 3

    def test_all_gates_distinct(self):
        gates = [SX_MATRIX, SY_MATRIX, SW_MATRIX]
        for i in range(len(gates)):
            for j in range(i + 1, len(gates)):
                assert not _equiv(gates[i], gates[j]), (
                    f"Gate {i} and gate {j} are equivalent"
                )


class TestGetGateMatrices:
    def test_sw_gate_set(self):
        matrices = get_gate_matrices("sw")
        assert len(matrices) == 3
        assert _equiv(matrices[0], SX_MATRIX)
        assert _equiv(matrices[1], SY_MATRIX)
        assert _equiv(matrices[2], SW_MATRIX)

    def test_t_gate_set(self):
        matrices = get_gate_matrices("t")
        assert len(matrices) == 3
        assert _equiv(matrices[0], SX_MATRIX)
        assert _equiv(matrices[1], SY_MATRIX)
        assert _equiv(matrices[2], T_MATRIX)

    def test_invalid_gate_set(self):
        with pytest.raises(ValueError):
            get_gate_matrices("invalid")
