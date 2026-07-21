"""Unitary-matrix tests for clifford_tables.py.

Verifies:
  1. Every alternative decomposition is equivalent (up to global phase) to
     the corresponding reference decomposition.
  2. Alternative decompositions contain no virtual Z gates.
  3. The Cayley table: CAYLEY[i][j] = k  <=>  U(C_i) @ U(C_j) ~ U(C_k).
  4. The inverse table: INVERSES[i] = j  <=>  U(C_i) @ U(C_j) ~ I.
  5. The EPG formula derivation:
     (a) Depolarizing channels compose multiplicatively in their α parameter.
     (b) The RB decay rate equals the Clifford-average of α_gate^{n_k}.
     (c) The approximation α_gate ≈ α_RB^{1/⟨n_g⟩} holds to within the
         second-order correction Var(n_k)·(log α_gate)² / 2.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from clifford_tables import (
    CAYLEY,
    INVERSES,
    NUM_CLIFFORDS,
    _ALTERNATIVE_DECOMPOSITION_SEQUENCES,
    _DECOMPOSITION_SEQUENCES,
    _PHYSICAL_GATES,
    avg_physical_gates_per_clifford,
)

# ---------------------------------------------------------------------------
# Gate unitaries  Rn(θ) = exp(-i θ/2 σ_n)
# ---------------------------------------------------------------------------

_s = 1.0 / np.sqrt(2)

_GATE_MATRICES: dict[str, np.ndarray] = {
    "x90":  np.array([[_s, -1j * _s], [-1j * _s,  _s]], dtype=complex),
    "x180": np.array([[0,  -1j     ], [-1j,        0 ]], dtype=complex),
    "xm90": np.array([[_s,  1j * _s], [ 1j * _s,  _s]], dtype=complex),
    "y90":  np.array([[_s,  -_s    ], [ _s,       _s ]], dtype=complex),
    "y180": np.array([[0,   -1     ], [ 1,         0 ]], dtype=complex),
    "ym90": np.array([[_s,   _s    ], [-_s,        _s]], dtype=complex),
    "z90":  np.array([[_s - 1j * _s, 0           ], [0,            _s + 1j * _s]], dtype=complex),
    "z180": np.array([[-1j,          0           ], [0,            1j          ]], dtype=complex),
    "zm90": np.array([[_s + 1j * _s, 0           ], [0,            _s - 1j * _s]], dtype=complex),
}


def _seq_unitary(seq: list[str]) -> np.ndarray:
    """2x2 unitary for a gate sequence applied left-to-right in time."""
    mat = np.eye(2, dtype=complex)
    for gate in seq:
        mat = _GATE_MATRICES[gate] @ mat
    return mat


def _equiv(U: np.ndarray, V: np.ndarray, tol: float = 1e-10) -> bool:
    """True if U and V are equal up to a global phase (U V† ∝ I)."""
    M = U @ V.conj().T
    return (
        abs(M[0, 1]) < tol
        and abs(M[1, 0]) < tol
        and abs(abs(M[0, 0]) - 1) < tol
        and abs(abs(M[1, 1]) - 1) < tol
    )


# ---------------------------------------------------------------------------
# Pre-compute reference unitaries once
# ---------------------------------------------------------------------------

_REF = [_seq_unitary(seq) for seq in _DECOMPOSITION_SEQUENCES]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAlternativeDecompositions:
    @pytest.mark.parametrize("k", range(NUM_CLIFFORDS))
    def test_equivalent_to_reference(self, k: int) -> None:
        alt = _seq_unitary(_ALTERNATIVE_DECOMPOSITION_SEQUENCES[k])
        assert _equiv(_REF[k], alt), (
            f"Clifford {k}: alt={_ALTERNATIVE_DECOMPOSITION_SEQUENCES[k]} "
            f"!= ref={_DECOMPOSITION_SEQUENCES[k]}"
        )

    def test_no_z_gates(self) -> None:
        z_gates = {"z90", "z180", "zm90"}
        for k, seq in enumerate(_ALTERNATIVE_DECOMPOSITION_SEQUENCES):
            used = z_gates & set(seq)
            assert not used, f"Clifford {k} alternative uses Z gates: {used}"


class TestCayleyTable:
    @pytest.mark.parametrize("i", range(NUM_CLIFFORDS))
    def test_row(self, i: int) -> None:
        for j in range(NUM_CLIFFORDS):
            k = CAYLEY[i][j]
            product = _REF[i] @ _REF[j]
            assert _equiv(product, _REF[k]), (
                f"CAYLEY[{i}][{j}] = {k} wrong: U(C_{i}) @ U(C_{j}) != U(C_{k})"
            )


class TestInverseTable:
    @pytest.mark.parametrize("i", range(NUM_CLIFFORDS))
    def test_inverse(self, i: int) -> None:
        j = INVERSES[i]
        product = _REF[i] @ _REF[j]
        assert _equiv(product, np.eye(2, dtype=complex)), (
            f"INVERSES[{i}] = {j} wrong: U(C_{i}) @ U(C_{j}) != I"
        )


# ---------------------------------------------------------------------------
# EPG formula verification
# ---------------------------------------------------------------------------
# The formula alpha_gate = alpha_RB^(1/<n_g>) rests on three steps:
#
#   Step 1. Depolarizing channels compose multiplicatively.
#           E_1 then E_2  =>  alpha_total = alpha_1 * alpha_2
#
#   Step 2. The RB decay rate is the Clifford-average of alpha_gate^{n_k}.
#           alpha_RB = (1/24) sum_k  alpha_gate^{n_k}
#
#   Step 3. For small errors, <alpha_gate^{n_k}> ~ alpha_gate^{<n_k>}.
#           The residual is  O( Var(n_k) * (log alpha_gate)^2 / 2 ).
# ---------------------------------------------------------------------------


def _depolarising(rho: np.ndarray, alpha: float) -> np.ndarray:
    """Single-qubit depolarising channel: alpha*rho + (1-alpha)*I/2."""
    return alpha * rho + (1.0 - alpha) * np.eye(2) / 2.0


def _exact_alpha_gate(gate_counts: list[int], alpha_rb: float) -> float:
    """Solve (1/N) sum_k x^{n_k} = alpha_rb for x in (0, 1] numerically."""
    n = len(gate_counts)

    def residual(x: float) -> float:
        return sum(x ** nk for nk in gate_counts) / n - alpha_rb

    if residual(1.0) < 0:
        return 1.0  # alpha_rb > 1, unphysical — return identity
    return brentq(residual, 1e-12, 1.0)


class TestEPGFormula:
    """Verify the derivation behind alpha_gate = alpha_RB^(1/<n_g>)."""

    def test_step1_depolarising_composes_multiplicatively(self) -> None:
        """E_alpha1 composed with E_alpha2 gives depolarising parameter alpha1*alpha2."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            # Random density matrix
            psi = rng.standard_normal(2) + 1j * rng.standard_normal(2)
            psi /= np.linalg.norm(psi)
            rho = np.outer(psi, psi.conj())

            alpha1, alpha2 = rng.uniform(0.9, 1.0, 2)
            composed = _depolarising(_depolarising(rho, alpha2), alpha1)
            direct = _depolarising(rho, alpha1 * alpha2)
            assert np.allclose(composed, direct, atol=1e-14), (
                f"Composition failed: alpha1={alpha1:.4f}, alpha2={alpha2:.4f}"
            )

    @pytest.mark.parametrize("sequences,name", [
        (_DECOMPOSITION_SEQUENCES, "reference"),
        (_ALTERNATIVE_DECOMPOSITION_SEQUENCES, "alternative"),
    ])
    def test_step2_rb_decay_is_clifford_average(
        self, sequences: list[list[str]], name: str
    ) -> None:
        """alpha_RB = (1/24) sum_k alpha_gate^{n_k} for a concrete alpha_gate."""
        gate_counts = [
            sum(1 for g in seq if g in _PHYSICAL_GATES) for seq in sequences
        ]
        alpha_gate = 0.999  # representative high-fidelity gate
        alpha_rb_exact = sum(alpha_gate ** nk for nk in gate_counts) / NUM_CLIFFORDS

        # Reconstruct alpha_gate via exact numerical inversion
        alpha_gate_recovered = _exact_alpha_gate(gate_counts, alpha_rb_exact)
        assert abs(alpha_gate_recovered - alpha_gate) < 1e-10, (
            f"[{name}] exact inversion failed: "
            f"in={alpha_gate:.6f}, recovered={alpha_gate_recovered:.6f}"
        )

    @pytest.mark.parametrize("epg", [1e-2, 1e-3, 1e-4])
    @pytest.mark.parametrize("sequences,name", [
        (_DECOMPOSITION_SEQUENCES, "reference"),
        (_ALTERNATIVE_DECOMPOSITION_SEQUENCES, "alternative"),
    ])
    def test_step3_approximation_error(
        self, sequences: list[list[str]], name: str, epg: float
    ) -> None:
        """alpha_gate ≈ alpha_RB^(1/<n_g>) to within O(Var(n_k)*(log alpha_gate)^2/2).

        The approximation should be negligible for realistic gate errors.
        """
        gate_counts = [
            sum(1 for g in seq if g in _PHYSICAL_GATES) for seq in sequences
        ]
        avg_ng = avg_physical_gates_per_clifford(sequences)
        var_ng = np.var(gate_counts)

        d = 2
        alpha_gate_true = 1.0 - epg * d / (d - 1)  # from epg = (1-alpha)*(d-1)/d
        alpha_rb = sum(alpha_gate_true ** nk for nk in gate_counts) / NUM_CLIFFORDS

        # Approximate inversion
        alpha_gate_approx = alpha_rb ** (1.0 / avg_ng)

        # Second-order bound on the log-space error
        log_alpha = np.log(alpha_gate_true)
        second_order_bound = abs(var_ng * log_alpha**2 / 2)

        log_error = abs(np.log(alpha_gate_approx) - np.log(alpha_gate_true))
        assert log_error < 2 * second_order_bound + 1e-12, (
            f"[{name}, epg={epg}] approximation error {log_error:.2e} exceeds "
            f"2 * second_order_bound {2*second_order_bound:.2e}"
        )

        # For realistic errors the absolute EPG error should be < 1% of EPG
        epg_approx = (1.0 - alpha_gate_approx) * (d - 1) / d
        epg_exact = (1.0 - _exact_alpha_gate(gate_counts, alpha_rb)) * (d - 1) / d
        relative_error = abs(epg_approx - epg_exact) / epg
        if epg <= 1e-2:
            assert relative_error < 0.01, (
                f"[{name}, epg={epg}] relative EPG error {relative_error:.2e} > 1%"
            )