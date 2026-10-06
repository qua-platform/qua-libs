"""Shared state fidelity, purity, and metric validation helpers."""

import numpy as np
from scipy.linalg import sqrtm


def purity(rho: np.ndarray) -> float:
    """Return Tr(rho^2) for a density matrix."""
    return float(np.abs(np.trace(rho @ rho)))


def fidelity_with_pure_state(rho: np.ndarray, psi: np.ndarray) -> float:
    """Return fidelity with a pure target state, F = <psi|rho|psi>."""
    fidelity = np.real(np.vdot(psi, rho @ psi))
    return float(np.clip(fidelity, 0.0, 1.0))


def fidelity_with_state(rho: np.ndarray, rho_target: np.ndarray) -> float:
    """Return fidelity with a target density matrix using the Uhlmann formula."""
    s_target = sqrtm(rho_target)
    fidelity = float(np.abs(np.trace(sqrtm(s_target @ rho @ s_target))) ** 2)
    return float(np.clip(fidelity, 0.0, 1.0))


def z_basis_ghz_fidelity(probs: np.ndarray) -> float:
    """Return GHZ Z-basis population fidelity, P(|0...0>) + P(|1...1>)."""
    probs = np.asarray(probs, dtype=float)
    return float(probs[0] + probs[-1])


def is_valid_probability_vector(probs: np.ndarray, sum_tol: float = 1e-3) -> bool:
    """Return True if ``probs`` is finite, non-negative, and sums to one."""
    probs = np.asarray(probs, dtype=float)
    if probs.size == 0 or not np.all(np.isfinite(probs)):
        return False
    if np.any(probs < -sum_tol):
        return False
    return bool(abs(float(probs.sum()) - 1.0) <= sum_tol)


def is_valid_unit_metric(value: float, tol: float = 1e-6) -> bool:
    """Return True if ``value`` is finite and lies in ``[0, 1]`` (within tolerance)."""
    if not np.isfinite(value):
        return False
    return bool(-tol <= value <= 1.0 + tol)


def is_normalized_density_matrix(rho: np.ndarray, trace_tol: float = 1e-3) -> bool:
    """Return True if ``rho`` has finite trace close to one."""
    trace = np.trace(rho)
    return bool(np.isfinite(trace) and abs(float(np.real(trace)) - 1.0) <= trace_tol)


def are_tomography_metrics_valid(
    fidelity: float,
    purity: float,
    rho: np.ndarray,
    trace_tol: float = 1e-3,
) -> bool:
    """Return True when tomography fidelity, purity, and trace are physically valid."""
    if not is_valid_unit_metric(fidelity) or not is_valid_unit_metric(purity):
        return False
    return is_normalized_density_matrix(rho, trace_tol=trace_tol)
