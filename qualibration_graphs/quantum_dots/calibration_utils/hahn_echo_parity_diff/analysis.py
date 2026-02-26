"""Hahn echo (spin echo) T₂ analysis — profiled differential evolution.

The Hahn echo sequence is  π/2 – τ – π – τ – π/2,  where τ is the per-arm
idle time.  The echo amplitude decays as

    P(τ) = offset + A · exp(−2τ / T₂_echo)

The factor of 2 accounts for the total evolution time 2τ.  Unlike T₂* from a
Ramsey experiment, T₂_echo is insensitive to static (low-frequency) dephasing
because the π pulse refocuses it.  T₂_echo therefore measures irreversible
decoherence from high-frequency noise sources.

**Fitting strategy — profiled differential evolution (DE)**

The model has three parameters: ``offset``, ``A`` (amplitude), and
``T2_echo``.  Only ``T2_echo`` enters non-linearly.  For each candidate
``T2_echo`` proposed by DE, the linear parameters ``[offset, A]`` are solved
analytically via ``np.linalg.lstsq``.  This reduces the search from 3-D to
1-D, dramatically improving convergence speed and robustness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from scipy.optimize import differential_evolution

import xarray as xr

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Fit result container
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class FitParameters:
    """Fitted parameters for a single qubit's Hahn echo decay."""

    T2_echo: float = 0.0
    """Hahn echo coherence time in nanoseconds."""
    amplitude: float = 0.0
    """Contrast (amplitude of the exponential decay)."""
    offset: float = 0.0
    """Baseline / background level."""
    decay_rate: float = 0.0
    """Effective decay rate γ = 2 / T₂_echo (1/ns)."""
    success: bool = False
    """Whether the fit converged to a physically sensible result."""


# ──────────────────────────────────────────────────────────────────────────────
# Internal fitting
# ──────────────────────────────────────────────────────────────────────────────


def _fit_single_qubit(
    tau_ns: np.ndarray,
    pdiff: np.ndarray,
) -> dict[str, Any]:
    """Fit P(τ) = offset + A·exp(−2τ / T₂_echo) using profiled DE.

    Parameters
    ----------
    tau_ns : 1-D array
        Per-arm idle times in nanoseconds.
    pdiff : 1-D array
        Parity-difference signal (same length as *tau_ns*).

    Returns
    -------
    dict
        Keys: ``T2_echo``, ``amplitude``, ``offset``, ``decay_rate``,
        ``fitted_curve``, ``success``.
    """
    tau = tau_ns.astype(np.float64)
    y = pdiff.astype(np.float64)

    t_span = float(tau.max() - tau.min())
    if t_span <= 0 or len(tau) < 4:
        return dict(FitParameters().__dict__, fitted_curve=np.full_like(y, np.nan))

    # DE bounds for T2_echo: [tau_step .. 10× sweep range]
    t2_lo = max(float(np.min(np.diff(np.sort(tau)))), 1.0)
    t2_hi = 10.0 * t_span

    def _cost(params: np.ndarray) -> float:
        (t2,) = params
        basis = np.column_stack([np.ones_like(tau), np.exp(-2.0 * tau / t2)])
        coeffs, *_ = np.linalg.lstsq(basis, y, rcond=None)
        residuals = y - basis @ coeffs
        return float(np.sum(residuals**2))

    result = differential_evolution(
        _cost,
        bounds=[(t2_lo, t2_hi)],
        seed=42,
        tol=1e-10,
        atol=1e-10,
        maxiter=2000,
        polish=True,
    )

    t2_best = float(result.x[0])

    # Recover linear parameters at the optimum
    basis = np.column_stack([np.ones_like(tau), np.exp(-2.0 * tau / t2_best)])
    coeffs, *_ = np.linalg.lstsq(basis, y, rcond=None)
    offset_best = float(coeffs[0])
    amp_best = float(coeffs[1])
    fitted_curve = basis @ coeffs

    # Sanity: T2 should be positive and finite, amplitude should be non-zero
    success = bool(
        result.success
        and np.isfinite(t2_best)
        and t2_best > 0
        and np.isfinite(amp_best)
        and abs(amp_best) > 1e-6
    )

    decay_rate = 2.0 / t2_best if t2_best > 0 else 0.0

    return {
        "T2_echo": t2_best,
        "amplitude": amp_best,
        "offset": offset_best,
        "decay_rate": decay_rate,
        "fitted_curve": fitted_curve,
        "success": success,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def fit_raw_data(
    ds_raw: xr.Dataset,
    qubits: list[Any],
) -> dict[str, dict[str, Any]]:
    """Run the Hahn echo exponential-decay fit for every qubit.

    Parameters
    ----------
    ds_raw : xr.Dataset
        Raw dataset with coordinates ``tau`` (ns) and data variables
        ``pdiff_<qubit_name>``.
    qubits : list
        Qubit objects (each must have a ``.name`` attribute).

    Returns
    -------
    dict
        ``{qubit_name: {T2_echo, amplitude, offset, decay_rate,
        fitted_curve, success}}``.
    """
    tau_ns = ds_raw.coords["tau"].values.astype(np.float64)
    results: Dict[str, dict[str, Any]] = {}

    for qubit in qubits:
        qname = qubit.name
        var_name = f"pdiff_{qname}"
        if var_name not in ds_raw.data_vars:
            logger.warning("No pdiff variable for qubit %s — skipping.", qname)
            results[qname] = dict(FitParameters().__dict__, fitted_curve=np.array([]))
            continue

        pdiff = ds_raw[var_name].values.astype(np.float64)
        results[qname] = _fit_single_qubit(tau_ns, pdiff)

    return results


def log_fitted_results(
    fit_results: dict[str, dict[str, Any]],
    node_logger: Any | None = None,
) -> None:
    """Log fitted Hahn echo results for all qubits.

    Parameters
    ----------
    fit_results : dict
        Output of :func:`fit_raw_data`.
    node_logger : callable, optional
        Logging function (e.g. ``node.log``).  Falls back to module logger.
    """
    _log = node_logger or logger.info
    for qname, r in sorted(fit_results.items()):
        status = "OK" if r["success"] else "FAILED"
        msg = (
            f"  {qname}: [{status}] T2_echo={r['T2_echo']:.1f} ns, "
            f"A={r['amplitude']:.4f}, γ={r['decay_rate']:.6f} 1/ns"
        )
        _log(msg)
