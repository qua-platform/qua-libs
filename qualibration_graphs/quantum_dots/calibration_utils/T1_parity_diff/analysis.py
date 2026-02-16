"""T₁ relaxation-time analysis from parity-difference decay.

This module fits an exponential decay to the parity-difference signal
measured after a π–idle–measure sequence:

.. math::

    P(\\tau) = \\mathrm{offset} + A\\,\\exp(-\\tau / T_1)

The excited-state population decays back to thermal equilibrium with
time constant T₁.

Fitting method
~~~~~~~~~~~~~~

A **profiled differential-evolution** search is used: DE optimises
over the single non-linear parameter T₁, while the two linear
parameters (offset, A) are solved exactly via ``np.linalg.lstsq``
at each candidate T₁.  This makes the fit a robust 1-D global search
that avoids local-minimum traps common with gradient-based methods.

Extracted quantities
--------------------
* **T1** — spin-lattice relaxation time (ns).
* **amplitude** — decay amplitude A (dimensionless parity units).
* **offset** — asymptotic baseline (τ → ∞).
* **decay_rate** — 1/T₁ (1/ns).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import differential_evolution

from qualibrate import QualibrationNode

_logger = logging.getLogger(__name__)


# ── Fit result dataclass ─────────────────────────────────────────────────────


@dataclass
class FitParameters:
    """Extracted parameters from a T₁ decay measurement.

    Attributes
    ----------
    T1 : float
        Relaxation time T₁ (ns).
    amplitude : float
        Decay amplitude A (parity-difference units).
    offset : float
        Asymptotic baseline offset.
    decay_rate : float
        Decay rate 1/T₁ (1/ns).
    success : bool
        ``True`` if the fit converged and T₁ is physical.
    """

    T1: float
    amplitude: float
    offset: float
    decay_rate: float
    success: bool


# ── Per-qubit exponential fit ────────────────────────────────────────────────


def _fit_single_qubit(
    pdiff: np.ndarray,
    tau_ns: np.ndarray,
) -> Dict[str, Any]:
    r"""Fit an exponential decay to a single qubit's T₁ data.

    Uses profiled DE: 1-D search over T₁ with linear solve for
    (offset, A) at each candidate.

    Parameters
    ----------
    pdiff : 1-D array (n_tau,)
        Parity-difference values.
    tau_ns : 1-D array (n_tau,)
        Idle-time values in nanoseconds.

    Returns
    -------
    dict
        ``T1`` (ns), ``amplitude``, ``offset``, ``decay_rate`` (1/ns),
        ``fitted_curve``, ``success``.
    """
    y = np.asarray(pdiff, dtype=float)
    t = np.asarray(tau_ns, dtype=float)
    n = len(t)
    t_span = float(t[-1] - t[0]) if n > 1 else 1.0

    result: Dict[str, Any] = {
        "T1": np.nan,
        "amplitude": 0.0,
        "offset": np.nan,
        "decay_rate": np.nan,
        "fitted_curve": np.full_like(y, np.nan),
        "success": False,
    }

    # DE bounds for T1: from a few time steps to several times the span
    t_step = float(np.min(np.diff(t))) if n > 1 else 1.0
    t1_lo = max(t_step, 1.0)
    t1_hi = max(10.0 * t_span, t1_lo * 10.0)

    def _profile_residual(t1_arr):
        t1 = t1_arr[0]
        basis = np.exp(-t / t1)
        A_mat = np.column_stack([np.ones(n), basis])
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
        return float(np.sum((y - A_mat @ coeffs) ** 2))

    try:
        de_result = differential_evolution(
            _profile_residual,
            bounds=[(t1_lo, t1_hi)],
            seed=42,
            maxiter=1000,
            tol=1e-12,
            polish=True,
            popsize=30,
        )
        t1_best = float(de_result.x[0])

        # Solve linear system at best T1
        basis = np.exp(-t / t1_best)
        A_mat = np.column_stack([np.ones(n), basis])
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
        offset, amplitude = float(coeffs[0]), float(coeffs[1])

        decay_rate = 1.0 / t1_best if t1_best > 0 else np.nan
        fitted_curve = A_mat @ coeffs

        result["T1"] = t1_best
        result["amplitude"] = amplitude
        result["offset"] = offset
        result["decay_rate"] = decay_rate
        result["fitted_curve"] = fitted_curve
        result["success"] = np.isfinite(t1_best) and t1_best > 0 and abs(amplitude) > 1e-6

        _logger.debug(
            "T1 fit: T1=%.1f ns, A=%.4f, offset=%.4f, γ=%.6f 1/ns",
            t1_best,
            amplitude,
            offset,
            decay_rate,
        )
    except Exception:
        _logger.debug("T1 profiled DE fit failed", exc_info=True)

    return result


# ── Public API ───────────────────────────────────────────────────────────────


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, Dict[str, Any]]]:
    """Fit T₁ exponential decay for each qubit.

    Expects a 1-D dataset with coordinate ``tau`` (ns) and data
    variables ``pdiff_<qubit>`` of shape ``(n_tau,)``.

    Parameters
    ----------
    ds : xr.Dataset
        Raw measurement data.
    node : QualibrationNode
        Calibration node (provides qubit list).

    Returns
    -------
    (ds_fit, fit_results) : tuple
        *ds_fit* is a copy of the input dataset.  *fit_results* maps
        qubit name → dict of :class:`FitParameters` fields plus
        ``_diag`` with raw solver outputs for plotting.
    """
    qubits = node.namespace["qubits"]
    tau_ns = np.asarray(ds.tau.values, dtype=float)

    pdiff_vars = [v for v in ds.data_vars if v.startswith("pdiff_")]
    qubit_names = [v.replace("pdiff_", "") for v in sorted(pdiff_vars)]
    if not qubit_names:
        qubit_names = [getattr(q, "name", f"Q{i}") for i, q in enumerate(qubits)]

    fit_results: Dict[str, Dict[str, Any]] = {}

    for qname in qubit_names:
        pdiff_var = f"pdiff_{qname}"
        if pdiff_var not in ds.data_vars:
            fp = FitParameters(
                T1=np.nan,
                amplitude=0.0,
                offset=np.nan,
                decay_rate=np.nan,
                success=False,
            )
            fit_results[qname] = asdict(fp)
            continue

        pdiff = np.asarray(ds[pdiff_var].values, dtype=float)
        raw = _fit_single_qubit(pdiff, tau_ns)

        fp = FitParameters(
            T1=raw["T1"],
            amplitude=raw["amplitude"],
            offset=raw["offset"],
            decay_rate=raw["decay_rate"],
            success=raw["success"],
        )
        fit_results[qname] = asdict(fp)
        fit_results[qname]["_diag"] = raw

    ds_fit = ds.copy()
    return ds_fit, fit_results


def log_fitted_results(
    fit_results: Dict[str, Any],
    log_callable=None,
) -> None:
    """Log fitted T₁ parameters for all qubits.

    Parameters
    ----------
    fit_results : dict
        Qubit name → fit-result dict (as returned by :func:`fit_raw_data`).
    log_callable : callable, optional
        Logging function; defaults to ``_logger.info``.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for qname, r in fit_results.items():
        t1 = r.get("T1", np.nan)
        amp = r.get("amplitude", 0)
        offset = r.get("offset", np.nan)
        gamma = r.get("decay_rate", np.nan)
        success = r.get("success", False)
        msg = (
            f"Results for {qname}: "
            f"T1={t1:.1f} ns, "
            f"A={amp:.4f}, "
            f"offset={offset:.4f}, "
            f"γ={gamma:.6f} 1/ns, "
            f"success={success}"
        )
        log_callable(msg)
