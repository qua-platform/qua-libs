"""Single-qubit randomized benchmarking analysis — exponential decay fit.

The survival probability at circuit depth *m* follows

    F(m) = A · α^m + B

where α is the depolarising parameter.  The average error per Clifford is

    epc = (1 − α) · (d − 1) / d

with d = 2 for a single qubit, and the average Clifford gate fidelity is
F_avg = 1 − epc.

The fit uses ``scipy.optimize.curve_fit`` with physical bounds on A, α, B.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

import xarray as xr

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Fit result container
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class FitParameters:
    """Fitted parameters for a single qubit's RB decay."""

    alpha: float = 0.0
    """Depolarising parameter (decay constant per Clifford)."""
    A: float = 0.0
    """Amplitude of the exponential decay."""
    B: float = 0.0
    """Asymptotic offset (ideally 0.5 for single qubit)."""
    error_per_clifford: float = 0.0
    """Average error per Clifford gate."""
    gate_fidelity: float = 0.0
    """Average Clifford gate fidelity (1 − epc)."""
    success: bool = False
    """Whether the fit converged to a physically sensible result."""


# ──────────────────────────────────────────────────────────────────────────────
# Internal fitting
# ──────────────────────────────────────────────────────────────────────────────


def _rb_decay(m: np.ndarray, A: float, alpha: float, B: float) -> np.ndarray:
    """Exponential decay model: F(m) = A · α^m + B."""
    return A * alpha**m + B


def _fit_single_qubit(
    depths: np.ndarray,
    survival_prob: np.ndarray,
) -> dict[str, Any]:
    """Fit survival probability vs circuit depth to extract gate fidelity.

    Parameters
    ----------
    depths : 1-D array
        Circuit depths (number of Cliffords).
    survival_prob : 1-D array
        Average survival probability at each depth.

    Returns
    -------
    dict
        Keys: ``alpha``, ``A``, ``B``, ``error_per_clifford``,
        ``gate_fidelity``, ``fitted_curve``, ``success``.
    """
    depths = np.asarray(depths, dtype=np.float64)
    y = np.asarray(survival_prob, dtype=np.float64)

    if len(depths) < 3 or len(y) < 3:
        return dict(
            FitParameters().__dict__,
            fitted_curve=np.full_like(y, np.nan),
        )

    try:
        popt, _ = curve_fit(
            _rb_decay,
            depths,
            y,
            p0=[0.5, 0.99, 0.5],
            bounds=([0, 0, 0], [1, 1, 1]),
            maxfev=10_000,
        )
        A, alpha, B = popt
        success = bool(np.isfinite(alpha) and 0 < alpha <= 1)
    except (RuntimeError, ValueError):
        A, alpha, B = 0.5, 0.5, 0.5
        success = False

    d = 2  # single qubit
    epc = (1.0 - alpha) * (d - 1) / d
    fidelity = 1.0 - epc
    fitted_curve = _rb_decay(depths, A, alpha, B)

    return {
        "alpha": float(alpha),
        "A": float(A),
        "B": float(B),
        "error_per_clifford": float(epc),
        "gate_fidelity": float(fidelity),
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
    """Run the RB exponential-decay fit for every qubit.

    Parameters
    ----------
    ds_raw : xr.Dataset
        Raw dataset with coordinates ``depth`` and ``circuit``, and data
        variables ``state_<qubit_name>`` shaped ``[num_circuits, num_depths]``.
    qubits : list
        Qubit objects (each must have a ``.name`` attribute).

    Returns
    -------
    dict
        ``{qubit_name: {alpha, A, B, error_per_clifford, gate_fidelity,
        fitted_curve, success}}``.
    """
    depths = ds_raw.coords["depth"].values.astype(np.float64)
    results: dict[str, dict[str, Any]] = {}

    for qubit in qubits:
        qname = qubit.name
        var_name = f"state_{qname}"
        if var_name not in ds_raw.data_vars:
            logger.warning("No state variable for qubit %s — skipping.", qname)
            results[qname] = dict(
                FitParameters().__dict__,
                fitted_curve=np.array([]),
            )
            continue

        state_data = ds_raw[var_name].values  # [num_circuits, num_depths]
        survival_prob = np.mean(state_data, axis=0)  # average over circuits
        results[qname] = _fit_single_qubit(depths, survival_prob)

    return results


def log_fitted_results(
    fit_results: dict[str, dict[str, Any]],
    node_logger: Any | None = None,
) -> None:
    """Log fitted RB results for all qubits.

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
            f"  {qname}: [{status}] "
            f"F_cliff = {r['gate_fidelity']:.4f}, "
            f"epc = {r['error_per_clifford']:.5f}, "
            f"α = {r['alpha']:.5f}"
        )
        _log(msg)
