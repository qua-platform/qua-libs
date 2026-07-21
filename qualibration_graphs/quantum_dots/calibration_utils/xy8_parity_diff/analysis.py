"""XY8 dynamical decoupling T₂ analysis — profiled differential evolution.

The XY8 sequence uses CPMG timing with 8 alternating X/Y refocusing pi pulses:

    π/2 – τ – X – 2τ – Y – 2τ – X – 2τ – Y – 2τ – Y – 2τ – X – 2τ – Y – 2τ – X – τ – π/2

where τ is the half inter-pulse spacing.  Total idle time = 16τ.
The signal decays as

    P(τ) = offset + A · exp(−16τ / T₂_XY8)

T₂_XY8 is the physical XY8 coherence time — directly comparable to T₂_echo
from the Hahn echo node (which uses exp(−2τ / T₂_echo) with 2 idle segments).

**Fitting strategy — profiled differential evolution (DE)**

The model has three parameters: ``offset``, ``A`` (amplitude), and
``T2_xy8``.  Only ``T2_xy8`` enters non-linearly.  For each candidate
``T2_xy8`` proposed by DE, the linear parameters ``[offset, A]`` are solved
analytically via ``np.linalg.lstsq``.  This reduces the search from 3-D to
1-D, dramatically improving convergence speed and robustness.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import numpy as np
from scipy.optimize import differential_evolution

import xarray as xr
from qualibrate.core import QualibrationNode
from calibration_utils.common_utils.parity_streams import get_parity_item_names

logger = logging.getLogger(__name__)

IDLE_FACTOR = 16.0


@dataclass
class FitParameters:
    """Fitted parameters for a single qubit's XY8 decay."""

    T2_xy8: float = 0.0
    """Physical XY8 coherence time in nanoseconds."""
    amplitude: float = 0.0
    """Contrast (amplitude of the exponential decay)."""
    offset: float = 0.0
    """Baseline / background level."""
    decay_rate: float = 0.0
    """Effective decay rate γ = 16 / T₂_XY8 (1/ns)."""
    success: bool = False
    """Whether the fit converged to a physically sensible result."""


def _fit_single_qubit(
    tau_ns: np.ndarray,
    y_signal: np.ndarray,
) -> dict[str, Any]:
    """Fit P(τ) = offset + A·exp(−16τ / T₂_XY8) using profiled DE.

    Parameters
    ----------
    tau_ns : 1-D array
        Half inter-pulse spacings in nanoseconds.
    y_signal : 1-D array
        Conditional readout signal (same length as *tau_ns*).

    Returns
    -------
    dict
        Keys: ``T2_xy8``, ``amplitude``, ``offset``, ``decay_rate``,
        ``fitted_curve``, ``signal``, ``success``.
    """
    tau_all = tau_ns.astype(np.float64)
    y_all = y_signal.astype(np.float64)

    mask = np.isfinite(y_all) & np.isfinite(tau_all)
    tau = tau_all[mask]
    y = y_all[mask]

    result: dict[str, Any] = {
        "T2_xy8": np.nan,
        "amplitude": 0.0,
        "offset": np.nan,
        "decay_rate": np.nan,
        "fitted_curve": np.full_like(y_all, np.nan),
        "signal": y_all.copy(),
        "success": False,
    }

    t_span = float(tau.max() - tau.min()) if len(tau) > 1 else 0.0
    if t_span <= 0 or len(tau) < 4:
        return result

    t2_lo = max(float(np.min(np.diff(np.sort(tau)))), 1.0)
    t2_hi = 10.0 * IDLE_FACTOR * t_span

    def _cost(params: np.ndarray) -> float:
        (t2,) = params
        exponent = np.clip(-IDLE_FACTOR * tau / t2, -700, 0)
        basis = np.column_stack([np.ones_like(tau), np.exp(exponent)])
        coeffs, *_ = np.linalg.lstsq(basis, y, rcond=None)
        ss = float(np.sum((y - basis @ coeffs) ** 2))
        return ss if np.isfinite(ss) else 1e30

    try:
        de_result = differential_evolution(
            _cost,
            bounds=[(t2_lo, t2_hi)],
            seed=42,
            tol=1e-10,
            atol=1e-10,
            maxiter=2000,
            polish=True,
        )
        t2_best = float(de_result.x[0])

        exponent = np.clip(-IDLE_FACTOR * tau / t2_best, -700, 0)
        basis = np.column_stack([np.ones_like(tau), np.exp(exponent)])
        coeffs, *_ = np.linalg.lstsq(basis, y, rcond=None)
        offset_best = float(coeffs[0])
        amp_best = float(coeffs[1])

        result["T2_xy8"] = t2_best
        result["amplitude"] = amp_best
        result["offset"] = offset_best
        result["decay_rate"] = IDLE_FACTOR / t2_best if t2_best > 0 else 0.0
        result["fitted_curve"] = basis @ coeffs
        result["success"] = bool(
            de_result.success
            and np.isfinite(t2_best)
            and t2_best > 0
            and np.isfinite(amp_best)
            and abs(amp_best) > 1e-6
        )
    except Exception:
        logger.warning("XY8 T2 fit failed", exc_info=True)

    return result


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, Dict[str, Any]]]:
    """Run the XY8 exponential-decay fit for every qubit.

    Expects joint-outcome streams processed by
    :func:`~calibration_utils.common_utils.parity_streams.process_joint_streams`,
    so the analysis uses ``{analysis_signal}_{qubit}`` (default
    ``E_p2_given_p1_0_<qubit>``) of shape ``(n_tau,)`` with coordinate
    ``tau`` (half inter-pulse spacing in ns).

    Parameters
    ----------
    ds : xr.Dataset
        Raw measurement data (after joint-stream processing).
    node : QualibrationNode
        Calibration node (provides qubit list and ``analysis_signal``).

    Returns
    -------
    (ds_fit, fit_results) : tuple
        *ds_fit* is a copy of the input dataset.  *fit_results* maps
        qubit name → dict of :class:`FitParameters` fields plus ``_diag``
        with per-trace data for plotting.
    """
    qubits = node.namespace["qubits"]
    tau_ns = np.asarray(ds.tau.values, dtype=float)

    analysis_signal = getattr(node.parameters, "analysis_signal", "E_p2_given_p1_0")
    qubit_names = get_parity_item_names(
        ds,
        analysis_signal,
        item_names=[getattr(q, "name", f"Q{i}") for i, q in enumerate(qubits)],
    )

    fit_results: Dict[str, Dict[str, Any]] = {}

    for qname in qubit_names:
        signal_var = f"{analysis_signal}_{qname}"
        if signal_var not in ds.data_vars:
            logger.warning("No analysis signal for qubit %s — skipping.", qname)
            fp = FitParameters(
                T2_xy8=float("nan"),
                amplitude=0.0,
                offset=float("nan"),
                decay_rate=float("nan"),
                success=False,
            )
            fit_results[qname] = asdict(fp)
            continue

        signal_1d = np.asarray(ds[signal_var].values, dtype=float)
        if signal_1d.ndim != 1:
            logger.warning(
                "Expected 1-D shape (n_tau,) for %s, got %s — skipping.",
                signal_var,
                getattr(signal_1d, "shape", None),
            )
            fp = FitParameters(
                T2_xy8=float("nan"),
                amplitude=0.0,
                offset=float("nan"),
                decay_rate=float("nan"),
                success=False,
            )
            fit_results[qname] = asdict(fp)
            continue

        result = _fit_single_qubit(tau_ns, signal_1d)
        fp = FitParameters(
            T2_xy8=result["T2_xy8"],
            amplitude=result["amplitude"],
            offset=result["offset"],
            decay_rate=result["decay_rate"],
            success=result["success"],
        )
        fit_results[qname] = asdict(fp)
        fit_results[qname]["_diag"] = result

    ds_fit = ds.copy()
    return ds_fit, fit_results


def log_fitted_results(
    fit_results: dict[str, dict[str, Any]],
    log_callable: Any | None = None,
) -> None:
    """Log fitted XY8 results for all qubits.

    Parameters
    ----------
    fit_results : dict
        Output of :func:`fit_raw_data`.
    log_callable : callable, optional
        Logging function (e.g. ``node.log``).  Falls back to module logger.
    """
    _log = log_callable or logger.info
    for qname, r in sorted(fit_results.items()):
        status = "OK" if r["success"] else "FAILED"
        msg = (
            f"  {qname}: [{status}] T2_xy8={r['T2_xy8']:.1f} ns, "
            f"A={r['amplitude']:.4f}, γ={r['decay_rate']:.6f} 1/ns"
        )
        _log(msg)
