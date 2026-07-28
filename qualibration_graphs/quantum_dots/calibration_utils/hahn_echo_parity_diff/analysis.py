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
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import numpy as np
from scipy.optimize import differential_evolution

import xarray as xr
from qualibrate.core import QualibrationNode
from calibration_utils.measurement_utils import get_parity_item_names, process_streams

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
    y_signal: np.ndarray,
) -> dict[str, Any]:
    """Fit P(τ) = offset + A·exp(−2τ / T₂_echo) using profiled DE.

    Parameters
    ----------
    tau_ns : 1-D array
        Per-arm idle times in nanoseconds.
    y_signal : 1-D array
        Conditional readout signal (same length as *tau_ns*).

    Returns
    -------
    dict
        Keys: ``T2_echo``, ``amplitude``, ``offset``, ``decay_rate``,
        ``fitted_curve``, ``signal``, ``success``.
    """
    tau_all = tau_ns.astype(np.float64)
    y_all = y_signal.astype(np.float64)

    mask = np.isfinite(y_all) & np.isfinite(tau_all)
    tau = tau_all[mask]
    y = y_all[mask]

    result: dict[str, Any] = {
        "T2_echo": np.nan,
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
    t2_hi = 10.0 * t_span

    def _cost(params: np.ndarray) -> float:
        (t2,) = params
        exponent = np.clip(-2.0 * tau / t2, -700, 0)
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

        exponent = np.clip(-2.0 * tau / t2_best, -700, 0)
        basis = np.column_stack([np.ones_like(tau), np.exp(exponent)])
        coeffs, *_ = np.linalg.lstsq(basis, y, rcond=None)
        offset_best = float(coeffs[0])
        amp_best = float(coeffs[1])

        result["T2_echo"] = t2_best
        result["amplitude"] = amp_best
        result["offset"] = offset_best
        result["decay_rate"] = 2.0 / t2_best if t2_best > 0 else 0.0
        result["fitted_curve"] = basis @ coeffs
        result["success"] = bool(
            de_result.success
            and np.isfinite(t2_best)
            and t2_best > 0
            and np.isfinite(amp_best)
            and abs(amp_best) > 1e-6
        )
    except Exception:
        logger.warning("Hahn echo T2 fit failed", exc_info=True)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Compute conditional expectations from joint-outcome streams.

    Returns a new dataset; ``ds_raw`` in the node is left unchanged.
    """
    qubits = node.namespace["qubits"]
    return process_streams(
        ds,
        [q.name for q in qubits],
        parity_measurement=node.parameters.parity_measurement,
        sweep_dims=("tau",),
    )


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, Dict[str, Any]]]:
    """Run the Hahn echo exponential-decay fit for every qubit.

    Expects data processed by :func:`process_raw_dataset`, which adds
    ``{analysis_signal}_{qubit}`` variables (default
    ``E_p1_given_p0_0_<qubit>``) of shape ``(n_tau,)`` with coordinate
    ``tau`` (per-arm idle time in ns).

    Parameters
    ----------
    ds : xr.Dataset
        Raw measurement data (after joint-stream processing).
    node : QualibrationNode
        Calibration node (provides qubit list and ``analysis_signal``).

    Returns
    -------
    (ds_fit, fit_results) : tuple
        *ds_fit* contains processed streams, per-qubit fitted curves
        (``{analysis_signal}_fit_{qubit}``), and summary scalars on a
        ``qubit`` coordinate.  *fit_results* maps qubit name →
        :class:`FitParameters` fields as plain dicts.
    """
    qubits = node.namespace["qubits"]
    tau_ns = np.asarray(ds.tau.values, dtype=float)

    analysis_signal = node.parameters.analysis_signal
    qubit_names = get_parity_item_names(
        ds,
        analysis_signal,
        item_names=[getattr(q, "name", f"Q{i}") for i, q in enumerate(qubits)],
    )

    fit_results: Dict[str, Dict[str, Any]] = {}
    fit_curve_vars: Dict[str, Tuple[list[str], np.ndarray]] = {}

    for qname in qubit_names:
        signal_var = f"{analysis_signal}_{qname}"
        if signal_var not in ds.data_vars:
            logger.warning("No analysis signal for qubit %s — skipping.", qname)
            fp = FitParameters(
                T2_echo=float("nan"),
                amplitude=0.0,
                offset=float("nan"),
                decay_rate=float("nan"),
                success=False,
            )
            fit_results[qname] = asdict(fp)
            fit_curve_vars[f"{analysis_signal}_fit_{qname}"] = (
                ["tau"],
                np.full_like(tau_ns, np.nan, dtype=float),
            )
            continue

        signal_1d = np.asarray(ds[signal_var].values, dtype=float)
        if signal_1d.ndim != 1:
            logger.warning(
                "Expected 1-D shape (n_tau,) for %s, got %s — skipping.",
                signal_var,
                getattr(signal_1d, "shape", None),
            )
            fp = FitParameters(
                T2_echo=float("nan"),
                amplitude=0.0,
                offset=float("nan"),
                decay_rate=float("nan"),
                success=False,
            )
            fit_results[qname] = asdict(fp)
            fit_curve_vars[f"{analysis_signal}_fit_{qname}"] = (
                ["tau"],
                np.full_like(tau_ns, np.nan, dtype=float),
            )
            continue

        result = _fit_single_qubit(tau_ns, signal_1d)
        fp = FitParameters(
            T2_echo=result["T2_echo"],
            amplitude=result["amplitude"],
            offset=result["offset"],
            decay_rate=result["decay_rate"],
            success=result["success"],
        )
        fit_results[qname] = asdict(fp)
        fit_curve_vars[f"{analysis_signal}_fit_{qname}"] = (
            ["tau"],
            np.asarray(result["fitted_curve"], dtype=float),
        )

    ds_fit = ds.assign(
        {
            name: xr.DataArray(
                data,
                dims=dims,
                coords={"tau": ds.tau},
                attrs={"long_name": "fitted echo decay"},
            )
            for name, (dims, data) in fit_curve_vars.items()
        }
    )
    ds_fit = ds_fit.assign(
        T2_echo=("qubit", [fit_results[q]["T2_echo"] for q in qubit_names]),
        amplitude=("qubit", [fit_results[q]["amplitude"] for q in qubit_names]),
        offset=("qubit", [fit_results[q]["offset"] for q in qubit_names]),
        decay_rate=("qubit", [fit_results[q]["decay_rate"] for q in qubit_names]),
        success=("qubit", [fit_results[q]["success"] for q in qubit_names]),
    ).assign_coords(qubit=qubit_names)
    ds_fit["T2_echo"].attrs = {"long_name": "Hahn echo T2", "units": "ns"}
    ds_fit["amplitude"].attrs = {"long_name": "echo contrast"}
    ds_fit["offset"].attrs = {"long_name": "baseline"}
    ds_fit["decay_rate"].attrs = {"long_name": "decay rate", "units": "1/ns"}

    return ds_fit, fit_results


def log_fitted_results(
    fit_results: dict[str, dict[str, Any]],
    log_callable: Any | None = None,
) -> None:
    """Log fitted Hahn echo results for all qubits.

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
            f"  {qname}: [{status}] T2_echo={r['T2_echo']:.1f} ns, "
            f"A={r['amplitude']:.4f}, γ={r['decay_rate']:.6f} 1/ns"
        )
        _log(msg)
