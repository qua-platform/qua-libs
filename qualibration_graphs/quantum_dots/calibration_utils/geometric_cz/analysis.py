"""Geometric CZ gate calibration analysis — quadrature readout.

Extracts the conditional phase from 2D sweeps of exchange amplitude vs
duration using I/Q quadrature readout, then finds the CZ operating point
(V*, t*) where the conditional phase reaches pi at a user-specified target
duration.

Phase extraction
----------------
For each amplitude slice the four measurements (2 control states x 2
quadrature axes) give I/Q pairs for each branch:

    phi0(t) = arctan2(Q0 - c, I0 - c)   (control |0>)
    phi1(t) = arctan2(Q1 - c, I1 - c)   (control |1>)

The parity readout contributes a known pi offset between branches.
Both branches also contain a common-mode Stark shift from the barrier gate
voltage.  The conditional phase is the *difference* minus the parity offset:

    cond_phase(V, t) = phi1(V, t) - phi0(V, t) - pi

The Stark shift cancels in the subtraction.  The conditional phase starts
near 0 at zero exchange time and crosses pi at the CZ operating point.

CZ point selection
------------------
The user specifies a target duration t*.  The analysis interpolates
cond_phase(V, t*) vs V and finds the first amplitude where it crosses pi.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

_QUA_DURATION_GRANULARITY_NS = 4.0


@dataclass
class GeometricCZFitResult:
    """Aggregated fit result for one qubit pair."""

    optimal_amplitude: float = float("nan")
    """Barrier gate voltage at the CZ operating point (V)."""
    optimal_duration: float = float("nan")
    """Exchange pulse duration at the CZ operating point (ns)."""
    t_pi_cphase: list = None
    """Duration where conditional phase reaches pi at each amplitude (ns)."""
    conditional_phase_rate: float = 0.0
    """Mean conditional-phase rate at the optimal amplitude (rad/ns)."""
    success: bool = False

    def __post_init__(self):
        if self.t_pi_cphase is None:
            self.t_pi_cphase = []


def _quantize_duration(duration_ns: float) -> float:
    """Round a duration to the nearest multiple of the QUA clock granularity."""
    if not np.isfinite(duration_ns):
        return duration_ns
    return round(duration_ns / _QUA_DURATION_GRANULARITY_NS) * _QUA_DURATION_GRANULARITY_NS


def _first_crossing(
    x: np.ndarray,
    y: np.ndarray,
    target: float,
) -> tuple[float, bool]:
    """First *x* where *y* crosses *target* (linear interpolation)."""
    for i in range(len(x) - 1):
        y0, y1 = float(y[i]), float(y[i + 1])
        if not np.isfinite(y0) or not np.isfinite(y1):
            continue
        if (y0 <= target <= y1) or (y1 <= target <= y0):
            denom = y1 - y0
            if abs(denom) < 1e-15:
                return float(x[i]), True
            frac = (target - y0) / denom
            x_cross = float(x[i]) + frac * (float(x[i + 1]) - float(x[i]))
            return x_cross, True
    return np.nan, False


def _extract_conditional_phase(
    data: xr.DataArray,
    durations: np.ndarray,
    center: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-amplitude-slice phase traces and t_pi curve.

    Returns
    -------
    cond_phase : ndarray, shape (n_amps, n_durs)
        Conditional phase phi1 - phi0 - pi.
    phi0_all : ndarray, shape (n_amps, n_durs)
        Unwrapped phase of control-|0> branch.
    phi1_all : ndarray, shape (n_amps, n_durs)
        Unwrapped phase of control-|1> branch.
    t_pi : ndarray, shape (n_amps,)
        Duration where conditional phase first crosses pi (NaN if none).
    """
    i0_2d = data.sel(control_state=0, analysis_axis=0).values.astype(np.float64)
    q0_2d = data.sel(control_state=0, analysis_axis=1).values.astype(np.float64)
    i1_2d = data.sel(control_state=1, analysis_axis=0).values.astype(np.float64)
    q1_2d = data.sel(control_state=1, analysis_axis=1).values.astype(np.float64)

    n_amps, n_durs = i0_2d.shape
    cond_phase = np.full((n_amps, n_durs), np.nan)
    phi0_all = np.full((n_amps, n_durs), np.nan)
    phi1_all = np.full((n_amps, n_durs), np.nan)
    t_pi = np.full(n_amps, np.nan)

    for i in range(n_amps):
        if np.any(~np.isfinite(i0_2d[i])) or np.any(~np.isfinite(i1_2d[i])):
            continue

        phi0 = np.unwrap(np.arctan2(q0_2d[i] - center, i0_2d[i] - center))
        phi1 = np.unwrap(np.arctan2(q1_2d[i] - center, i1_2d[i] - center))

        # Conditional phase via complex conjugate product.
        # angle(z1·z0*) = φ₁−φ₀ ≈ π (parity offset) at zero exchange.
        # Negating shifts the branch cut away from the initial value:
        # angle(−z1·z0*) = φ₁−φ₀−π, starting near 0 → safe for unwrap.
        z0 = (i0_2d[i] - center) + 1j * (q0_2d[i] - center)
        z1 = (i1_2d[i] - center) + 1j * (q1_2d[i] - center)
        cp = np.unwrap(np.angle(-z1 * np.conj(z0)))

        phi0_all[i] = phi0
        phi1_all[i] = phi1
        cond_phase[i] = cp

        t_cross, found = _first_crossing(durations, cp, np.pi)
        if found:
            t_pi[i] = t_cross

    return cond_phase, phi0_all, phi1_all, t_pi


def fit_raw_data(
    ds_raw: xr.Dataset,
    qubit_pairs: list[Any],
    *,
    target_duration_ns: float,
    analysis_signal: str = "E_p2_given_p1_0",
    quadrature_signal_center: float = 0.5,
) -> tuple[xr.Dataset, dict[str, dict[str, Any]]]:
    """Run geometric CZ analysis for every qubit pair.

    Parameters
    ----------
    ds_raw : xr.Dataset
        Dataset with coordinates ``(control_state, analysis_axis,
        exchange_amplitude, exchange_duration)`` and variables
        ``{analysis_signal}_{pair_name}``.
    qubit_pairs : list
        Qubit pair objects (each must have a ``.name`` attribute).
    target_duration_ns : float
        Target exchange duration (ns).  The analysis finds the amplitude
        where the conditional phase reaches pi at this duration.
    analysis_signal : str
        Which conditional expectation to fit.
    quadrature_signal_center : float
        Expected centre of the I/Q signal.

    Returns
    -------
    ds_fit : xr.Dataset
        Fitted quantities vs exchange_amplitude per pair.
    fit_results : dict
        Per-pair results with optimal_amplitude, optimal_duration, success.
    """
    durations = ds_raw.coords["exchange_duration"].values.astype(np.float64)
    amplitudes = ds_raw.coords["exchange_amplitude"].values.astype(np.float64)
    fit_results: Dict[str, dict[str, Any]] = {}
    fit_vars: dict[str, xr.DataArray] = {}

    target_dur = _quantize_duration(float(target_duration_ns))

    for qp in qubit_pairs:
        var_name = f"{analysis_signal}_{qp.name}"
        if var_name not in ds_raw.data_vars:
            logger.warning("No %s variable for pair %s — skipping.", var_name, qp.name)
            fit_results[qp.name] = asdict(GeometricCZFitResult())
            continue

        data = ds_raw[var_name]
        center = float(quadrature_signal_center)

        cond_phase, phi0_all, phi1_all, t_pi = _extract_conditional_phase(
            data, durations, center,
        )

        # --- Find V* at the target duration ---
        t_idx = int(np.clip(np.searchsorted(durations, target_dur), 0, len(durations) - 1))
        if abs(durations[t_idx] - target_dur) > _QUA_DURATION_GRANULARITY_NS:
            t_idx_alt = max(0, t_idx - 1)
            if abs(durations[t_idx_alt] - target_dur) < abs(durations[t_idx] - target_dur):
                t_idx = t_idx_alt

        cond_at_target = cond_phase[:, t_idx]
        v_opt, root_ok = _first_crossing(amplitudes, cond_at_target, np.pi)

        success = bool(root_ok and np.isfinite(v_opt))
        t_opt = target_dur

        cond_rate = 0.0
        if success:
            amp_idx = int(np.clip(np.searchsorted(amplitudes, v_opt), 0, len(amplitudes) - 1))
            trace = cond_phase[amp_idx]
            if np.sum(np.isfinite(trace)) >= 2:
                cond_rate = float(
                    np.nanmean(np.gradient(trace[np.isfinite(trace)],
                                           durations[np.isfinite(trace)]))
                )

        result = GeometricCZFitResult(
            optimal_amplitude=float(v_opt) if np.isfinite(v_opt) else float("nan"),
            optimal_duration=float(t_opt),
            t_pi_cphase=t_pi.tolist(),
            conditional_phase_rate=cond_rate,
            success=success,
        )
        fit_results[qp.name] = asdict(result)

        fit_vars[f"conditional_phase_{qp.name}"] = xr.DataArray(
            cond_phase,
            dims=["exchange_amplitude", "exchange_duration"],
            attrs={
                "long_name": "conditional phase φ₁−φ₀−π",
                "units": "rad",
            },
        )
        fit_vars[f"phi_ctrl_ground_{qp.name}"] = xr.DataArray(
            phi0_all,
            dims=["exchange_amplitude", "exchange_duration"],
            attrs={"long_name": "Ramsey phase (control |0>, unwrapped)", "units": "rad"},
        )
        fit_vars[f"phi_ctrl_excited_{qp.name}"] = xr.DataArray(
            phi1_all,
            dims=["exchange_amplitude", "exchange_duration"],
            attrs={"long_name": "Ramsey phase (control |1>, unwrapped)", "units": "rad"},
        )
        fit_vars[f"t_pi_cphase_{qp.name}"] = xr.DataArray(
            t_pi,
            dims=["exchange_amplitude"],
            attrs={"long_name": "t(π cphase)", "units": "ns"},
        )
        fit_vars[f"cos_conditional_phase_{qp.name}"] = xr.DataArray(
            np.cos(cond_phase),
            dims=["exchange_amplitude", "exchange_duration"],
            attrs={
                "long_name": "cos(φ₁−φ₀−π)",
                "units": "",
            },
        )
        fit_vars[f"cond_phase_at_target_{qp.name}"] = xr.DataArray(
            cond_at_target,
            dims=["exchange_amplitude"],
            attrs={
                "long_name": f"conditional phase at t={target_dur:.0f} ns",
                "units": "rad",
            },
        )

    ds_fit = xr.Dataset(
        fit_vars,
        coords={"exchange_amplitude": amplitudes, "exchange_duration": durations},
    )
    return ds_fit, fit_results


def log_fitted_results(
    fit_results: dict[str, dict[str, Any]],
    log_callable: Any | None = None,
) -> None:
    """Log geometric CZ fit results for all qubit pairs."""
    _log = log_callable or logger.info
    for name, r in sorted(fit_results.items()):
        status = "OK" if r["success"] else "FAILED"
        msg = (
            f"  {name}: [{status}] "
            f"V* = {r['optimal_amplitude']:.4f} V, "
            f"t* = {r['optimal_duration']:.0f} ns, "
            f"rate = {r['conditional_phase_rate'] * 1e3:.3f} mrad/ns"
        )
        _log(msg)
