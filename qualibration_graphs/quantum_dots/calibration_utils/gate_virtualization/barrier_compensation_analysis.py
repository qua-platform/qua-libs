"""Analysis functions for barrier-barrier compensation (node 03).

Implements an offline-first variant of the stepwise ``B* -> B†`` method from:
T.-K. Hsiao et al., arXiv:2001.07671.

The workflow is:
1. Extract tunnel coupling ``t_i`` from detuning traces using a finite-temperature
   two-level model.
2. Fit local derivatives ``m_ij = dt_i / dB_j`` from small barrier sweeps.
3. Compose the stepwise barrier transform matrix and report residual crosstalk.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import xarray as xr


def finite_temperature_excess_charge(
    detuning: np.ndarray,
    tunnel_coupling: float,
    thermal_energy: float,
    center: float,
) -> np.ndarray:
    """Return the finite-temperature two-level charge-transition model.

    The returned value is dimensionless and later scaled/offset to fit
    measured sensor response.
    """
    eps = np.asarray(detuning, dtype=float) - float(center)
    t = abs(float(tunnel_coupling))
    kbt = max(abs(float(thermal_energy)), 1e-18)
    omega = np.sqrt(eps * eps + (2.0 * t) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(omega > 0.0, eps / omega, 0.0)
        thermal = np.tanh(omega / (2.0 * kbt))
    return 0.5 * (1.0 - ratio * thermal)


def _solve_linear_scale_offset(y: np.ndarray, base: np.ndarray) -> Tuple[float, float, np.ndarray, float]:
    """Solve ``y ~= offset + amplitude * base`` in least squares sense."""
    design = np.column_stack([np.ones_like(base), base])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coeffs
    rss = float(np.sum((y - pred) ** 2))
    return float(coeffs[0]), float(coeffs[1]), pred, rss


def _gradient_center_seed(detuning: np.ndarray, signal: np.ndarray) -> float:
    """Estimate transition center from the maximal absolute gradient."""
    grad = np.gradient(signal, detuning)
    idx = int(np.argmax(np.abs(grad)))
    return float(detuning[idx])


def fit_finite_temperature_two_level(
    detuning: np.ndarray,
    signal: np.ndarray,
    n_tunnel: int = 36,
    n_thermal: int = 30,
    n_center: int = 31,
) -> Dict[str, Any]:
    """Fit tunnel coupling from a detuning trace using a grid-search model fit.

    A robust grid search is used to avoid hard dependency on SciPy in test
    environments.
    """
    x = np.asarray(detuning, dtype=float)
    y = np.asarray(signal, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size < 5:
        return {
            "success": False,
            "reason": "insufficient_points",
            "n_points": int(x.size),
        }

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    span = float(np.ptp(x))
    if span <= 0.0:
        return {
            "success": False,
            "reason": "zero_detuning_span",
            "n_points": int(x.size),
        }

    center_seed = _gradient_center_seed(x, y)
    t_min = max(span * 1e-4, 1e-8)
    t_max = max(span * 0.6, t_min * 10.0)
    kbt_min = max(span * 2e-4, 1e-8)
    kbt_max = max(span * 0.7, kbt_min * 10.0)
    center_half_span = 0.25 * span

    tunnel_grid = np.geomspace(t_min, t_max, num=max(n_tunnel, 8))
    thermal_grid = np.geomspace(kbt_min, kbt_max, num=max(n_thermal, 8))
    center_grid = np.linspace(center_seed - center_half_span, center_seed + center_half_span, num=max(n_center, 9))

    best: Dict[str, Any] | None = None
    for t in tunnel_grid:
        for kbt in thermal_grid:
            for center in center_grid:
                base = finite_temperature_excess_charge(x, t, kbt, center)
                offset, amplitude, pred, rss = _solve_linear_scale_offset(y, base)
                if not np.isfinite(rss):
                    continue
                if best is None or rss < best["rss"]:
                    best = {
                        "tunnel_coupling": float(t),
                        "thermal_energy": float(kbt),
                        "center": float(center),
                        "offset": float(offset),
                        "amplitude": float(amplitude),
                        "pred": pred,
                        "rss": float(rss),
                    }

    if best is None:
        return {
            "success": False,
            "reason": "fit_failed",
            "n_points": int(x.size),
        }

    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - best["rss"] / tss if tss > 0.0 else 1.0
    residual_std = float(np.sqrt(best["rss"] / max(x.size - 2, 1)))

    return {
        "success": True,
        "n_points": int(x.size),
        "tunnel_coupling": float(best["tunnel_coupling"]),
        "thermal_energy": float(best["thermal_energy"]),
        "center": float(best["center"]),
        "offset": float(best["offset"]),
        "amplitude": float(best["amplitude"]),
        "fit_quality": float(r2),
        "rss": float(best["rss"]),
        "residual_std": residual_std,
        "detuning": x,
        "signal": y,
        "signal_fit": np.asarray(best["pred"], dtype=float),
    }


def _select_signal_dataarray(ds: xr.Dataset) -> xr.DataArray:
    """Select a signal variable suitable for tunnel-coupling extraction."""
    for candidate in ("amplitude", "IQ_abs", "I"):
        if candidate in ds.data_vars:
            signal = ds[candidate]
            if "sensors" in signal.dims and signal.sizes.get("sensors", 0) > 0:
                signal = signal.isel(sensors=0)
            return signal

    # Fallback: choose first numeric variable.
    for name, data_var in ds.data_vars.items():
        if np.issubdtype(data_var.dtype, np.number):
            signal = data_var
            if "sensors" in signal.dims and signal.sizes.get("sensors", 0) > 0:
                signal = signal.isel(sensors=0)
            return signal

    raise ValueError("No numeric signal variable found in dataset.")


def extract_tunnel_coupling_vs_drive(
    ds: xr.Dataset,
    drive_axis: str,
    detuning_axis: str,
) -> Dict[str, Any]:
    """Extract tunnel coupling for each drive-axis setpoint.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing at least a 2D signal map over ``drive_axis`` and
        ``detuning_axis``.
    drive_axis : str
        Axis along which barrier values are swept.
    detuning_axis : str
        Axis used to fit the finite-temperature transition model.
    """
    if drive_axis not in ds.coords:
        raise KeyError(f"Drive axis '{drive_axis}' not found in dataset coordinates.")
    if detuning_axis not in ds.coords:
        raise KeyError(f"Detuning axis '{detuning_axis}' not found in dataset coordinates.")

    signal = _select_signal_dataarray(ds)
    if drive_axis not in signal.dims or detuning_axis not in signal.dims:
        raise ValueError(
            f"Signal dims {signal.dims} do not include required axes " f"'{drive_axis}' and '{detuning_axis}'."
        )

    drive_values = np.asarray(ds.coords[drive_axis].values, dtype=float)
    detuning_values = np.asarray(ds.coords[detuning_axis].values, dtype=float)

    drive_idx = signal.get_axis_num(drive_axis)
    detuning_idx = signal.get_axis_num(detuning_axis)
    values = np.asarray(signal.values, dtype=float)
    values = np.moveaxis(values, (drive_idx, detuning_idx), (0, 1))
    if values.ndim > 2:
        values = np.nanmean(values, axis=tuple(range(2, values.ndim)))

    tunnel = np.full(drive_values.shape, np.nan, dtype=float)
    fit_quality = np.full(drive_values.shape, np.nan, dtype=float)
    per_point_fits: List[Dict[str, Any]] = []

    for idx, drive_value in enumerate(drive_values):
        fit = fit_finite_temperature_two_level(detuning_values, values[idx, :])
        fit["drive_value"] = float(drive_value)
        per_point_fits.append(fit)
        if fit.get("success", False):
            tunnel[idx] = float(fit["tunnel_coupling"])
            fit_quality[idx] = float(fit.get("fit_quality", np.nan))

    return {
        "drive_values": drive_values,
        "detuning_values": detuning_values,
        "tunnel_couplings": tunnel,
        "point_fit_quality": fit_quality,
        "point_fits": per_point_fits,
    }


def fit_linear_slope(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Fit ``y = slope * x + intercept`` with uncertainty and quality metrics."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size < 2:
        return {
            "success": False,
            "reason": "insufficient_points",
            "n_points": int(x.size),
        }

    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    residual = y - pred
    rss = float(np.sum(residual**2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / tss if tss > 0.0 else 1.0

    if x.size >= 3:
        sxx = float(np.sum((x - np.mean(x)) ** 2))
        if sxx > 0.0:
            sigma2 = rss / (x.size - 2)
            slope_stderr = float(np.sqrt(max(sigma2 / sxx, 0.0)))
        else:
            slope_stderr = np.nan
    else:
        slope_stderr = np.nan

    return {
        "success": True,
        "n_points": int(x.size),
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_stderr": float(slope_stderr),
        "fit_quality": float(r2),
        "x": x,
        "y": y,
        "y_fit": pred,
        "rss": rss,
    }


def fit_barrier_cross_talk(
    ds: xr.Dataset,
    barrier_axis: str,
    compensation_axis: str,
) -> Dict[str, Any]:
    """Fit barrier cross-talk by extracting ``t`` vs barrier and regressing slope.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset containing a 2D signal map.
    barrier_axis : str
        Coordinate name for the swept drive-barrier axis.
    compensation_axis : str
        Coordinate name for the detuning axis.

    Returns
    -------
    dict
        ``{"coefficient": float, "fit_quality": float, ...}``
    """
    tunnel_curve = extract_tunnel_coupling_vs_drive(ds, barrier_axis, compensation_axis)
    slope_fit = fit_linear_slope(tunnel_curve["drive_values"], tunnel_curve["tunnel_couplings"])

    result = {
        "barrier_axis": barrier_axis,
        "compensation_axis": compensation_axis,
        "drive_values": tunnel_curve["drive_values"],
        "tunnel_couplings": tunnel_curve["tunnel_couplings"],
        "point_fit_quality": tunnel_curve["point_fit_quality"],
        "point_fits": tunnel_curve["point_fits"],
        "success": bool(slope_fit.get("success", False)),
        "coefficient": float(slope_fit.get("slope", np.nan)),
        "fit_quality": float(slope_fit.get("fit_quality", np.nan)),
        "slope_stderr": float(slope_fit.get("slope_stderr", np.nan)),
        "n_points": int(slope_fit.get("n_points", 0)),
    }
    result.update({f"slope_fit_{k}": v for k, v in slope_fit.items()})
    return result


def extract_barrier_compensation_coefficients(
    ds: xr.Dataset,
    barrier_gate_name: str,
    compensation_gate_name: str,
) -> Dict[str, float]:
    """Extract barrier compensation coefficients from one 2D scan.

    Axis names are resolved as follows:
    - If ``barrier_gate_name`` exists in coordinates, it is used.
    - Otherwise, fall back to ``x_volts``.
    - If ``compensation_gate_name`` exists in coordinates, it is used.
    - Otherwise, fall back to ``y_volts``.
    """
    barrier_axis = barrier_gate_name if barrier_gate_name in ds.coords else "x_volts"
    compensation_axis = compensation_gate_name if compensation_gate_name in ds.coords else "y_volts"
    return fit_barrier_cross_talk(ds, barrier_axis, compensation_axis)


def compute_barrier_matrix_entries(
    fit_results: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    """Convert per-pair fit results into compensation matrix entries."""
    entries: List[Dict[str, Any]] = []
    for pair_key, result in fit_results.items():
        if not result.get("success", False):
            continue
        row = result.get("target_barrier")
        col = result.get("drive_barrier")
        if row is None or col is None:
            if "_vs_" in pair_key:
                row, col = pair_key.split("_vs_", 1)
            else:
                continue
        entries.append(
            {
                "row": str(row),
                "col": str(col),
                "value": float(result.get("coefficient", np.nan)),
                "fit_quality": float(result.get("fit_quality", np.nan)),
            }
        )
    return entries


def assemble_slope_matrix(
    fit_results: Dict[str, Dict[str, Any]],
    barrier_order: Sequence[str],
) -> np.ndarray:
    """Assemble the raw slope matrix ``m_ij = dt_i/dB_j`` from fit results."""
    barrier_order = list(barrier_order)
    index = {name: idx for idx, name in enumerate(barrier_order)}
    matrix = np.full((len(barrier_order), len(barrier_order)), np.nan, dtype=float)

    for pair_key, result in fit_results.items():
        target = result.get("target_barrier")
        drive = result.get("drive_barrier")
        if target is None or drive is None:
            if "_vs_" in pair_key:
                target, drive = pair_key.split("_vs_", 1)
            else:
                continue

        if target not in index or drive not in index:
            continue

        matrix[index[target], index[drive]] = float(result.get("coefficient", np.nan))

    return matrix


def row_normalize_with_unit_diagonal(matrix: np.ndarray, min_abs_diag: float = 1e-12) -> np.ndarray:
    """Normalize each row so diagonal elements are exactly 1."""
    out = np.asarray(matrix, dtype=float).copy()
    n = out.shape[0]
    for row in range(n):
        diag = out[row, row]
        if not np.isfinite(diag) or abs(diag) < min_abs_diag:
            raise ValueError(f"Invalid diagonal element at row {row}: {diag}")
        out[row, :] /= diag
        out[row, row] = 1.0
    return out


def compute_residual_crosstalk_ratios(
    slope_matrix: np.ndarray,
    barrier_names: Sequence[str],
    min_abs_diag: float = 1e-12,
) -> Dict[str, Any]:
    """Compute per-row off-diagonal crosstalk ratios ``|m_ij / m_ii|``."""
    m = np.asarray(slope_matrix, dtype=float)
    n = m.shape[0]
    per_barrier: Dict[str, Dict[str, float]] = {}
    all_ratios: List[float] = []

    for row in range(n):
        row_name = barrier_names[row]
        diag = m[row, row]
        if not np.isfinite(diag) or abs(diag) < min_abs_diag:
            raise ValueError(f"Invalid self slope for barrier '{row_name}': {diag}")

        row_ratios: Dict[str, float] = {}
        for col in range(n):
            if col == row:
                continue
            ratio = abs(float(m[row, col] / diag)) if np.isfinite(m[row, col]) else np.inf
            row_ratios[barrier_names[col]] = ratio
            all_ratios.append(ratio)
        per_barrier[row_name] = row_ratios

    max_ratio = float(max(all_ratios)) if all_ratios else 0.0
    return {
        "per_barrier": per_barrier,
        "max_ratio": max_ratio,
    }


def calibrate_stepwise_barrier_virtualization(
    slope_matrix_raw: np.ndarray,
    barrier_names: Sequence[str],
    calibration_order: Sequence[str] | None = None,
    residual_target: float = 0.10,
    max_refinement_rounds: int = 2,
    min_abs_self_slope: float = 1e-12,
) -> Dict[str, Any]:
    """Compose ``B* -> B†`` barrier transform from a raw slope matrix.

    The implementation follows the stepwise spirit of the paper while using
    an offline-predictive update model:
    - Effective slope matrix in current virtual basis:
      ``M_eff = M_raw @ inv(C)``.
    - Per-row update matrix ``D_i`` is built from normalized row ratios in
      ``M_eff``.
    - Cumulative transform is updated as ``C <- D_i @ C``.
    """
    barrier_names = list(barrier_names)
    n = len(barrier_names)
    if n == 0:
        raise ValueError("barrier_names must be non-empty")

    m_raw = np.asarray(slope_matrix_raw, dtype=float)
    if m_raw.shape != (n, n):
        raise ValueError(f"slope_matrix_raw shape must be {(n, n)}, got {m_raw.shape}")

    if calibration_order is None:
        calibration_order = barrier_names
    calibration_order = list(calibration_order)
    unknown = [name for name in calibration_order if name not in barrier_names]
    if unknown:
        raise KeyError(f"Unknown barrier names in calibration_order: {unknown}")

    order_idx = [barrier_names.index(name) for name in calibration_order]

    c_transform = np.eye(n, dtype=float)
    history: List[Dict[str, Any]] = []

    total_rounds = max(1, int(max_refinement_rounds) + 1)
    step_counter = 0

    for round_idx in range(total_rounds):
        c_inv = np.linalg.pinv(c_transform)
        m_eff = m_raw @ c_inv

        for row_idx in order_idx:
            diag = m_eff[row_idx, row_idx]
            if not np.isfinite(diag) or abs(diag) < min_abs_self_slope:
                raise ValueError(f"Invalid self slope for barrier '{barrier_names[row_idx]}': {diag}")

            d_step = np.eye(n, dtype=float)
            d_step[row_idx, :] = m_eff[row_idx, :] / diag
            d_step[row_idx, row_idx] = 1.0

            c_transform = d_step @ c_transform
            c_transform = row_normalize_with_unit_diagonal(c_transform, min_abs_diag=min_abs_self_slope)

            step_counter += 1
            c_inv_step = np.linalg.pinv(c_transform)
            m_eff_step = m_raw @ c_inv_step
            residual_step = compute_residual_crosstalk_ratios(
                m_eff_step,
                barrier_names,
                min_abs_diag=min_abs_self_slope,
            )
            history.append(
                {
                    "label": f"B*{step_counter}",
                    "round": round_idx + 1,
                    "target_barrier": barrier_names[row_idx],
                    "matrix": c_transform.copy(),
                    "max_residual": float(residual_step["max_ratio"]),
                }
            )

        final_eff = m_raw @ np.linalg.pinv(c_transform)
        residual = compute_residual_crosstalk_ratios(
            final_eff,
            barrier_names,
            min_abs_diag=min_abs_self_slope,
        )
        if residual["max_ratio"] <= residual_target:
            break

    final_eff = m_raw @ np.linalg.pinv(c_transform)
    residual = compute_residual_crosstalk_ratios(
        final_eff,
        barrier_names,
        min_abs_diag=min_abs_self_slope,
    )

    history.append(
        {
            "label": "B†",
            "round": len(history) // max(len(order_idx), 1) + 1,
            "target_barrier": None,
            "matrix": c_transform.copy(),
            "max_residual": float(residual["max_ratio"]),
        }
    )

    return {
        "barrier_transform_final": c_transform,
        "barrier_transform_history": history,
        "residual_crosstalk": residual["per_barrier"],
        "max_residual_crosstalk": float(residual["max_ratio"]),
        "effective_slope_matrix": final_eff,
        "slope_matrix_raw": m_raw,
    }
