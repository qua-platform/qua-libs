"""Analysis and topology utilities for barrier-barrier virtualization (node 03).

Implements an offline-first variant of the stepwise ``B* -> B†`` method from:
T.-K. Hsiao et al., arXiv:2001.07671.

The workflow is:
1. Resolve pair-driven topology (target barriers + nearest-neighbor drives).
2. Extract tunnel coupling ``t_i`` from detuning traces using a finite-temperature
   two-level model.
3. Fit local derivatives ``m_ij = dt_i / dB_j`` from small barrier sweeps.
4. Compose the stepwise barrier transform matrix and report residual crosstalk.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import xarray as xr


def _pair_dot_ids(pair_obj: Any) -> Tuple[str, str]:
    """Return the 2 quantum-dot ids for a pair-like object."""
    dots = getattr(pair_obj, "quantum_dots", None)
    if dots is None or len(dots) != 2:
        raise ValueError(f"Pair '{getattr(pair_obj, 'id', pair_obj)}' must contain exactly 2 quantum dots.")

    ids: List[str] = []
    for dot in dots:
        ids.append(str(getattr(dot, "id", dot)))
    return ids[0], ids[1]


def _pair_barrier_id(pair_obj: Any) -> str | None:
    """Return the associated barrier id for a pair-like object."""
    barrier_gate = getattr(pair_obj, "barrier_gate", None)
    if barrier_gate is None:
        return None
    barrier_id = getattr(barrier_gate, "id", None)
    return str(barrier_id) if barrier_id is not None else None


def _pair_gate_set_id(pair_obj: Any) -> str | None:
    """Return the VirtualGateSet id for a pair-like object, when available."""
    try:
        gate_set = pair_obj.voltage_sequence.gate_set
    except Exception:
        return None
    gate_set_id = getattr(gate_set, "id", None)
    return str(gate_set_id) if gate_set_id is not None else None


def resolve_pair_calibration_topology(machine: Any, pair_names: Sequence[str]) -> Dict[str, Any]:
    """Resolve pair-driven calibration topology from machine objects.

    Parameters
    ----------
    machine : Any
        QuAM-like object exposing ``quantum_dot_pairs`` and optionally
        ``qubit_pairs``.
    pair_names : sequence of str
        Ordered calibration targets. Each name can reference a
        ``quantum_dot_pair`` directly or a ``qubit_pair``.

    Returns
    -------
    dict
        Contains serializable topology metadata:
        - ``pair_resolution``
        - ``barrier_order``
        - ``drive_mapping``
        - ``target_barrier_to_pair_id``
    """
    ordered_names = [str(name) for name in pair_names]
    if len(ordered_names) == 0:
        raise ValueError("pair_names must be provided and non-empty.")

    quantum_dot_pairs = dict(getattr(machine, "quantum_dot_pairs", {}))
    qubit_pairs = dict(getattr(machine, "qubit_pairs", {}))
    if not quantum_dot_pairs:
        raise ValueError("Machine has no quantum_dot_pairs; cannot derive barrier topology.")

    pair_resolution: List[Dict[str, Any]] = []
    barrier_order: List[str] = []
    drive_mapping: Dict[str, List[str]] = {}
    target_barrier_to_pair_id: Dict[str, str] = {}

    for input_name in ordered_names:
        if input_name in quantum_dot_pairs:
            pair_type = "quantum_dot_pair"
            target_pair = quantum_dot_pairs[input_name]
        elif input_name in qubit_pairs:
            pair_type = "qubit_pair"
            qubit_pair = qubit_pairs[input_name]
            target_pair = getattr(qubit_pair, "quantum_dot_pair", None)
            if target_pair is None:
                raise ValueError(f"qubit_pair '{input_name}' has no quantum_dot_pair reference.")
        else:
            raise KeyError(
                f"Pair '{input_name}' not found in machine.quantum_dot_pairs or machine.qubit_pairs."
            )

        target_pair_id = str(getattr(target_pair, "id", input_name))
        target_barrier = _pair_barrier_id(target_pair)
        if target_barrier is None:
            raise ValueError(f"Target pair '{target_pair_id}' has no barrier_gate.")

        if not callable(getattr(target_pair, "ramp_to_detuning", None)):
            raise ValueError(f"Target pair '{target_pair_id}' has no callable ramp_to_detuning method.")

        detuning_axis_name = str(getattr(target_pair, "detuning_axis_name", ""))
        if detuning_axis_name == "":
            raise ValueError(f"Target pair '{target_pair_id}' has empty detuning_axis_name.")

        dot_ids = _pair_dot_ids(target_pair)
        target_gate_set_id = _pair_gate_set_id(target_pair)

        if target_barrier in target_barrier_to_pair_id:
            existing_pair = target_barrier_to_pair_id[target_barrier]
            raise ValueError(
                f"Duplicate target barrier '{target_barrier}' for pairs "
                f"'{existing_pair}' and '{target_pair_id}'."
            )
        target_barrier_to_pair_id[target_barrier] = target_pair_id

        neighbor_pair_ids: List[str] = []
        neighbor_barriers: List[str] = []
        target_dots = set(dot_ids)

        for pair_id_key, other_pair in quantum_dot_pairs.items():
            other_pair_id = str(getattr(other_pair, "id", pair_id_key))
            if other_pair_id == target_pair_id:
                continue

            other_barrier = _pair_barrier_id(other_pair)
            if other_barrier is None:
                continue

            other_dots = set(_pair_dot_ids(other_pair))
            if len(target_dots.intersection(other_dots)) != 1:
                continue

            other_gate_set_id = _pair_gate_set_id(other_pair)
            if (
                target_gate_set_id is not None
                and other_gate_set_id is not None
                and target_gate_set_id != other_gate_set_id
            ):
                continue

            neighbor_pair_ids.append(other_pair_id)
            neighbor_barriers.append(other_barrier)

        drive_barriers = [target_barrier]
        drive_barriers.extend(sorted({b for b in neighbor_barriers if b != target_barrier}))

        pair_resolution.append(
            {
                "input_name": input_name,
                "pair_type": pair_type,
                "target_pair_id": target_pair_id,
                "target_barrier": target_barrier,
                "dot_ids": [dot_ids[0], dot_ids[1]],
                "detuning_axis_name": detuning_axis_name,
                "neighbor_pair_ids": sorted(set(neighbor_pair_ids)),
                "drive_barriers": drive_barriers,
            }
        )

        barrier_order.append(target_barrier)
        drive_mapping[target_barrier] = drive_barriers

    return {
        "pair_resolution": pair_resolution,
        "barrier_order": barrier_order,
        "drive_mapping": drive_mapping,
        "target_barrier_to_pair_id": target_barrier_to_pair_id,
    }


def finite_temperature_excess_charge(
    detuning: np.ndarray,
    tunnel_coupling: float,
    thermal_energy: float,
    center: float,
) -> np.ndarray:
    """Return the finite-temperature two-level charge-transition model."""
    eps = np.asarray(detuning, dtype=float) - float(center)
    t = abs(float(tunnel_coupling))
    kbt = max(abs(float(thermal_energy)), 1e-18)
    omega = np.sqrt(eps * eps + (2.0 * t) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(omega > 0.0, eps / omega, 0.0)
        thermal = np.tanh(omega / (2.0 * kbt))
    return 0.5 * (1.0 - ratio * thermal)


def _solve_paper_signal_model(
    y: np.ndarray,
    excess_charge: np.ndarray,
    detuning_centered: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray, float]:
    """Solve the paper-style linear sensor model for fixed nonlinear parameters.

    Model (caption text of Fig. 2 in arXiv:1803.10352):
        V(eps) = V0 + dV * Q + [s0 + (s1 - s0) * Q] * eps
    """
    q = np.asarray(excess_charge, dtype=float)
    eps = np.asarray(detuning_centered, dtype=float)
    design = np.column_stack([np.ones_like(q), q, eps, q * eps])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coeffs
    rss = float(np.sum((y - pred) ** 2))
    params = {
        "offset": float(coeffs[0]),
        "delta_signal": float(coeffs[1]),
        "slope_q0": float(coeffs[2]),
        "slope_q1": float(coeffs[2] + coeffs[3]),
        "slope_q_diff": float(coeffs[3]),
    }
    return params, pred, rss


def _gradient_center_seed(detuning: np.ndarray, signal: np.ndarray) -> float:
    """Estimate transition center from the maximal absolute gradient."""
    grad = np.gradient(signal, detuning)
    idx = int(np.argmax(np.abs(grad)))
    return float(detuning[idx])


def _model_rss_for_tunnel(
    detuning: np.ndarray,
    signal: np.ndarray,
    tunnel_coupling: float,
    thermal_energy: float,
    center: float,
) -> float:
    base = finite_temperature_excess_charge(detuning, tunnel_coupling, thermal_energy, center)
    _, _, rss = _solve_paper_signal_model(signal, base, detuning - center)
    return float(rss)


def _refine_two_level_fit_locally(
    detuning: np.ndarray,
    signal: np.ndarray,
    best: Dict[str, Any],
    span: float,
    rounds: int = 4,
) -> Dict[str, Any]:
    """Refine (t, kBT, center) around a coarse-grid optimum.

    The coarse grid provides robust initialization, then this local search
    reduces parameter quantization artifacts in extracted tunnel couplings.
    """
    x = np.asarray(detuning, dtype=float)
    y = np.asarray(signal, dtype=float)

    t_floor = max(1e-10, span * 1e-6)
    kbt_floor = max(1e-10, span * 1e-6)
    t_ceiling = max(span * 2.0, t_floor * 20.0)
    kbt_ceiling = max(span * 2.0, kbt_floor * 20.0)
    center_min = float(np.min(x) - 0.2 * span)
    center_max = float(np.max(x) + 0.2 * span)

    log_t = float(np.log(np.clip(float(best["tunnel_coupling"]), t_floor, t_ceiling)))
    log_kbt = float(np.log(np.clip(float(best["thermal_energy"]), kbt_floor, kbt_ceiling)))
    center = float(np.clip(float(best["center"]), center_min, center_max))

    step_log_t = 0.45
    step_log_kbt = 0.45
    step_center = 0.12 * span

    best_local = dict(best)

    for _ in range(max(int(rounds), 1)):
        trial_best: Dict[str, Any] | None = None
        t_trials = np.linspace(log_t - step_log_t, log_t + step_log_t, num=7)
        kbt_trials = np.linspace(log_kbt - step_log_kbt, log_kbt + step_log_kbt, num=7)
        center_trials = np.linspace(center - step_center, center + step_center, num=9)

        for trial_log_t in t_trials:
            t = float(np.clip(np.exp(trial_log_t), t_floor, t_ceiling))
            for trial_log_kbt in kbt_trials:
                kbt = float(np.clip(np.exp(trial_log_kbt), kbt_floor, kbt_ceiling))
                for trial_center in center_trials:
                    center_c = float(np.clip(trial_center, center_min, center_max))
                    base = finite_temperature_excess_charge(x, t, kbt, center_c)
                    params, pred, rss = _solve_paper_signal_model(
                        y=y,
                        excess_charge=base,
                        detuning_centered=(x - center_c),
                    )
                    if not np.isfinite(rss):
                        continue
                    if trial_best is None or rss < trial_best["rss"]:
                        trial_best = {
                            "tunnel_coupling": t,
                            "thermal_energy": kbt,
                            "center": center_c,
                            **params,
                            "pred": pred,
                            "rss": float(rss),
                        }

        if trial_best is None:
            break

        best_local = trial_best
        log_t = float(np.log(np.clip(float(best_local["tunnel_coupling"]), t_floor, t_ceiling)))
        log_kbt = float(np.log(np.clip(float(best_local["thermal_energy"]), kbt_floor, kbt_ceiling)))
        center = float(np.clip(float(best_local["center"]), center_min, center_max))

        step_log_t *= 0.45
        step_log_kbt *= 0.45
        step_center *= 0.45

    return best_local


def _estimate_tunnel_sigma(
    detuning: np.ndarray,
    signal: np.ndarray,
    tunnel_coupling: float,
    thermal_energy: float,
    center: float,
    rss_best: float,
) -> float:
    """Estimate local uncertainty on tunnel coupling from RSS curvature."""
    t = max(float(tunnel_coupling), 1e-12)
    span = max(float(np.ptp(detuning)), 1e-12)
    delta = max(0.05 * abs(t), span * 1e-3, 1e-9)
    if t <= delta:
        return float("nan")

    t_minus = t - delta
    t_plus = t + delta

    rss_minus = _model_rss_for_tunnel(detuning, signal, t_minus, thermal_energy, center)
    rss_plus = _model_rss_for_tunnel(detuning, signal, t_plus, thermal_energy, center)

    curvature = (rss_plus - 2.0 * rss_best + rss_minus) / (delta * delta)
    if not np.isfinite(curvature) or curvature <= 0.0:
        return float("nan")

    dof = max(detuning.size - 7, 1)
    sigma2 = rss_best / dof
    sigma_t = np.sqrt(max(2.0 * sigma2 / curvature, 0.0))
    return float(sigma_t) if np.isfinite(sigma_t) else float("nan")


def fit_finite_temperature_two_level(
    detuning: np.ndarray,
    signal: np.ndarray,
    n_tunnel: int = 36,
    n_thermal: int = 30,
    n_center: int = 31,
) -> Dict[str, Any]:
    """Fit tunnel coupling from a detuning trace using a grid-search model fit."""
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
                params, pred, rss = _solve_paper_signal_model(
                    y=y,
                    excess_charge=base,
                    detuning_centered=(x - center),
                )
                if not np.isfinite(rss):
                    continue
                if best is None or rss < best["rss"]:
                    best = {
                        "tunnel_coupling": float(t),
                        "thermal_energy": float(kbt),
                        "center": float(center),
                        **params,
                        "pred": pred,
                        "rss": float(rss),
                    }

    if best is None:
        return {
            "success": False,
            "reason": "fit_failed",
            "n_points": int(x.size),
        }

    best = _refine_two_level_fit_locally(
        detuning=x,
        signal=y,
        best=best,
        span=span,
        rounds=4,
    )

    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - best["rss"] / tss if tss > 0.0 else 1.0
    residual_std = float(np.sqrt(best["rss"] / max(x.size - 2, 1)))
    tunnel_sigma = _estimate_tunnel_sigma(
        detuning=x,
        signal=y,
        tunnel_coupling=float(best["tunnel_coupling"]),
        thermal_energy=float(best["thermal_energy"]),
        center=float(best["center"]),
        rss_best=float(best["rss"]),
    )

    return {
        "success": True,
        "n_points": int(x.size),
        "tunnel_coupling": float(best["tunnel_coupling"]),
        "tunnel_sigma": float(tunnel_sigma),
        "thermal_energy": float(best["thermal_energy"]),
        "center": float(best["center"]),
        "offset": float(best["offset"]),
        "delta_signal": float(best["delta_signal"]),
        "slope_q0": float(best["slope_q0"]),
        "slope_q1": float(best["slope_q1"]),
        "slope_q_diff": float(best["slope_q_diff"]),
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

    for _, data_var in ds.data_vars.items():
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
    detuning_scale: float = 1.0,
) -> Dict[str, Any]:
    """Extract tunnel coupling for each drive-axis setpoint."""
    if drive_axis not in ds.coords:
        raise KeyError(f"Drive axis '{drive_axis}' not found in dataset coordinates.")
    if detuning_axis not in ds.coords:
        raise KeyError(f"Detuning axis '{detuning_axis}' not found in dataset coordinates.")

    signal = _select_signal_dataarray(ds)
    if drive_axis not in signal.dims or detuning_axis not in signal.dims:
        raise ValueError(
            f"Signal dims {signal.dims} do not include required axes "
            f"'{drive_axis}' and '{detuning_axis}'."
        )

    drive_values = np.asarray(ds.coords[drive_axis].values, dtype=float)
    detuning_values = np.asarray(ds.coords[detuning_axis].values, dtype=float) * float(detuning_scale)

    drive_idx = signal.get_axis_num(drive_axis)
    detuning_idx = signal.get_axis_num(detuning_axis)
    values = np.asarray(signal.values, dtype=float)
    values = np.moveaxis(values, (drive_idx, detuning_idx), (0, 1))
    if values.ndim > 2:
        values = np.nanmean(values, axis=tuple(range(2, values.ndim)))

    tunnel = np.full(drive_values.shape, np.nan, dtype=float)
    tunnel_sigma = np.full(drive_values.shape, np.nan, dtype=float)
    fit_quality = np.full(drive_values.shape, np.nan, dtype=float)
    per_point_fits: List[Dict[str, Any]] = []

    for idx, drive_value in enumerate(drive_values):
        fit = fit_finite_temperature_two_level(detuning_values, values[idx, :])
        fit["drive_value"] = float(drive_value)
        per_point_fits.append(fit)
        if fit.get("success", False):
            tunnel[idx] = float(fit["tunnel_coupling"])
            tunnel_sigma[idx] = float(fit.get("tunnel_sigma", np.nan))
            fit_quality[idx] = float(fit.get("fit_quality", np.nan))

    return {
        "drive_values": drive_values,
        "detuning_values": detuning_values,
        "detuning_scale": float(detuning_scale),
        "tunnel_couplings": tunnel,
        "tunnel_sigmas": tunnel_sigma,
        "median_tunnel_sigma": float(np.nanmedian(tunnel_sigma)),
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
    detuning_scale: float = 1.0,
) -> Dict[str, Any]:
    """Fit barrier cross-talk by extracting ``t`` vs barrier and regressing slope."""
    tunnel_curve = extract_tunnel_coupling_vs_drive(
        ds,
        barrier_axis,
        compensation_axis,
        detuning_scale=detuning_scale,
    )
    slope_fit = fit_linear_slope(tunnel_curve["drive_values"], tunnel_curve["tunnel_couplings"])

    result = {
        "barrier_axis": barrier_axis,
        "compensation_axis": compensation_axis,
        "detuning_scale": float(detuning_scale),
        "drive_values": tunnel_curve["drive_values"],
        "tunnel_couplings": tunnel_curve["tunnel_couplings"],
        "tunnel_sigmas": tunnel_curve["tunnel_sigmas"],
        "median_tunnel_sigma": float(tunnel_curve["median_tunnel_sigma"]),
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


def evaluate_slope_fit_acceptance(
    fit_result: Dict[str, Any],
    min_slope_snr: float = 5.0,
    min_tunnel_span_sigma: float = 3.0,
    min_pair_fit_r2: float = 0.9,
) -> Dict[str, Any]:
    """Evaluate fit-quality and sensitivity acceptance gates for one pair fit."""
    slope = float(fit_result.get("coefficient", np.nan))
    slope_stderr = float(fit_result.get("slope_stderr", np.nan))
    fit_r2 = float(fit_result.get("fit_quality", np.nan))

    if np.isfinite(slope_stderr) and slope_stderr > 0.0:
        slope_snr = abs(slope) / slope_stderr
    elif np.isfinite(slope) and slope != 0.0 and slope_stderr == 0.0:
        slope_snr = float("inf")
    else:
        slope_snr = float("nan")

    tunnel_values = np.asarray(fit_result.get("tunnel_couplings", []), dtype=float)
    tunnel_sigmas = np.asarray(fit_result.get("tunnel_sigmas", []), dtype=float)
    valid_tunnel = np.isfinite(tunnel_values)

    tunnel_span = float(np.nanmax(tunnel_values) - np.nanmin(tunnel_values)) if np.any(valid_tunnel) else float("nan")
    median_sigma = float(np.nanmedian(tunnel_sigmas)) if np.any(np.isfinite(tunnel_sigmas)) else float("nan")
    required_span = float(min_tunnel_span_sigma) * median_sigma if np.isfinite(median_sigma) else float("nan")

    r2_ok = bool(np.isfinite(fit_r2) and fit_r2 >= float(min_pair_fit_r2))
    snr_ok = bool(np.isfinite(slope_snr) and slope_snr >= float(min_slope_snr))
    span_ok = bool(np.isfinite(tunnel_span) and np.isfinite(required_span) and tunnel_span >= required_span)

    reasons: List[str] = []
    if not r2_ok:
        reasons.append("low_r2")
    if not snr_ok:
        reasons.append("low_slope_snr")
    if not span_ok:
        reasons.append("low_tunnel_span")

    accepted = bool(fit_result.get("success", False) and r2_ok and snr_ok and span_ok)

    return {
        "accepted": accepted,
        "r2_ok": r2_ok,
        "snr_ok": snr_ok,
        "span_ok": span_ok,
        "slope_snr": float(slope_snr),
        "tunnel_span": float(tunnel_span),
        "median_tunnel_sigma": float(median_sigma),
        "required_tunnel_span": float(required_span),
        "reasons": reasons,
    }


def extract_barrier_compensation_coefficients(
    ds: xr.Dataset,
    barrier_gate_name: str,
    compensation_gate_name: str,
    detuning_scale: float = 1.0,
) -> Dict[str, float]:
    """Extract barrier compensation coefficients from one 2D scan."""
    barrier_axis = barrier_gate_name if barrier_gate_name in ds.coords else "x_volts"
    compensation_axis = compensation_gate_name if compensation_gate_name in ds.coords else "y_volts"
    return fit_barrier_cross_talk(
        ds,
        barrier_axis,
        compensation_axis,
        detuning_scale=detuning_scale,
    )


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
    """Compose ``B* -> B†`` barrier transform from a raw slope matrix."""
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
