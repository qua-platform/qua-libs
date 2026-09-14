"""CZ phase compensation analysis.

Fits sinusoidal oscillations from Ramsey-like sequences with swept virtual
frame rotations to extract the residual single-qubit phases induced by the
CZ gate. For each qubit pair, 4 experiment types are analysed:

    0: Target in superposition, control in |0⟩ → target unconditional phase
    1: Target in superposition, control in |1⟩ → target + conditional phase
    2: Control in superposition, target in |0⟩ → control unconditional phase
    3: Control in superposition, target in |1⟩ → control + conditional phase

The ground-state-partner experiments (0, 2) give the phase corrections.
The excited-state-partner experiments (1, 3) provide a cross-check that
the conditional phase is pi.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np
from scipy.optimize import curve_fit

import xarray as xr

logger = logging.getLogger(__name__)


@dataclass
class CZPhaseCompensationFitResult:
    """Fit result for one qubit pair."""

    control_phase_correction: float = 0.0
    """Residual phase on control qubit (units of 2pi)."""
    target_phase_correction: float = 0.0
    """Residual phase on target qubit (units of 2pi)."""
    conditional_phase_target: float = 0.0
    """Measured conditional phase on target (radians). Should be ~pi."""
    conditional_phase_control: float = 0.0
    """Measured conditional phase on control (radians). Should be ~pi."""
    success: bool = False


def _sinusoid(
    frame: np.ndarray,
    amplitude: float,
    frequency: float,
    phase: float,
    offset: float,
) -> np.ndarray:
    return amplitude * np.cos(2 * np.pi * frequency * frame + phase) + offset


def _fit_sinusoid(frames: np.ndarray, signal: np.ndarray) -> dict[str, Any]:
    """Fit a sinusoid to a 1-D frame sweep trace.

    Returns dict with amplitude, frequency, phase, offset, fitted_curve, success.
    """
    result: dict[str, Any] = {
        "amplitude": 0.0,
        "frequency": 1.0,
        "phase": 0.0,
        "offset": 0.0,
        "fitted_curve": np.full_like(signal, np.nan),
        "success": False,
    }

    if len(frames) < 4:
        return result

    try:
        amp_guess = float(np.nanmax(signal) - np.nanmin(signal)) / 2.0
        offset_guess = float(np.nanmean(signal))

        fft_vals = np.fft.rfft(signal - offset_guess)
        fft_freqs = np.fft.rfftfreq(len(frames), d=(frames[1] - frames[0]))
        fft_magnitudes = np.abs(fft_vals[1:])
        peak_idx = int(np.argmax(fft_magnitudes)) + 1
        freq_guess = float(fft_freqs[peak_idx]) if peak_idx > 0 else 1.0
        phase_guess = float(-np.angle(fft_vals[peak_idx]))

        freq_guess = max(freq_guess, 0.5)

        popt, _ = curve_fit(
            _sinusoid,
            frames,
            signal,
            p0=[amp_guess, freq_guess, phase_guess, offset_guess],
            bounds=(
                [0, 0.5, -2 * np.pi, -np.inf],
                [np.inf, 2.0, 2 * np.pi, np.inf],
            ),
            maxfev=5000,
        )

        result["amplitude"] = float(popt[0])
        result["frequency"] = float(popt[1])
        result["phase"] = float(popt[2])
        result["offset"] = float(popt[3])
        result["fitted_curve"] = _sinusoid(frames, *popt)
        result["success"] = bool(np.isfinite(popt).all() and popt[0] > 0)
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        pass

    return result


def _phase_to_2pi(phase_rad: float) -> float:
    """Convert a phase in radians to [0, 1) in units of 2pi."""
    return (phase_rad / (2 * np.pi)) % 1


def _unwrap_conditional_phase(raw_diff_rad: float) -> float:
    """Map a raw phase difference to a value near π (same equivalence class mod 2π).

    Fits return phases in (-2π, 2π]; the difference can be π or −π for an ideal CZ.
    This reports a single representative close to π for logging and tolerances.
    """
    return float(np.pi + np.angle(np.exp(1j * (raw_diff_rad - np.pi))))


def fit_raw_data(
    ds_raw: xr.Dataset,
    qubit_pairs: list[Any],
    *,
    analysis_signal: str = "E_p2_given_p1_0",
    conditional_phase_tolerance: float = 0.15,
) -> tuple[xr.Dataset, dict[str, dict[str, Any]]]:
    """Run CZ phase compensation analysis for every qubit pair.

    Parameters
    ----------
    ds_raw : xr.Dataset
        Dataset after ``process_joint_streams`` with coordinates
        ``experiment_type`` and ``frame``, and variables
        ``{analysis_signal}_{pair_name}``.
    qubit_pairs : list
        Qubit pair objects (each must have a ``.name`` attribute).
    analysis_signal : str
        Which conditional expectation to fit.
    conditional_phase_tolerance : float
        Tolerance for the conditional phase cross-check (units of 2pi).

    Returns
    -------
    ds_fit : xr.Dataset
        Fitted quantities per pair.
    fit_results : dict
        Per-pair results with phase corrections, conditional phases, success.
    """
    frames = ds_raw.coords["frame"].values.astype(np.float64)
    fit_results: Dict[str, dict[str, Any]] = {}
    fit_vars: dict[str, xr.DataArray] = {}

    for qp in qubit_pairs:
        var_name = f"{analysis_signal}_{qp.name}"
        if var_name not in ds_raw.data_vars:
            logger.warning("No %s variable for pair %s — skipping.", var_name, qp.name)
            fit_results[qp.name] = asdict(CZPhaseCompensationFitResult())
            continue

        data = ds_raw[var_name].values.astype(np.float64)
        # shape: (n_experiment_types, n_frames)

        fits = {}
        for exp_type in range(4):
            fits[exp_type] = _fit_sinusoid(frames, data[exp_type])

        ground_fits_ok = fits[0]["success"] and fits[2]["success"]

        target_phase_correction = (
            _phase_to_2pi(fits[0]["phase"]) if fits[0]["success"] else 0.0
        )
        control_phase_correction = (
            _phase_to_2pi(fits[2]["phase"]) if fits[2]["success"] else 0.0
        )

        conditional_phase_target = 0.0
        conditional_phase_control = 0.0
        if fits[0]["success"] and fits[1]["success"]:
            raw = float(fits[1]["phase"] - fits[0]["phase"])
            conditional_phase_target = _unwrap_conditional_phase(raw)
            delta = abs(np.angle(np.exp(1j * (raw - np.pi))))
            if delta > conditional_phase_tolerance * 2 * np.pi:
                logger.warning(
                    "Pair %s: target conditional phase %.3f rad deviates from pi "
                    "by %.1f° (tolerance %.1f°).",
                    qp.name,
                    conditional_phase_target,
                    np.degrees(delta),
                    np.degrees(conditional_phase_tolerance * 2 * np.pi),
                )

        if fits[2]["success"] and fits[3]["success"]:
            raw = float(fits[3]["phase"] - fits[2]["phase"])
            conditional_phase_control = _unwrap_conditional_phase(raw)
            delta = abs(np.angle(np.exp(1j * (raw - np.pi))))
            if delta > conditional_phase_tolerance * 2 * np.pi:
                logger.warning(
                    "Pair %s: control conditional phase %.3f rad deviates from pi "
                    "by %.1f° (tolerance %.1f°).",
                    qp.name,
                    conditional_phase_control,
                    np.degrees(delta),
                    np.degrees(conditional_phase_tolerance * 2 * np.pi),
                )

        result = CZPhaseCompensationFitResult(
            control_phase_correction=control_phase_correction,
            target_phase_correction=target_phase_correction,
            conditional_phase_target=conditional_phase_target,
            conditional_phase_control=conditional_phase_control,
            success=ground_fits_ok,
        )
        fit_results[qp.name] = asdict(result)

        exp_labels = ["target_ctrl0", "target_ctrl1", "control_tgt0", "control_tgt1"]
        for exp_type, label in enumerate(exp_labels):
            fit_vars[f"fitted_{label}_{qp.name}"] = xr.DataArray(
                fits[exp_type]["fitted_curve"],
                dims=["frame"],
                attrs={"long_name": f"fitted {label}"},
            )
            fit_vars[f"phase_{label}_{qp.name}"] = xr.DataArray(
                fits[exp_type]["phase"],
                attrs={"long_name": f"fitted phase {label}", "units": "rad"},
            )

    ds_fit = xr.Dataset(fit_vars, coords={"frame": frames})
    return ds_fit, fit_results


def log_fitted_results(
    fit_results: dict[str, dict[str, Any]],
    log_callable: Any | None = None,
) -> None:
    """Log CZ phase compensation fit results for all qubit pairs."""
    _log = log_callable or logger.info
    for name, r in sorted(fit_results.items()):
        status = "OK" if r["success"] else "FAILED"
        msg = (
            f"  {name}: [{status}] "
            f"control_correction = {r['control_phase_correction']:.4f} × 2π, "
            f"target_correction = {r['target_phase_correction']:.4f} × 2π, "
            f"cond_phase_target = {r['conditional_phase_target']:.3f} rad, "
            f"cond_phase_control = {r['conditional_phase_control']:.3f} rad"
        )
        _log(msg)
