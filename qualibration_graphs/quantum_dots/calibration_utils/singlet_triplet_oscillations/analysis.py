from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import xarray as xr


@dataclass
class SingletTripletFitResult:
    """Per-qubit-pair result of the singlet-triplet oscillation analysis."""

    success: bool = False
    signal_name: str = ""
    oscillation_frequency_mhz: float = float("nan")
    oscillation_period_ns: float = float("nan")
    oscillation_amplitude: float = float("nan")
    phase_rad: float = float("nan")
    offset: float = float("nan")
    r_squared: float = float("nan")


def _estimate_frequency_fft(
    times_ns: np.ndarray,
    signal: np.ndarray,
) -> tuple[float, float]:
    """Estimate oscillation frequency (MHz) and SNR-like peak ratio from FFT."""
    centered = signal - np.mean(signal)
    n = centered.size
    if n < 8:
        return np.nan, 0.0

    dt_s = float(times_ns[1] - times_ns[0]) * 1e-9 if n > 1 else 1e-9
    fft_vals = np.fft.rfft(centered)
    freqs_hz = np.fft.rfftfreq(n, d=dt_s)

    power = np.abs(fft_vals) ** 2
    if power.size <= 2:
        return np.nan, 0.0

    power[0] = 0.0
    peak_idx = int(np.argmax(power))
    peak_power = float(power[peak_idx])
    baseline = float(np.median(power[1:])) if power.size > 2 else 1e-20
    snr_like = peak_power / max(baseline, 1e-20)
    return float(freqs_hz[peak_idx] * 1e-6), snr_like


def _fit_sinusoid_at_frequency(
    times_ns: np.ndarray,
    signal: np.ndarray,
    frequency_mhz: float,
) -> tuple[float, float, float, float]:
    """Fit y = offset + c*cos(wt) + s*sin(wt) and return (amp, phase, offset, r2)."""
    t_us = times_ns * 1e-3
    omega = 2.0 * np.pi * frequency_mhz

    design = np.column_stack(
        [
            np.ones_like(t_us),
            np.cos(omega * t_us),
            np.sin(omega * t_us),
        ]
    )
    coeffs, _, _, _ = np.linalg.lstsq(design, signal, rcond=None)
    offset, c_term, s_term = coeffs

    model = design @ coeffs
    residual = signal - model
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((signal - np.mean(signal)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else np.nan

    amplitude = float(np.sqrt(c_term**2 + s_term**2))
    phase = float(np.arctan2(-s_term, c_term))
    return amplitude, phase, float(offset), r2


def analyse_singlet_triplet_oscillations(
    ds_raw: xr.Dataset,
    qubit_pair_name: str,
    *,
    analysis_signal: str = "state",
) -> tuple[xr.Dataset, dict[str, Any]]:
    """Analyse the wait-time oscillation trace for one qubit pair."""
    wait_ns = ds_raw.coords["wait_duration"].values.astype(np.float64)
    signal_key = f"{analysis_signal}_{qubit_pair_name}"
    if signal_key not in ds_raw:
        raise KeyError(f"Could not find signal '{signal_key}' in dataset.")

    signal = ds_raw[signal_key].values.astype(np.float64)
    freq_mhz, _snr_like = _estimate_frequency_fft(wait_ns, signal)

    result = SingletTripletFitResult(signal_name=analysis_signal)
    ds_fit = xr.Dataset(coords={"wait_duration": wait_ns})

    if np.isfinite(freq_mhz) and freq_mhz > 0.0:
        amp, phase, offset, r2 = _fit_sinusoid_at_frequency(wait_ns, signal, freq_mhz)
        period_ns = 1e3 / freq_mhz
        model = offset + amp * np.cos(2.0 * np.pi * freq_mhz * wait_ns * 1e-3 + phase)

        result = SingletTripletFitResult(
            success=bool(np.isfinite(r2)),
            signal_name=analysis_signal,
            oscillation_frequency_mhz=float(freq_mhz),
            oscillation_period_ns=float(period_ns),
            oscillation_amplitude=float(amp),
            phase_rad=float(phase),
            offset=float(offset),
            r_squared=float(r2),
        )

        ds_fit["model"] = xr.DataArray(
            model,
            dims=["wait_duration"],
            coords={"wait_duration": wait_ns},
            attrs={"long_name": "sinusoidal fit", "units": ""},
        )

    return ds_fit, asdict(result)


def log_fitted_results(
    qubit_pair_name: str,
    fit_result: dict[str, Any],
    log_callable: Any | None = None,
) -> None:
    """Log singlet-triplet fit summary."""
    _log = log_callable or print
    if not fit_result.get("success", False):
        _log(f"  {qubit_pair_name}: [FAILED] could not extract oscillation period.")
        return

    _log(
        f"  {qubit_pair_name}: [OK] "
        f"signal={fit_result['signal_name']}, "
        f"f={fit_result['oscillation_frequency_mhz']:.4f} MHz, "
        f"T={fit_result['oscillation_period_ns']:.1f} ns, "
        f"A={fit_result['oscillation_amplitude']:.4f}, "
        f"R2={fit_result['r_squared']:.4f}"
    )
