"""Heuristic init for chevron fit: FFT per slice → peak fit → t_π(Δ) = π/√(Ω²+δ²)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import curve_fit

FFT_FREQ_MIN = 0.0005  # 1/ns (exclude DC) → t_π max ~1000 ns
FFT_FREQ_MAX = 0.03  # 1/ns → t_π min ~17 ns; restricts to plausible Rabi range


def _lorentzian(x: np.ndarray, amp: float, x0: float, gamma: float) -> np.ndarray:
    """Lorentzian A / (1 + ((x - x0) / gamma)^2). Gamma is HWHM."""
    return amp / (1.0 + ((x - x0) / (gamma + 1e-12)) ** 2)


def _gaussian(x: np.ndarray, amp: float, x0: float, sigma: float) -> np.ndarray:
    """Gaussian amp * exp(-(x - x0)^2 / (2 sigma^2))."""
    return amp * np.exp(-((x - x0) ** 2) / (2.0 * (sigma + 1e-12) ** 2))


def _fit_lorentzian_to_fft(
    freqs_fft: np.ndarray,
    magnitude: np.ndarray,
    freq_min: float,
    freq_max: float,
) -> Tuple[float | None, np.ndarray | None]:
    """Fit Lorentzian to FFT magnitude (↔ exponential decay). Returns (peak_freq, curve) or (None, None)."""
    mask = (freqs_fft >= freq_min) & (freqs_fft <= freq_max)
    if np.sum(mask) < 4:
        return None, None
    f = freqs_fft[mask]
    m = magnitude[mask].astype(float)
    if np.max(m) < 1e-10:
        return None, None
    peak_idx = int(np.argmax(m))
    x0_0 = float(f[peak_idx])
    amp0 = float(m[peak_idx])
    gamma0 = max(1e-6, (f[-1] - f[0]) / 4.0)
    x0_0 = np.clip(x0_0, freq_min + gamma0, freq_max - gamma0)
    try:
        popt, _ = curve_fit(
            _lorentzian,
            f,
            m,
            p0=[amp0, x0_0, gamma0],
            bounds=([0, freq_min, 1e-6], [np.inf, freq_max, np.inf]),
            maxfev=500,
            method="trf",
        )
        x0_fit = float(popt[1])
        if x0_fit <= freq_min + 1e-8 or x0_fit >= freq_max - 1e-8:
            return None, None
        curve = _lorentzian(freqs_fft, popt[0], popt[1], popt[2])
        return x0_fit, curve
    except Exception:
        return None, None


def _fit_gaussian_to_fft(
    freqs_fft: np.ndarray,
    magnitude: np.ndarray,
    freq_min: float,
    freq_max: float,
) -> Tuple[float | None, np.ndarray | None]:
    """Fit Gaussian to FFT magnitude (↔ inhomogeneous broadening). Returns (peak_freq, curve) or (None, None)."""
    mask = (freqs_fft >= freq_min) & (freqs_fft <= freq_max)
    if np.sum(mask) < 4:
        return None, None
    f = freqs_fft[mask]
    m = magnitude[mask].astype(float)
    if np.max(m) < 1e-10:
        return None, None
    peak_idx = int(np.argmax(m))
    x0_0 = float(f[peak_idx])
    amp0 = float(m[peak_idx])
    sigma0 = max(1e-6, (f[-1] - f[0]) / 4.0)
    x0_0 = np.clip(x0_0, freq_min + sigma0, freq_max - sigma0)
    try:
        popt, _ = curve_fit(
            _gaussian,
            f,
            m,
            p0=[amp0, x0_0, sigma0],
            bounds=([0, freq_min, 1e-6], [np.inf, freq_max, np.inf]),
            maxfev=500,
            method="trf",
        )
        x0_fit = float(popt[1])
        if x0_fit <= freq_min + 1e-8 or x0_fit >= freq_max - 1e-8:
            return None, None
        curve = _gaussian(freqs_fft, popt[0], popt[1], popt[2])
        return x0_fit, curve
    except Exception:
        return None, None


def compute_fft_diagnostics(
    pdiff: np.ndarray,
    freqs_hz: np.ndarray,
    durations_ns: np.ndarray,
) -> Dict[str, Any]:
    """FFT per slice, peak fit (Lorentzian/Gaussian), Rabi t_π(Δ) fit. Returns dict of diagnostics."""
    n_freqs, n_dur = pdiff.shape
    dt_ns = float(durations_ns[1] - durations_ns[0]) if len(durations_ns) > 1 else 1.0
    if dt_ns <= 0:
        dt_ns = 1.0

    freqs_fft = np.fft.rfftfreq(n_dur, dt_ns)
    t_pi_per_freq = np.full(n_freqs, np.nan)
    magnitude_per_slice: List[np.ndarray] = []
    lorentzian_mean_per_slice: List[float | None] = []
    lorentzian_curve_per_slice: List[np.ndarray | None] = []

    for i in range(n_freqs):
        trace = np.asarray(pdiff[i, :], dtype=float) - np.mean(pdiff[i, :])
        magnitude_per_slice.append(np.abs(np.fft.rfft(trace)))
        if np.ptp(pdiff[i, :]) < 0.05:
            lorentzian_mean_per_slice.append(None)
            lorentzian_curve_per_slice.append(None)
            continue
        mag = magnitude_per_slice[-1]
        mu, peak_curve = _fit_lorentzian_to_fft(freqs_fft, mag, FFT_FREQ_MIN, FFT_FREQ_MAX)
        if mu is None and peak_curve is None:
            mu, peak_curve = _fit_gaussian_to_fft(freqs_fft, mag, FFT_FREQ_MIN, FFT_FREQ_MAX)
        lorentzian_mean_per_slice.append(mu)
        lorentzian_curve_per_slice.append(peak_curve)
        if mu is not None and mu > 1e-6:
            t_pi_per_freq[i] = np.pi / (2.0 * np.pi * mu)

    # Restrict t_π to plausible range (10–500 ns) to reject noise-induced fits
    t_pi_min, t_pi_max = 10.0, 500.0
    valid = np.isfinite(t_pi_per_freq) & (t_pi_per_freq >= t_pi_min) & (t_pi_per_freq <= t_pi_max)
    f_res_est = float(freqs_hz.mean())
    omega_est = np.pi / 200.0
    rabi_curve = np.full_like(freqs_hz, np.nan)

    # Physics: t_π(Δ) = π/√(Ω² + δ²), δ = 2π·Δf_Hz×1e-9 rad/ns
    # Fit in detuning space to avoid f~1e10 scale; delta_res = f_res - f_center
    def _rabi_t_pi_detuning(delta_hz: np.ndarray, delta_res_hz: float, omega: float) -> np.ndarray:
        """t_π from detuning Δf_Hz (f - f_center); delta_res = f_res - f_center."""
        delta_rad_ns = 2.0 * np.pi * (np.asarray(delta_hz, dtype=float) - delta_res_hz) * 1e-9
        omega_R = np.sqrt(omega**2 + delta_rad_ns**2)
        return np.pi / np.maximum(omega_R, 1e-12)

    if np.sum(valid) >= 3:
        f_center = float(np.mean(freqs_hz))
        f_valid = np.asarray(freqs_hz[valid], dtype=float)
        delta_valid = f_valid - f_center  # detuning in Hz
        t_valid = np.asarray(t_pi_per_freq[valid], dtype=float)
        idx_min = int(np.argmin(t_valid))
        delta_res_init = float(delta_valid[idx_min])
        omega_init = np.pi / float(np.clip(t_valid[idx_min], 1.0, 1000.0))
        delta_span = float(max(np.ptp(delta_valid), 1e6))
        try:
            popt, _ = curve_fit(
                _rabi_t_pi_detuning,
                delta_valid,
                t_valid,
                p0=[delta_res_init, omega_init],
                bounds=(
                    [-delta_span, 1e-6],
                    [delta_span, np.pi / 5.0],
                ),
                maxfev=2000,
                method="trf",
            )
            delta_res_hz = float(popt[0])
            omega_est = float(popt[1])
            f_res_est = f_center + delta_res_hz
            delta_all = freqs_hz - f_center
            rabi_curve = _rabi_t_pi_detuning(delta_all, delta_res_hz, omega_est)
        except Exception:
            pass

    resonance_idx = int(np.argmin(np.abs(freqs_hz - f_res_est)))

    return {
        "t_pi_per_freq": t_pi_per_freq,
        "freqs_hz": freqs_hz,
        "resonance_idx": resonance_idx,
        "fft_freqs": freqs_fft,
        "magnitude_per_slice": magnitude_per_slice,
        "lorentzian_mean_per_slice": lorentzian_mean_per_slice,
        "lorentzian_curve_per_slice": lorentzian_curve_per_slice,
        "f_res_est": f_res_est,
        "omega_est": omega_est,
        "rabi_curve": rabi_curve,
        "parabola_curve": rabi_curve,  # deprecated, use rabi_curve
    }


def _estimate_f_res_and_omega_from_chevron(
    pdiff: np.ndarray,
    freqs_hz: np.ndarray,
    durations_ns: np.ndarray,
    nominal_freq_hz: float,
) -> Tuple[float, float]:
    """Estimate f_res and Ω from chevron: FFT → peak fit → t_π per slice → fit t_π(Δ)=π/√(Ω²+δ²)."""
    n_freqs, n_dur = pdiff.shape
    if n_freqs < 3 or n_dur < 4:
        return nominal_freq_hz, np.pi / 200.0

    diag = compute_fft_diagnostics(pdiff, freqs_hz, durations_ns)
    f_res_est = diag["f_res_est"]
    omega_est = diag["omega_est"]
    valid = np.isfinite(diag["t_pi_per_freq"]) & (diag["t_pi_per_freq"] > 4)
    if not np.any(valid) or not (freqs_hz.min() <= f_res_est <= freqs_hz.max()):
        return nominal_freq_hz, np.pi / 200.0
    return float(f_res_est), float(omega_est)
