r"""Heuristic chevron analysis: per-slice FFT → marginalised peak → hyperbola.

Strategy
--------
1. For each drive-detuning slice, compute the FFT and fit the spectral
   peak with a **Gaussian** (natural line-shape for a Gaussian pulse
   envelope).  Falls back to Lorentzian if the Gaussian fit fails.
2. Find **f_res** from the amplitude-weighted centroid of per-slice
   peak positions (robust even when off-resonance fits are noisy).
3. Extract **f_Ω** from a *marginalised* spectrum: average the top-K
   highest-amplitude FFT slices (near resonance) and fit a single
   Gaussian peak.  This avoids relying on scattered off-resonance
   peak positions.
4. Derive *t*\ :sub:`π` = π / Ω  and the analytical hyperbola
   *f*\ :sub:`Rabi`\(δ) = √(*f*\ :sub:`Ω`\ ² + δ²) for plotting.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import curve_fit

_logger = logging.getLogger(__name__)

FFT_FREQ_MIN = 0.0005  # 1/ns (exclude DC) → t_π max ~1000 ns
FFT_FREQ_MAX = 0.03  # 1/ns → t_π min ~17 ns; restricts to plausible Rabi range


# ── Peak line-shapes ──────────────────────────────────────────────────────────


def _lorentzian(x: np.ndarray, amp: float, x0: float, gamma: float) -> np.ndarray:
    """Lorentzian A / (1 + ((x - x0) / gamma)^2). Gamma is HWHM."""
    return amp / (1.0 + ((x - x0) / (gamma + 1e-12)) ** 2)


def _gaussian(x: np.ndarray, amp: float, x0: float, sigma: float) -> np.ndarray:
    """Gaussian amp * exp(-(x - x0)^2 / (2 sigma^2))."""
    return amp * np.exp(-((x - x0) ** 2) / (2.0 * (sigma + 1e-12) ** 2))


# ── Per-slice peak fitting ────────────────────────────────────────────────────


def _fit_peak_to_fft(
    freqs_fft: np.ndarray,
    magnitude: np.ndarray,
    freq_min: float,
    freq_max: float,
    model: str = "gaussian",
) -> Tuple[float | None, float | None, np.ndarray | None]:
    """Fit a peak model to the FFT magnitude spectrum.

    Returns ``(peak_freq, peak_amp, full_curve)`` or ``(None, None, None)``.
    *peak_amp* is the fitted amplitude at the peak (for weighting).
    """
    mask = (freqs_fft >= freq_min) & (freqs_fft <= freq_max)
    if np.sum(mask) < 4:
        return None, None, None
    f = freqs_fft[mask]
    m = magnitude[mask].astype(float)
    if np.max(m) < 1e-10:
        return None, None, None
    peak_idx = int(np.argmax(m))
    x0_0 = float(f[peak_idx])
    amp0 = float(m[peak_idx])
    width0 = max(1e-6, (f[-1] - f[0]) / 4.0)
    x0_0 = np.clip(x0_0, freq_min + width0, freq_max - width0)
    func = _gaussian if model == "gaussian" else _lorentzian
    try:
        popt, _ = curve_fit(
            func,
            f,
            m,
            p0=[amp0, x0_0, width0],
            bounds=([0, freq_min, 1e-6], [np.inf, freq_max, np.inf]),
            maxfev=500,
            method="trf",
        )
        x0_fit = float(popt[1])
        if x0_fit <= freq_min + 1e-8 or x0_fit >= freq_max - 1e-8:
            return None, None, None
        curve = func(freqs_fft, *popt)
        return x0_fit, float(popt[0]), curve
    except Exception:
        return None, None, None


# ── Main diagnostics entry-point ──────────────────────────────────────────────


def compute_fft_diagnostics(
    pdiff: np.ndarray,
    freqs_hz: np.ndarray,
    durations_ns: np.ndarray,
) -> Dict[str, Any]:
    """Per-slice FFT, Gaussian peak fit, and marginalised Ω extraction.

    Returns a dict with all diagnostic arrays needed by the plotting and
    analysis modules.
    """
    n_freqs, n_dur = pdiff.shape
    dt_ns = float(durations_ns[1] - durations_ns[0]) if len(durations_ns) > 1 else 1.0
    if dt_ns <= 0:
        dt_ns = 1.0

    freqs_fft = np.fft.rfftfreq(n_dur, dt_ns)

    # ── Per-slice FFT + peak detection ────────────────────────────────────
    peak_freq_per_slice = np.full(n_freqs, np.nan)  # cycles/ns
    peak_amp_per_slice = np.full(n_freqs, 0.0)  # for weighting
    t_pi_per_freq = np.full(n_freqs, np.nan)
    magnitude_per_slice: List[np.ndarray] = []
    peak_curve_per_slice: List[np.ndarray | None] = []

    for i in range(n_freqs):
        trace = np.asarray(pdiff[i, :], dtype=float) - np.mean(pdiff[i, :])
        magnitude_per_slice.append(np.abs(np.fft.rfft(trace)))

        if np.ptp(pdiff[i, :]) < 0.05:
            peak_curve_per_slice.append(None)
            continue

        mag = magnitude_per_slice[-1]

        # Prefer Gaussian (natural for Gaussian pulse envelope);
        # fall back to Lorentzian (natural for exponential decay).
        mu, amp, curve = _fit_peak_to_fft(freqs_fft, mag, FFT_FREQ_MIN, FFT_FREQ_MAX, "gaussian")
        if mu is None:
            mu, amp, curve = _fit_peak_to_fft(freqs_fft, mag, FFT_FREQ_MIN, FFT_FREQ_MAX, "lorentzian")

        peak_curve_per_slice.append(curve)
        if mu is not None and mu > 1e-6:
            peak_freq_per_slice[i] = mu
            peak_amp_per_slice[i] = amp if amp is not None else 0.0
            t_pi_per_freq[i] = 1.0 / (2.0 * mu)  # π / (2π·f_peak)

    # ==================================================================
    # Step 2 — Find f_res from amplitude-weighted peak positions
    # ==================================================================
    valid = np.isfinite(peak_freq_per_slice) & (peak_amp_per_slice > 0)
    f_center = float(np.mean(freqs_hz))
    f_res_est = f_center
    omega_est = np.pi / 200.0  # fallback
    f_omega_est = omega_est / (2.0 * np.pi)

    if np.sum(valid) >= 1:
        w = peak_amp_per_slice[valid]
        f_res_est = float(np.average(freqs_hz[valid], weights=w))

    resonance_idx = int(np.argmin(np.abs(freqs_hz - f_res_est)))

    # ==================================================================
    # Step 3 — Extract f_Ω from a marginalised spectrum
    # ==================================================================
    N_MARGINALISE = 7
    if np.sum(valid) >= 1:
        n_top = min(N_MARGINALISE, int(np.sum(valid)))
        top_idxs = np.argsort(peak_amp_per_slice)[-n_top:]
        marginalised = np.mean(
            np.array([magnitude_per_slice[i] for i in top_idxs], dtype=float),
            axis=0,
        )
        mu_m, _, _ = _fit_peak_to_fft(freqs_fft, marginalised, FFT_FREQ_MIN, FFT_FREQ_MAX, "gaussian")
        if mu_m is None:
            mu_m, _, _ = _fit_peak_to_fft(freqs_fft, marginalised, FFT_FREQ_MIN, FFT_FREQ_MAX, "lorentzian")
        if mu_m is not None and mu_m > 1e-6:
            f_omega_est = mu_m
            omega_est = 2.0 * np.pi * f_omega_est
            _logger.debug(
                "Marginalised FFT peak: f_Ω = %.5f cycles/ns (Ω = %.5f rad/ns, t_π = %.1f ns)",
                f_omega_est,
                omega_est,
                np.pi / omega_est,
            )

    # ==================================================================
    # Step 4 — Analytical hyperbola from marginalised f_Ω and f_res
    # ==================================================================
    # Both the 2-D FFT overlay (in frequency space) and the t_π-vs-
    # detuning curve (in time space) are derived from the SAME f_Ω and
    # f_res so they are guaranteed to be consistent.
    delta_all_hz = freqs_hz - f_res_est
    delta_cyc_ns = delta_all_hz * 1e-9
    ridge_curve_cyc_ns = np.sqrt(f_omega_est**2 + delta_cyc_ns**2)

    delta_rad_ns = 2.0 * np.pi * delta_cyc_ns
    omega_R = np.sqrt(omega_est**2 + delta_rad_ns**2)
    rabi_curve = np.pi / np.maximum(omega_R, 1e-12)

    t_pi_from_omega = np.pi / omega_est if omega_est > 1e-12 else np.nan
    _logger.debug(
        "Final estimates: f_res = %.3f GHz, Ω = %.5f rad/ns, t_π = %.1f ns",
        f_res_est * 1e-9,
        omega_est,
        t_pi_from_omega,
    )

    return {
        # Per-slice data
        "peak_freq_per_slice": peak_freq_per_slice,
        "peak_amp_per_slice": peak_amp_per_slice,
        "t_pi_per_freq": t_pi_per_freq,
        "fft_freqs": freqs_fft,
        "magnitude_per_slice": magnitude_per_slice,
        "peak_curve_per_slice": peak_curve_per_slice,
        # Fit results
        "freqs_hz": freqs_hz,
        "resonance_idx": resonance_idx,
        "f_res_est": f_res_est,
        "omega_est": omega_est,
        "ridge_curve_cyc_ns": ridge_curve_cyc_ns,
        "rabi_curve": rabi_curve,
        # Backward-compatible aliases
        "lorentzian_mean_per_slice": [float(f) if np.isfinite(f) else None for f in peak_freq_per_slice],
        "lorentzian_curve_per_slice": peak_curve_per_slice,
    }


def _estimate_f_res_and_omega_from_chevron(
    pdiff: np.ndarray,
    freqs_hz: np.ndarray,
    durations_ns: np.ndarray,
    nominal_freq_hz: float,
) -> Tuple[float, float, float]:
    """Estimate f_res, Ω, and γ from a Rabi chevron.

    Uses :func:`compute_fft_diagnostics` which performs:

    1. Per-slice Gaussian peak fit (FFT of each detuning row).
    2. Marginalised spectrum for robust f_Ω extraction.

    γ is estimated from the Lorentzian HWHM of the on-resonance FFT
    peak.  For Gaussian pulses the spectral width is dominated by
    the pulse shape; the γ estimate is therefore approximate.

    Returns
    -------
    f_res_est : float  (Hz)
    omega_est : float  (rad / ns)
    gamma_est : float  (1 / ns, decay rate)
    """
    n_freqs, n_dur = pdiff.shape
    if n_freqs < 3 or n_dur < 4:
        return nominal_freq_hz, np.pi / 200.0, 0.0

    diag = compute_fft_diagnostics(pdiff, freqs_hz, durations_ns)
    f_res_est = diag["f_res_est"]
    omega_est = diag["omega_est"]
    valid = np.isfinite(diag["t_pi_per_freq"]) & (diag["t_pi_per_freq"] > 4)
    if not np.any(valid) or not (freqs_hz.min() <= f_res_est <= freqs_hz.max()):
        return nominal_freq_hz, np.pi / 200.0, 0.0

    # ── γ estimate from Lorentzian HWHM at resonance ─────────────────────
    gamma_est = 0.0
    res_idx = diag.get("resonance_idx", n_freqs // 2)
    fft_freqs = diag.get("fft_freqs", None)
    mag_list = diag.get("magnitude_per_slice", [])

    if fft_freqs is not None and res_idx < len(mag_list):
        hwhm = _extract_lorentzian_hwhm(fft_freqs, mag_list[res_idx])
        if hwhm is not None:
            gamma_est = 2.0 * np.pi * hwhm
            _logger.debug(
                "FFT HWHM = %.6f cycles/ns → γ_est = %.6f /ns (T₂* ≈ %.0f ns)",
                hwhm,
                gamma_est,
                1.0 / gamma_est,
            )

    return float(f_res_est), float(omega_est), float(gamma_est)


def _extract_lorentzian_hwhm(
    freqs_fft: np.ndarray,
    magnitude: np.ndarray,
) -> float | None:
    """Fit Lorentzian to FFT magnitude and return HWHM (cycles/ns)."""
    _, _, curve = _fit_peak_to_fft(freqs_fft, magnitude, FFT_FREQ_MIN, FFT_FREQ_MAX, "lorentzian")
    if curve is None:
        return None
    mask = (freqs_fft >= FFT_FREQ_MIN) & (freqs_fft <= FFT_FREQ_MAX)
    f = freqs_fft[mask]
    m = magnitude[mask].astype(float)
    if len(f) < 4 or np.max(m) < 1e-10:
        return None
    peak_idx = int(np.argmax(m))
    try:
        popt, _ = curve_fit(
            _lorentzian,
            f,
            m,
            p0=[float(m[peak_idx]), float(f[peak_idx]), max(1e-6, (f[-1] - f[0]) / 4.0)],
            bounds=([0, FFT_FREQ_MIN, 1e-6], [np.inf, FFT_FREQ_MAX, np.inf]),
            maxfev=500,
        )
        hwhm = float(popt[2])
        if 1e-6 < hwhm < 0.05:
            return hwhm
    except Exception:
        pass
    return None
