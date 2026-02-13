"""Heuristic chevron analysis: per-slice FFT → amplitude-weighted ridge fit.

Strategy
--------
1. For each drive-detuning slice, compute the FFT and fit the spectral
   peak with a **Gaussian** (natural line-shape for a Gaussian pulse
   envelope).  Falls back to Lorentzian if the Gaussian fit fails.
2. Collect the per-slice peak frequencies into a *ridge* in the 2-D
   FFT image.
3. Fit the hyperbolic model

   .. math:: f_{\\mathrm{Rabi}}(\\Delta) = \\sqrt{f_\\Omega^2 + \\Delta^2}

   to the ridge, **weighted by peak amplitude** so that high-SNR
   slices near resonance dominate.
4. Derive *f*\\ :sub:`res` (detuning where *f*\\ :sub:`Rabi` is
   minimal), Ω = 2π *f*\\ :sub:`Ω`, and *t*\\ :sub:`π` = π / Ω.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import curve_fit

_logger = logging.getLogger(__name__)

FFT_FREQ_MIN = 0.0005  # 1/ns (exclude DC) → t_π max ~1000 ns
FFT_FREQ_MAX = 0.03  # 1/ns → t_π min ~17 ns; restricts to plausible Rabi range
FFT_ZERO_PAD_FACTOR = 16  # zero-pad FFT for smoother Lorentzian fits


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


def _fit_lorentzian_to_fft(
    freqs_fft: np.ndarray,
    magnitude: np.ndarray,
    freq_min: float,
    freq_max: float,
) -> Tuple[float | None, np.ndarray | None]:
    """Backward-compatible wrapper — returns ``(peak_freq, curve)``."""
    mu, _, curve = _fit_peak_to_fft(freqs_fft, magnitude, freq_min, freq_max, model="lorentzian")
    return mu, curve


def _fit_gaussian_to_fft(
    freqs_fft: np.ndarray,
    magnitude: np.ndarray,
    freq_min: float,
    freq_max: float,
) -> Tuple[float | None, np.ndarray | None]:
    """Backward-compatible wrapper — returns ``(peak_freq, curve)``."""
    mu, _, curve = _fit_peak_to_fft(freqs_fft, magnitude, freq_min, freq_max, model="gaussian")
    return mu, curve


# ── Hyperbolic ridge model ────────────────────────────────────────────────────


def _rabi_freq_ridge(
    delta_hz: np.ndarray,
    f_omega: float,
    delta_res_hz: float,
) -> np.ndarray:
    r"""Generalised Rabi frequency as a function of drive detuning.

    .. math:: f_{\mathrm{Rabi}}(\Delta) = \sqrt{f_\Omega^2 + \Delta^2}

    Parameters
    ----------
    delta_hz : array
        Drive detuning from *f*\ :sub:`center` (Hz).
    f_omega : float
        Bare Rabi frequency (cycles / ns).
    delta_res_hz : float
        Resonance offset from *f*\ :sub:`center` (Hz).

    Returns
    -------
    f_rabi : array
        Generalised Rabi frequency (cycles / ns).
    """
    delta_cyc_ns = (np.asarray(delta_hz, dtype=float) - delta_res_hz) * 1e-9
    return np.sqrt(f_omega**2 + delta_cyc_ns**2)


# ── Main diagnostics entry-point ──────────────────────────────────────────────


def compute_fft_diagnostics(
    pdiff: np.ndarray,
    freqs_hz: np.ndarray,
    durations_ns: np.ndarray,
) -> Dict[str, Any]:
    """Per-slice FFT, Gaussian peak fit, and amplitude-weighted ridge fit.

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
    # The resonance frequency is the detuning where the Rabi signal is
    # strongest.  An amplitude-weighted centroid of the per-slice peak
    # positions is robust even when off-resonance fits are noisy.
    valid = np.isfinite(peak_freq_per_slice) & (peak_amp_per_slice > 0)
    f_center = float(np.mean(freqs_hz))
    f_res_est = f_center
    omega_est = np.pi / 200.0  # fallback
    f_omega_est = omega_est / (2.0 * np.pi)
    ridge_curve_cyc_ns = np.full(n_freqs, np.nan)
    rabi_curve = np.full_like(freqs_hz, np.nan, dtype=float)

    if np.sum(valid) >= 1:
        w = peak_amp_per_slice[valid]
        f_res_est = float(np.average(freqs_hz[valid], weights=w))

    resonance_idx = int(np.argmin(np.abs(freqs_hz - f_res_est)))

    # ==================================================================
    # Step 3 — Extract f_Ω from a marginalised spectrum
    # ==================================================================
    # Average the FFT magnitudes of the top-K highest-amplitude slices
    # (concentrated near resonance) to get a high-SNR 1-D spectrum.
    # A single Gaussian peak fit on this gives f_Ω directly, without
    # relying on the shape of the off-resonance ridge.
    N_MARGINALISE = 7  # number of top slices to average
    if np.sum(valid) >= 1:
        n_top = min(N_MARGINALISE, int(np.sum(valid)))
        top_idxs = np.argsort(peak_amp_per_slice)[-n_top:]
        marginalised = np.mean(
            np.array([magnitude_per_slice[i] for i in top_idxs], dtype=float),
            axis=0,
        )
        mu_m, _, curve_m = _fit_peak_to_fft(freqs_fft, marginalised, FFT_FREQ_MIN, FFT_FREQ_MAX, "gaussian")
        if mu_m is None:
            mu_m, _, curve_m = _fit_peak_to_fft(freqs_fft, marginalised, FFT_FREQ_MIN, FFT_FREQ_MAX, "lorentzian")
        if mu_m is not None and mu_m > 1e-6:
            f_omega_est = mu_m
            omega_est = 2.0 * np.pi * f_omega_est
            _logger.debug(
                "Marginalised FFT peak: f_Ω = %.5f cycles/ns " "(Ω = %.5f rad/ns, t_π = %.1f ns)",
                f_omega_est,
                omega_est,
                np.pi / omega_est,
            )

    # ==================================================================
    # Step 4 — Ridge fit for visualisation only
    # ==================================================================
    # Fit the hyperbolic model f_peak(δ) = √(f_Ω² + δ²) to the valid
    # per-slice peaks, seeded with f_Ω from the marginalised fit.
    # This is ONLY used for the overlay curve on the 2-D FFT plot.
    if np.sum(valid) >= 3:
        f_valid = peak_freq_per_slice[valid]
        delta_valid_hz = freqs_hz[valid] - f_center
        weights = peak_amp_per_slice[valid].copy()
        weights /= np.max(weights) + 1e-12

        delta_res_init = float(f_res_est - f_center)
        delta_span = float(max(np.ptp(freqs_hz), 1e6))

        try:
            popt, _ = curve_fit(
                _rabi_freq_ridge,
                delta_valid_hz,
                f_valid,
                p0=[f_omega_est, delta_res_init],
                sigma=1.0 / (weights + 0.01),
                absolute_sigma=False,
                bounds=([1e-6, -delta_span], [FFT_FREQ_MAX, delta_span]),
                maxfev=2000,
                method="trf",
            )
            delta_all_hz = freqs_hz - f_center
            ridge_curve_cyc_ns = _rabi_freq_ridge(delta_all_hz, *popt)
        except Exception:
            _logger.debug("Ridge fit (visualisation) failed.")

    # Derive t_π curve from f_Ω and the hyperbolic model
    t_pi_from_omega = np.pi / omega_est if omega_est > 1e-12 else np.nan
    delta_all_hz = freqs_hz - f_res_est
    delta_rad_ns = 2.0 * np.pi * delta_all_hz * 1e-9
    omega_R = np.sqrt(omega_est**2 + delta_rad_ns**2)
    rabi_curve = np.pi / np.maximum(omega_R, 1e-12)

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
        "parabola_curve": rabi_curve,
    }


def _estimate_f_res_and_omega_from_chevron(
    pdiff: np.ndarray,
    freqs_hz: np.ndarray,
    durations_ns: np.ndarray,
    nominal_freq_hz: float,
) -> Tuple[float, float, float]:
    """Estimate f_res, Ω, and γ from a Rabi chevron via ridge fitting.

    Uses :func:`compute_fft_diagnostics` which performs:

    1. Per-slice Gaussian peak fit (FFT of each detuning row).
    2. Amplitude-weighted hyperbolic ridge fit in frequency space.

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
    # An exponentially-damped sinusoid has Lorentzian FFT with HWHM = γ/(2π).
    # For Gaussian pulses the spectral width is dominated by 1/(2πσ), so
    # this estimate is an upper bound on γ.
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
    mu, _, curve = _fit_peak_to_fft(freqs_fft, magnitude, FFT_FREQ_MIN, FFT_FREQ_MAX, "lorentzian")
    if mu is None or curve is None:
        return None
    # Recover HWHM by re-fitting (we could also extract from _fit_peak_to_fft
    # but keeping the existing sanity bounds is cleaner).
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
