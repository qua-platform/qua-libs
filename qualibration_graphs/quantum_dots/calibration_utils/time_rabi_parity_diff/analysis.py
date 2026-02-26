"""1D time-Rabi analysis: FFT-seeded damped-sinusoid fit.

Extracts t_π (= π/Ω) and γ (decay rate ≈ 1/T₂*) from a single
pulse-duration sweep.  The FFT is used to seed a time-domain
least-squares fit of a damped cosine, giving more accurate
estimates than the FFT alone.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from qualibrate import QualibrationNode

_logger = logging.getLogger(__name__)

# ── FFT frequency bounds (cycles / ns) ──────────────────────────────────────
FFT_FREQ_MIN = 0.0005  # exclude DC; t_π max ~ 1000 ns
FFT_FREQ_MAX = 0.03  # t_π min ~ 17 ns


# ── Peak line-shapes ─────────────────────────────────────────────────────────


def _lorentzian(x: np.ndarray, amp: float, x0: float, gamma: float) -> np.ndarray:
    """Lorentzian A / (1 + ((x - x0) / gamma)^2).  gamma is HWHM."""
    return amp / (1.0 + ((x - x0) / (gamma + 1e-12)) ** 2)


def _gaussian(x: np.ndarray, amp: float, x0: float, sigma: float) -> np.ndarray:
    """Gaussian amp * exp(-(x - x0)^2 / (2 sigma^2))."""
    return amp * np.exp(-((x - x0) ** 2) / (2.0 * (sigma + 1e-12) ** 2))


def _fit_peak_to_fft(
    freqs_fft: np.ndarray,
    magnitude: np.ndarray,
    freq_min: float,
    freq_max: float,
    model: str = "gaussian",
) -> Tuple[float | None, float | None, np.ndarray | None]:
    """Fit a peak model to the FFT magnitude spectrum.

    Returns ``(peak_freq, peak_amp, full_curve)`` or ``(None, None, None)``.
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


# ── Damped-sinusoid model ────────────────────────────────────────────────────


def _damped_sinusoid(t: np.ndarray, offset: float, amp: float, freq: float, gamma: float, phi: float) -> np.ndarray:
    """offset + amp * exp(-gamma * t) * cos(2π * freq * t + phi)."""
    return offset + amp * np.exp(-gamma * t) * np.cos(2.0 * np.pi * freq * t + phi)


def _fit_damped_sinusoid(
    durations_ns: np.ndarray,
    trace: np.ndarray,
    fft_freq_seed: float,
    fft_gamma_seed: float,
) -> Dict[str, Any] | None:
    """Fit a damped sinusoid to the Rabi trace, seeded by FFT estimates.

    Returns dict with fitted parameters or ``None`` if the fit fails.
    """
    t = durations_ns - durations_ns[0]  # shift so t starts at 0
    y = np.asarray(trace, dtype=float)

    # Seed estimates
    offset0 = float(np.mean(y))
    amp0 = float(np.ptp(y)) / 2.0
    freq0 = fft_freq_seed  # cycles/ns
    gamma0 = max(fft_gamma_seed, 1e-6)

    # Estimate phase from FFT complex coefficient at peak frequency
    n = len(t)
    dt = float(t[1] - t[0]) if n > 1 else 1.0
    fft_complex = np.fft.rfft(y - offset0)
    fft_freqs = np.fft.rfftfreq(n, dt)
    peak_idx = int(np.argmin(np.abs(fft_freqs - freq0)))
    phi0 = float(-np.angle(fft_complex[peak_idx]))

    t_span = float(t[-1] - t[0]) if len(t) > 1 else 1.0
    try:
        popt, pcov = curve_fit(
            _damped_sinusoid,
            t,
            y,
            p0=[offset0, amp0, freq0, gamma0, phi0],
            bounds=(
                [-np.inf, 0, FFT_FREQ_MIN, 0, -2 * np.pi],
                [np.inf, np.inf, FFT_FREQ_MAX, 10.0 / t_span, 2 * np.pi],
            ),
            maxfev=2000,
            method="trf",
        )
    except Exception as exc:
        _logger.debug("Damped sinusoid fit failed: %s", exc)
        return None

    offset_fit, amp_fit, freq_fit, gamma_fit, phi_fit = popt
    perr = np.sqrt(np.diag(pcov))

    # Reject if frequency uncertainty is too large relative to frequency
    if perr[2] > 0.5 * abs(freq_fit):
        _logger.debug("Damped sinusoid fit rejected: freq uncertainty %.4e > 50%% of freq %.4e", perr[2], freq_fit)
        return None

    fitted_curve = _damped_sinusoid(t, *popt)

    return {
        "offset": float(offset_fit),
        "amplitude": float(amp_fit),
        "frequency": float(freq_fit),  # cycles/ns
        "gamma": float(gamma_fit),  # 1/ns
        "phase": float(phi_fit),
        "fitted_curve": fitted_curve,
        "t_shifted": t,
    }


# ── Fit result dataclass ─────────────────────────────────────────────────────


@dataclass
class FitParameters:
    """Fit parameters for a single qubit from a 1D time-Rabi sweep."""

    optimal_duration: float  # ns (π-time)
    rabi_frequency: float  # rad/ns
    decay_rate: float  # 1/ns  (γ ≈ 1/T₂*)
    success: bool


# ── Single-qubit 1D analysis ─────────────────────────────────────────────────


def _fft_analyse_single_qubit(
    pdiff_1d: np.ndarray,
    durations_ns: np.ndarray,
) -> Dict[str, Any]:
    """Analyse one qubit's 1D Rabi trace: FFT seed → damped sinusoid fit.

    Uses the FFT to find the oscillation frequency, then fits a damped
    cosine in the time domain for more accurate parameter extraction.
    Falls back to FFT-only results if the time-domain fit fails.
    """
    n_dur = len(durations_ns)
    dt_ns = float(durations_ns[1] - durations_ns[0]) if n_dur > 1 else 1.0
    if dt_ns <= 0:
        dt_ns = 1.0

    trace = np.asarray(pdiff_1d, dtype=float)
    trace_centered = trace - np.mean(trace)

    freqs_fft = np.fft.rfftfreq(n_dur, dt_ns)
    magnitude = np.abs(np.fft.rfft(trace_centered))

    # ── Step 1: FFT peak detection (seed) ────────────────────────────────
    mu, amp, peak_curve = _fit_peak_to_fft(freqs_fft, magnitude, FFT_FREQ_MIN, FFT_FREQ_MAX, "gaussian")
    if mu is None:
        mu, amp, peak_curve = _fit_peak_to_fft(freqs_fft, magnitude, FFT_FREQ_MIN, FFT_FREQ_MAX, "lorentzian")

    if mu is None or mu < 1e-6:
        return {
            "optimal_duration": np.nan,
            "rabi_frequency": np.nan,
            "decay_rate": np.nan,
            "success": False,
            "fft_freqs": freqs_fft,
            "fft_magnitude": magnitude,
            "peak_curve": None,
            "sinusoid_fit": None,
        }

    # Rough γ seed from Lorentzian HWHM
    gamma_seed = 0.0
    mask = (freqs_fft >= FFT_FREQ_MIN) & (freqs_fft <= FFT_FREQ_MAX)
    f_masked = freqs_fft[mask]
    m_masked = magnitude[mask].astype(float)
    if len(f_masked) >= 4 and np.max(m_masked) > 1e-10:
        peak_idx = int(np.argmax(m_masked))
        try:
            popt_lor, _ = curve_fit(
                _lorentzian,
                f_masked,
                m_masked,
                p0=[
                    float(m_masked[peak_idx]),
                    float(f_masked[peak_idx]),
                    max(1e-6, (f_masked[-1] - f_masked[0]) / 4.0),
                ],
                bounds=([0, FFT_FREQ_MIN, 1e-6], [np.inf, FFT_FREQ_MAX, np.inf]),
                maxfev=500,
            )
            hwhm = float(popt_lor[2])
            if 1e-6 < hwhm < 0.05:
                gamma_seed = 2.0 * np.pi * hwhm
        except Exception:
            pass

    # ── Step 2: Damped sinusoid fit (time-domain) ────────────────────────
    sinusoid_result = _fit_damped_sinusoid(durations_ns, trace, mu, gamma_seed)

    if sinusoid_result is not None:
        f_rabi = sinusoid_result["frequency"]  # cycles/ns
        omega = 2.0 * np.pi * f_rabi  # rad/ns
        t_pi = 1.0 / (2.0 * f_rabi)  # ns
        gamma = sinusoid_result["gamma"]  # 1/ns (direct from envelope)
        _logger.debug(
            "1D Rabi damped-sinusoid fit: f = %.5f cycles/ns, Ω = %.5f rad/ns, " "t_π = %.1f ns, γ = %.6f /ns",
            f_rabi,
            omega,
            t_pi,
            gamma,
        )
    else:
        # Fall back to FFT-only estimates
        f_rabi = mu
        omega = 2.0 * np.pi * f_rabi
        t_pi = 1.0 / (2.0 * f_rabi)
        gamma = gamma_seed
        _logger.debug(
            "1D Rabi FFT fallback: f = %.5f cycles/ns, Ω = %.5f rad/ns, " "t_π = %.1f ns, γ = %.6f /ns",
            f_rabi,
            omega,
            t_pi,
            gamma,
        )

    success = np.isfinite(t_pi) and t_pi > 0

    return {
        "optimal_duration": float(t_pi),
        "rabi_frequency": float(omega),
        "decay_rate": float(gamma),
        "success": success,
        "fft_freqs": freqs_fft,
        "fft_magnitude": magnitude,
        "peak_curve": peak_curve,
        "sinusoid_fit": sinusoid_result,
    }


# ── Public API ────────────────────────────────────────────────────────────────


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, Dict[str, Any]]]:
    """Fit t_π per qubit from 1D time-Rabi data.  Returns ``(ds_fit, fit_results)``."""
    qubits = node.namespace["qubits"]
    # Resolve qubit names from dataset
    pdiff_vars = [v for v in ds.data_vars if v.startswith("pdiff_")]
    qubit_names = [v.replace("pdiff_", "") for v in sorted(pdiff_vars)]
    if not qubit_names:
        qubit_names = [getattr(q, "name", f"Q{i}") for i, q in enumerate(qubits)]

    durations_ns = np.asarray(ds.pulse_duration.values, dtype=float)

    fit_results: Dict[str, Dict[str, Any]] = {}
    fit_arrays: Dict[str, Any] = {}

    for qname in qubit_names:
        pdiff_var = f"pdiff_{qname}"
        if pdiff_var not in ds.data_vars:
            fp = FitParameters(
                optimal_duration=np.nan,
                rabi_frequency=np.nan,
                decay_rate=np.nan,
                success=False,
            )
            fit_results[qname] = asdict(fp)
            continue

        pdiff = np.asarray(ds[pdiff_var].values, dtype=float)
        result = _fft_analyse_single_qubit(pdiff, durations_ns)

        fp = FitParameters(
            optimal_duration=result["optimal_duration"],
            rabi_frequency=result["rabi_frequency"],
            decay_rate=result["decay_rate"],
            success=result["success"],
        )
        fit_results[qname] = asdict(fp)

        # Store diagnostics for plotting
        fit_results[qname]["_fft_diag"] = {
            "fft_freqs": result["fft_freqs"],
            "fft_magnitude": result["fft_magnitude"],
            "peak_curve": result["peak_curve"],
        }
        fit_results[qname]["_sinusoid_fit"] = result.get("sinusoid_fit")

    ds_fit = ds.copy()
    return ds_fit, fit_results


def log_fitted_results(
    fit_results: Dict[str, Any],
    log_callable=None,
) -> None:
    """Log fitted results for all qubits."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for qname, r in fit_results.items():
        t_pi = r.get("optimal_duration", 0)
        omega = r.get("rabi_frequency", 0)
        gamma = r.get("decay_rate", 0)
        t2_star = 1.0 / gamma if gamma > 0 else float("inf")
        success = r.get("success", False)
        msg = (
            f"Results for {qname}: "
            f"t_π={t_pi:.0f} ns, "
            f"Ω={omega:.5f} rad/ns, "
            f"γ={gamma:.5f} 1/ns (T₂*={t2_star:.0f} ns), "
            f"success={success}"
        )
        log_callable(msg)
