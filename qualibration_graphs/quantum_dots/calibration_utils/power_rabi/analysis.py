"""Power-Rabi analysis: FFT-seeded damped-sinusoid fit.

Extracts the optimal amplitude prefactor (a_π) from a parity-
difference sweep over drive amplitude.  The FFT is used to seed a
damped-cosine fit in the amplitude domain, giving accurate estimates
of a_π and the decay rate.
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

# ── FFT frequency bounds (cycles / unit amplitude) ──────────────────────────
# Amplitude prefactor spans ~[0, 2] with step ~0.005 → Nyquist ~100 c/u.a.
# Physically, the Rabi oscillation completes one full cycle over an amplitude
# range of ~0.1–2, so the relevant frequency is roughly 0.5–10 c/u.a.
FFT_FREQ_MIN = 0.1  # exclude DC; a_π max ~ 5
FFT_FREQ_MAX = 20.0  # a_π min ~ 0.025


# ── Peak line-shapes ─────────────────────────────────────────────────────────


def _lorentzian(x: np.ndarray, amp: float, x0: float, gamma: float) -> np.ndarray:
    return amp / (1.0 + ((x - x0) / (gamma + 1e-12)) ** 2)


def _gaussian(x: np.ndarray, amp: float, x0: float, sigma: float) -> np.ndarray:
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


def _damped_sinusoid(a: np.ndarray, offset: float, amp: float, freq: float, gamma: float, phi: float) -> np.ndarray:
    """offset + amp * exp(-gamma * a) * cos(2π * freq * a + phi)."""
    return offset + amp * np.exp(-gamma * a) * np.cos(2.0 * np.pi * freq * a + phi)


def _fit_damped_sinusoid(
    amps: np.ndarray,
    trace: np.ndarray,
    fft_freq_seed: float,
    fft_gamma_seed: float,
) -> Dict[str, Any] | None:
    """Fit a damped sinusoid to the power-Rabi trace, seeded by FFT estimates."""
    a = amps - amps[0]  # shift so a starts at 0
    y = np.asarray(trace, dtype=float)

    offset0 = float(np.mean(y))
    amp0 = float(np.ptp(y)) / 2.0
    freq0 = fft_freq_seed
    gamma0 = max(fft_gamma_seed, 1e-6)

    # Phase seed from FFT
    n = len(a)
    da = float(a[1] - a[0]) if n > 1 else 1.0
    fft_complex = np.fft.rfft(y - offset0)
    fft_freqs = np.fft.rfftfreq(n, da)
    peak_idx = int(np.argmin(np.abs(fft_freqs - freq0)))
    phi0 = float(-np.angle(fft_complex[peak_idx]))

    a_span = float(a[-1] - a[0]) if len(a) > 1 else 1.0
    try:
        popt, pcov = curve_fit(
            _damped_sinusoid,
            a,
            y,
            p0=[offset0, amp0, freq0, gamma0, phi0],
            bounds=(
                [-np.inf, 0, FFT_FREQ_MIN, 0, -2 * np.pi],
                [np.inf, np.inf, FFT_FREQ_MAX, 50.0 / a_span, 2 * np.pi],
            ),
            maxfev=2000,
            method="trf",
        )
    except Exception as exc:
        _logger.debug("Damped sinusoid fit failed: %s", exc)
        return None

    offset_fit, amp_fit, freq_fit, gamma_fit, phi_fit = popt
    perr = np.sqrt(np.diag(pcov))

    if perr[2] > 0.5 * abs(freq_fit):
        _logger.debug("Damped sinusoid fit rejected: freq uncertainty too large")
        return None

    fitted_curve = _damped_sinusoid(a, *popt)

    return {
        "offset": float(offset_fit),
        "amplitude": float(amp_fit),
        "frequency": float(freq_fit),  # cycles per unit amplitude
        "gamma": float(gamma_fit),
        "phase": float(phi_fit),
        "fitted_curve": fitted_curve,
        "a_shifted": a,
    }


# ── Fit result dataclass ─────────────────────────────────────────────────────


@dataclass
class FitParameters:
    """Fit parameters for a single qubit from a power-Rabi sweep."""

    opt_amp: float  # optimal amplitude prefactor (π-pulse)
    rabi_frequency: float  # rad per unit amplitude
    decay_rate: float  # 1 / unit amplitude
    success: bool


# ── Single-qubit analysis ────────────────────────────────────────────────────


def _analyse_single_qubit(
    pdiff_1d: np.ndarray,
    amps: np.ndarray,
) -> Dict[str, Any]:
    """Analyse one qubit's power-Rabi trace: FFT seed → damped sinusoid fit."""
    n = len(amps)
    da = float(amps[1] - amps[0]) if n > 1 else 1.0
    if da <= 0:
        da = 1.0

    trace = np.asarray(pdiff_1d, dtype=float)
    trace_centered = trace - np.mean(trace)

    freqs_fft = np.fft.rfftfreq(n, da)
    magnitude = np.abs(np.fft.rfft(trace_centered))

    # ── Step 1: FFT peak detection (seed) ────────────────────────────────
    mu, amp_peak, peak_curve = _fit_peak_to_fft(freqs_fft, magnitude, FFT_FREQ_MIN, FFT_FREQ_MAX, "gaussian")
    if mu is None:
        mu, amp_peak, peak_curve = _fit_peak_to_fft(freqs_fft, magnitude, FFT_FREQ_MIN, FFT_FREQ_MAX, "lorentzian")

    if mu is None or mu < 1e-6:
        return {
            "opt_amp": np.nan,
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
            if 1e-6 < hwhm < 5.0:
                gamma_seed = 2.0 * np.pi * hwhm
        except Exception:
            pass

    # ── Step 2: Damped sinusoid fit ──────────────────────────────────────
    sinusoid_result = _fit_damped_sinusoid(amps, trace, mu, gamma_seed)

    if sinusoid_result is not None:
        f_rabi = sinusoid_result["frequency"]  # cycles per unit amp
        omega = 2.0 * np.pi * f_rabi  # rad per unit amp
        a_pi = 1.0 / (2.0 * f_rabi)  # half-period = π-pulse prefactor
        gamma = sinusoid_result["gamma"]
        _logger.debug(
            "Power Rabi damped-sinusoid: f = %.3f c/u.a., a_π = %.4f, γ = %.4f",
            f_rabi,
            a_pi,
            gamma,
        )
    else:
        f_rabi = mu
        omega = 2.0 * np.pi * f_rabi
        a_pi = 1.0 / (2.0 * f_rabi)
        gamma = gamma_seed
        _logger.debug(
            "Power Rabi FFT fallback: f = %.3f c/u.a., a_π = %.4f, γ = %.4f",
            f_rabi,
            a_pi,
            gamma,
        )

    success = np.isfinite(a_pi) and a_pi > 0

    return {
        "opt_amp": float(a_pi),
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
    """Fit optimal amplitude per qubit from power-Rabi data."""
    qubits = node.namespace["qubits"]
    pdiff_vars = [v for v in ds.data_vars if v.startswith("pdiff_")]
    qubit_names = [v.replace("pdiff_", "") for v in sorted(pdiff_vars)]
    if not qubit_names:
        qubit_names = [getattr(q, "name", f"Q{i}") for i, q in enumerate(qubits)]

    amps = np.asarray(ds.amp_prefactor.values, dtype=float)

    fit_results: Dict[str, Dict[str, Any]] = {}

    for qname in qubit_names:
        pdiff_var = f"pdiff_{qname}"
        if pdiff_var not in ds.data_vars:
            fp = FitParameters(
                opt_amp=np.nan,
                rabi_frequency=np.nan,
                decay_rate=np.nan,
                success=False,
            )
            fit_results[qname] = asdict(fp)
            continue

        pdiff = np.asarray(ds[pdiff_var].values, dtype=float)
        result = _analyse_single_qubit(pdiff, amps)

        fp = FitParameters(
            opt_amp=result["opt_amp"],
            rabi_frequency=result["rabi_frequency"],
            decay_rate=result["decay_rate"],
            success=result["success"],
        )
        fit_results[qname] = asdict(fp)

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
        a_pi = r.get("opt_amp", 0)
        omega = r.get("rabi_frequency", 0)
        gamma = r.get("decay_rate", 0)
        success = r.get("success", False)
        msg = (
            f"Results for {qname}: "
            f"a_π={a_pi:.4f}, "
            f"Ω={omega:.3f} rad/u.a., "
            f"γ={gamma:.4f} 1/u.a., "
            f"success={success}"
        )
        log_callable(msg)
