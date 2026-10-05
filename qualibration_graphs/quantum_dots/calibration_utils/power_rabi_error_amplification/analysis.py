"""Error-amplified power-Rabi analysis: mean-signal resonance finding."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit, differential_evolution

from qualibrate.core import QualibrationNode

_logger = logging.getLogger(__name__)


def _make_mean_rabi_model(n_pulses: np.ndarray):
    n = np.asarray(n_pulses, dtype=float)

    def _model(x, amp, x0, gamma, sigma_g, bg, scale):
        da = np.atleast_1d(np.asarray(x, dtype=float)) - x0
        phase = 2.0 * np.pi * da[:, None] * n[None, :] * scale
        envelope = np.exp(-gamma * n - (sigma_g * n) ** 2)[None, :]
        result = bg + amp * np.mean(envelope * np.cos(phase), axis=1)
        return result if np.ndim(x) > 0 else float(result[0])

    return _model


def _effective_n_eff(gamma: float, sigma_g: float) -> float:
    if sigma_g < 1e-12:
        return 1.0 / gamma if gamma > 1e-12 else np.nan
    discriminant = gamma**2 + 4.0 * sigma_g**2
    return (-gamma + np.sqrt(discriminant)) / (2.0 * sigma_g**2)


@dataclass
class FitParameters:
    opt_amp: float
    rabi_frequency: float
    decay_rate: float
    gauss_decay_rate: float
    n_eff: float
    success: bool


def _fit_exponential_decay(
    n_pulses: np.ndarray,
    trace: np.ndarray,
    n_span: float,
) -> float | None:
    t = n_pulses - n_pulses[0]
    y = np.asarray(trace, dtype=float)

    bg0 = float(np.mean(y[-len(y) // 4 :]))
    amp0 = float(y[0]) - bg0
    if abs(amp0) < 0.01:
        return None
    gamma0 = 2.0 / n_span

    def _model(t, bg, amp, gamma):
        return bg + amp * np.exp(-gamma * t)

    try:
        popt, pcov = curve_fit(
            _model,
            t,
            y,
            p0=[bg0, amp0, gamma0],
            bounds=([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, 10.0 / n_span]),
            maxfev=3000,
        )
        perr = np.sqrt(np.diag(pcov))
        gamma = float(popt[2])
        if gamma < 2e-6 or (perr[2] > gamma * 5):
            return None
        return gamma
    except Exception:
        return None


def _validate_n_eff(
    signal_2d: np.ndarray,
    n_pulses: np.ndarray,
    opt_amp: float,  # noqa: ARG001
    amps: np.ndarray,  # noqa: ARG001
    resonance_idx: int,
    decay_rate_de: float,
    sigma_g_de: float,
    n_eff_de: float,
) -> tuple[float, float, float]:
    n_span = float(n_pulses[-1] - n_pulses[0]) if len(n_pulses) > 1 else 1.0

    if np.isfinite(n_eff_de) and 0 < n_eff_de < 5.0 * n_span:
        return decay_rate_de, sigma_g_de, n_eff_de

    n_amp = signal_2d.shape[1]
    half_w = max(1, n_amp // 20)
    lo = max(0, resonance_idx - half_w)
    hi = min(n_amp, resonance_idx + half_w + 1)
    near_trace = np.mean(signal_2d[:, lo:hi], axis=1)
    gamma_exp = _fit_exponential_decay(n_pulses, near_trace, n_span)
    if gamma_exp is not None:
        return gamma_exp, 0.0, 1.0 / gamma_exp

    return decay_rate_de, sigma_g_de, n_eff_de


def _analyse_single_qubit(
    signal_2d: np.ndarray,
    amps: np.ndarray,
    n_pulses: np.ndarray,
) -> Dict[str, Any]:
    n_np = signal_2d.shape[0]

    mean_signal = np.mean(signal_2d, axis=0)
    model = _make_mean_rabi_model(n_pulses)

    median_val = float(np.median(mean_signal))
    abs_dev = np.abs(mean_signal - median_val)
    extremum_idx = int(np.argmax(abs_dev))

    opt_amp = float(amps[extremum_idx])
    resonance_idx = extremum_idx
    mean_signal_fit = None
    decay_rate = np.nan
    sigma_g = 0.0
    n_eff = np.nan
    scale = np.nan
    de_converged = False

    try:
        ptp = float(np.ptp(mean_signal))
        amp_min, amp_max = float(amps.min()), float(amps.max())
        n_span = float(n_pulses[-1] - n_pulses[0]) if n_np > 1 else 1.0
        y_min, y_max = float(mean_signal.min()), float(mean_signal.max())
        amp_range = amp_max - amp_min

        n_max = float(n_pulses[-1])
        scale_min = 0.1 / (n_max * amp_range) if n_max * amp_range > 0 else 0.01
        scale_max = 50.0 / (float(n_pulses[0]) * amp_range) if float(n_pulses[0]) * amp_range > 0 else 100.0

        extremum_sign = float(np.sign(mean_signal[extremum_idx] - median_val))
        if extremum_sign > 0:
            amp_de_bounds = (0.0, ptp * 3)
        elif extremum_sign < 0:
            amp_de_bounds = (-ptp * 3, 0.0)
        else:
            amp_de_bounds = (-ptp * 3, ptp * 3)

        de_bounds = [
            amp_de_bounds,
            (amp_min, amp_max),
            (0.0, 10.0 / n_span),
            (0.0, 10.0 / n_span),
            (y_min - ptp, y_max + ptp),
            (scale_min, scale_max),
        ]

        def _objective(params):
            return np.sum((model(amps, *params) - mean_signal) ** 2)

        de_result = differential_evolution(
            _objective,
            de_bounds,
            seed=42,
            maxiter=2000,
            tol=1e-10,
            polish=True,
            popsize=25,
        )
        popt = de_result.x
        opt_amp = float(popt[1])
        decay_rate = float(popt[2])
        sigma_g = float(popt[3])
        scale = float(popt[5])
        resonance_idx = int(np.argmin(np.abs(amps - opt_amp)))
        n_eff = _effective_n_eff(decay_rate, sigma_g)
        mean_signal_fit = model(amps, *popt)
        de_converged = True
        _logger.debug(
            "Error-amp Rabi mean-signal fit (DE): a_π=%.4f, "
            "scale=%.4f c/u.a./pulse, gamma=%.5f, sigma_g=%.5f, N_eff=%.1f",
            opt_amp,
            scale,
            decay_rate,
            sigma_g,
            n_eff if np.isfinite(n_eff) else -1,
        )
    except Exception:
        _logger.debug(
            "Mean-signal fit failed; using raw extremum at a=%.4f",
            opt_amp,
        )

    decay_rate, sigma_g, n_eff = _validate_n_eff(
        signal_2d,
        n_pulses,
        opt_amp,
        amps,
        resonance_idx,
        decay_rate,
        sigma_g,
        n_eff,
    )

    amp_min, amp_max = float(amps.min()), float(amps.max())
    rabi_frequency = 2.0 * np.pi * scale if np.isfinite(scale) else np.nan
    success = bool(de_converged and np.isfinite(opt_amp) and amp_min <= opt_amp <= amp_max)

    return {
        "opt_amp": opt_amp,
        "rabi_frequency": float(rabi_frequency),
        "decay_rate": float(decay_rate),
        "gauss_decay_rate": float(sigma_g),
        "n_eff": float(n_eff),
        "success": success,
        "_diag": {
            "mean_signal": mean_signal,
            "mean_signal_fit": mean_signal_fit,
            "resonance_idx": resonance_idx,
        },
    }


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Return ``ds_raw`` unchanged (thresholded ``state`` needs no stream post-processing)."""
    return ds


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, Dict[str, Any]]]:
    """Fit optimal amplitude per qubit from error-amplified power-Rabi ``state`` data."""
    qubit_names = [str(v) for v in ds.qubit.values]
    amps = np.asarray(ds.amp_prefactor.values, dtype=float)
    n_pulses_array = np.asarray(ds.n_pulses.values, dtype=float)

    fit_results: Dict[str, Dict[str, Any]] = {}
    mean_curves: Dict[str, np.ndarray] = {}
    mean_fit_curves: Dict[str, np.ndarray] = {}

    for qname in qubit_names:
        signal_2d = ds.state.sel(qubit=qname, drop=True).transpose("n_pulses", "amp_prefactor").values.astype(float)
        result = _analyse_single_qubit(signal_2d, amps, n_pulses_array)

        fp = FitParameters(
            opt_amp=result["opt_amp"],
            rabi_frequency=result["rabi_frequency"],
            decay_rate=result["decay_rate"],
            gauss_decay_rate=result["gauss_decay_rate"],
            n_eff=result["n_eff"],
            success=result["success"],
        )
        fit_results[qname] = asdict(fp)

        diag = result.get("_diag", {})
        mean_signal = diag.get("mean_signal")
        mean_signal_fit = diag.get("mean_signal_fit")
        mean_curves[qname] = (
            np.asarray(mean_signal, dtype=float) if mean_signal is not None else np.full_like(amps, np.nan)
        )
        mean_fit_curves[qname] = (
            np.asarray(mean_signal_fit, dtype=float) if mean_signal_fit is not None else np.full_like(amps, np.nan)
        )

    mean_stack = np.stack([mean_curves[q] for q in qubit_names], axis=0)
    mean_fit_stack = np.stack([mean_fit_curves[q] for q in qubit_names], axis=0)
    ds_fit = ds.assign(
        state_mean=(["qubit", "amp_prefactor"], mean_stack),
        state_mean_fit=(["qubit", "amp_prefactor"], mean_fit_stack),
    )
    return ds_fit, fit_results


def log_fitted_results(
    fit_results: Dict[str, Any],
    log_callable=None,
) -> None:
    """Log fitted results for all qubits."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for qname, r in fit_results.items():
        if r.get("success", False):
            msg = (
                f"[{qname}] SUCCESS | "
                f"a_π = {r.get('opt_amp', 0):.4f} | "
                f"Ω = {r.get('rabi_frequency', 0):.3f} rad/u.a./pulse | "
                f"γ = {r.get('decay_rate', 0):.5f}/pulse | "
                f"σ_g = {r.get('gauss_decay_rate', 0):.5f}/pulse | "
                f"N_eff = {r.get('n_eff', 0):.0f}"
            )
        else:
            msg = f"[{qname}] FAIL | fit did not pass sanity checks"
        log_callable(msg)
