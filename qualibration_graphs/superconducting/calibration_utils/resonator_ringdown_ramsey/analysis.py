"""Fit Ramsey fringes and exponential residual-photon phase decay."""

import logging

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit


def _fit_fringe(signal: np.ndarray, frames: np.ndarray):
    theta = 2 * np.pi * frames
    design = np.column_stack([np.ones(theta.size), np.cos(theta), np.sin(theta)])
    offset, cosine, sine = np.linalg.lstsq(design, signal, rcond=None)[0]
    fitted = offset + cosine * np.cos(theta) + sine * np.sin(theta)
    return 2 * np.hypot(cosine, sine), np.arctan2(-sine, cosine), fitted


def _exponential(delay_ns, amplitude, tau_ns, asymptote):
    return amplitude * np.exp(-np.asarray(delay_ns) / tau_ns) + asymptote


def fit_raw_data(ds: xr.Dataset):
    """Return the Ramsey diagnostics and per-qubit ring-down fit results."""
    delays = np.asarray(ds.ringdown_delay, dtype=float)
    frames = np.asarray(ds.frame, dtype=float)
    contrast = np.full((ds.sizes["qubit"], len(delays)), np.nan)
    phase = np.full_like(contrast, np.nan)
    fringe_fit = np.full((ds.sizes["qubit"], len(delays), len(frames)), np.nan)
    phase_fit = np.full_like(contrast, np.nan)
    fit_results = {}

    for qubit_index, qubit_name in enumerate(ds.qubit.values):
        for delay_index in range(len(delays)):
            result = _fit_fringe(np.asarray(ds.state.isel(qubit=qubit_index, ringdown_delay=delay_index)), frames)
            (
                contrast[qubit_index, delay_index],
                phase[qubit_index, delay_index],
                fringe_fit[qubit_index, delay_index],
            ) = result
        unwrapped = np.unwrap(phase[qubit_index])
        amplitude_guess = float(unwrapped[0] - unwrapped[-1])
        tau_guess = max(100.0, float((delays[-1] - delays[0]) / 3))
        try:
            popt, pcov = curve_fit(
                _exponential,
                delays,
                unwrapped,
                p0=(amplitude_guess, tau_guess, float(unwrapped[-1])),
                bounds=([-4 * np.pi, 4.0, -20 * np.pi], [4 * np.pi, 20000.0, 20 * np.pi]),
                maxfev=20000,
            )
            uncertainty = float(np.sqrt(np.diag(pcov))[1])
            success = np.all(np.isfinite(popt))
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            popt = (np.nan, np.nan, np.nan)
            uncertainty = np.nan
            success = False
        phase[qubit_index] = unwrapped
        phase_fit[qubit_index] = _exponential(delays, *popt)
        fit_results[str(qubit_name)] = {
            "tau_ringdown_ns": float(popt[1]),
            "tau_ringdown_uncertainty_ns": uncertainty,
            "wait_for_5pct_residual_ns": float(-np.log(0.05) * popt[1]),
            "wait_for_1pct_residual_ns": float(-np.log(0.01) * popt[1]),
            "initial_phase_amplitude_rad": float(popt[0]),
            "phase_asymptote_rad": float(popt[2]),
            "success": bool(success),
        }

    ds = ds.assign(
        ramsey_contrast=(("qubit", "ringdown_delay"), contrast),
        ramsey_phase_rad=(("qubit", "ringdown_delay"), phase),
        ramsey_phase_fit_rad=(("qubit", "ringdown_delay"), phase_fit),
        ramsey_fringe_fit=(("qubit", "ringdown_delay", "frame"), fringe_fit),
    )
    return ds, fit_results


def log_fitted_results(fit_results, log_callable=None):
    """Report ring-down fits using the node logger or the module logger."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    log_callable(f"resonator ring-down fits: {fit_results}")
