import logging
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, List

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from qualibrate.core import QualibrationNode


@dataclass
class FitParameters:
    """Stores the bias tee filter fit results for a single element/sensor combination.

    Attributes:
        amplitude: Scaling factor A in the high-pass transfer function model.
        cutoff_frequency_Hz: Cutoff frequency f_c of the bias tee high-pass filter.
        offset: Baseline offset B.
        time_constant_ns: Derived time constant tau = 1 / (2*pi*f_c), in nanoseconds.
        success: Whether the fit converged to physically reasonable parameters.
    """

    amplitude: float
    cutoff_frequency_Hz: float
    offset: float
    time_constant_ns: float
    success: bool


def _high_pass_model(f: np.ndarray, A: float, f_c: float, B: float) -> np.ndarray:
    """High-pass filter transfer function: S(f) = A * f / sqrt(f^2 + f_c^2) + B.

    At low f (f << f_c): S ≈ A*f/f_c + B  (linear rise)
    At high f (f >> f_c): S ≈ A + B         (saturation)

    Args:
        f: Frequency array in Hz.
        A: Amplitude scaling (sign determines increasing/decreasing trend).
        f_c: Cutoff frequency in Hz.
        B: Baseline offset.
    """
    return A * f / np.sqrt(f**2 + f_c**2) + B


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Compute IQ amplitude for each element/sensor pair.

    Adds ``amplitude_{el_name}_{sensor_idx}`` variables to the dataset.
    """
    elements = node.namespace["elements"]
    sensors = node.namespace["sensors"]

    for el in elements:
        for i in range(len(sensors)):
            suffix = f"{el.name}_{i + 1}"
            I_key = f"I_{suffix}"
            Q_key = f"Q_{suffix}"
            if I_key in ds and Q_key in ds:
                amplitude = np.sqrt(ds[I_key] ** 2 + ds[Q_key] ** 2)
                amp_key = f"amplitude_{suffix}"
                ds = ds.assign({amp_key: amplitude})
                ds[amp_key].attrs = {"long_name": "IQ amplitude", "units": "V"}

    return ds


def _r_squared(signal: np.ndarray, fit: np.ndarray) -> float:
    ss_res = np.sum((signal - fit) ** 2)
    ss_tot = np.sum((signal - np.mean(signal)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _try_fit(
    frequencies: np.ndarray,
    signal: np.ndarray,
    p0: list,
) -> tuple:
    """Attempt a single curve_fit and return (popt, r_squared) or None on failure."""
    try:
        popt, _ = curve_fit(
            _high_pass_model,
            frequencies,
            signal,
            p0=p0,
            bounds=([-np.inf, 1e-3, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=10_000,
        )
        r2 = _r_squared(signal, _high_pass_model(frequencies, *popt))
        if popt[1] > 0:
            return popt, r2
    except (RuntimeError, ValueError):
        pass
    return None


def _fit_individual(
    frequencies: np.ndarray,
    signal: np.ndarray,
    estimated_tau_ns: float = None,
) -> FitParameters:
    """Fit a single signal-vs-frequency trace with the high-pass transfer function.

    Sign-agnostic: tries multiple initial guesses with both positive and
    negative amplitude, and several f_c values spanning the frequency range.
    Picks the fit with the highest R².

    Args:
        estimated_tau_ns: If provided, the corresponding f_c is included as
            a prioritised initial guess.
    """
    A_pos = float(signal[-1] - signal[0])
    A_neg = -A_pos
    B_low = float(signal[0])
    B_high = float(signal[-1])

    f_c_guesses = [
        float(frequencies[len(frequencies) // 8]),
        float(frequencies[len(frequencies) // 4]),
        float(np.median(frequencies)),
    ]
    if estimated_tau_ns is not None and estimated_tau_ns > 0:
        f_c_guesses.insert(0, 1e9 / (2 * np.pi * estimated_tau_ns))

    best_popt = None
    best_r2 = -np.inf

    for f_c_g in f_c_guesses:
        for A_g, B_g in [(A_pos, B_low), (A_neg, B_high)]:
            result = _try_fit(frequencies, signal, [A_g, f_c_g, B_g])
            if result is not None and result[1] > best_r2:
                best_popt, best_r2 = result

    if best_popt is not None and best_r2 > 0.5:
        A, f_c, B = best_popt
        tau_ns = 1e9 / (2 * np.pi * f_c)
        success = True
    else:
        A = A_pos
        f_c = float(frequencies[len(frequencies) // 4])
        B = B_low
        tau_ns = 1e9 / (2 * np.pi * f_c) if f_c > 0 else 0.0
        success = False

    return FitParameters(
        amplitude=float(A),
        cutoff_frequency_Hz=float(f_c),
        offset=float(B),
        time_constant_ns=float(tau_ns),
        success=success,
    )


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit the high-pass transfer function to all element/sensor pairs.

    Returns:
        ds_fit: Dataset augmented with ``fit_{el_name}_{sensor_idx}`` curves.
        fit_results: Dictionary of ``{el_name}_{sensor_name}: FitParameters``.
    """
    elements = node.namespace["elements"]
    sensors = node.namespace["sensors"]
    frequencies = node.namespace["frequencies"]

    ds_fit = ds.copy()
    fit_results = {}

    for el in elements:
        for i, sensor in enumerate(sensors):
            key = f"{el.name}_{sensor.name}"
            amp_key = f"amplitude_{el.name}_{i + 1}"

            if amp_key not in ds:
                continue

            signal = ds[amp_key].values
            fit_params = _fit_individual(
                frequencies,
                signal,
                estimated_tau_ns=node.parameters.estimated_bias_tee_tau_ns,
            )
            fit_results[key] = fit_params

            fit_curve = _high_pass_model(
                frequencies,
                fit_params.amplitude,
                fit_params.cutoff_frequency_Hz,
                fit_params.offset,
            )
            fit_key = f"fit_{el.name}_{i + 1}"
            ds_fit = ds_fit.assign(
                {fit_key: xr.DataArray(fit_curve, dims=["frequency"])}
            )

            if fit_params.success:
                H = frequencies / np.sqrt(
                    frequencies**2 + fit_params.cutoff_frequency_Hz**2
                )
                correction = fit_params.amplitude * (1 - H)
                corrected = signal + correction
                corr_key = f"amplitude_corrected_{el.name}_{i + 1}"
                ds_fit = ds_fit.assign(
                    {corr_key: xr.DataArray(corrected, dims=["frequency"])}
                )

    node.outcomes = {
        key: ("successful" if fp.success else "failed")
        for key, fp in fit_results.items()
    }

    return ds_fit, fit_results


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log the fitted bias tee filter parameters for all element/sensor pairs."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for key, r in fit_results.items():
        if isinstance(r, FitParameters):
            r = asdict(r)
        status = "SUCCESS" if r["success"] else "FAIL"
        s = (
            f"Results for {key}: {status}!\n"
            f"\tCutoff frequency: {r['cutoff_frequency_Hz']:.1f} Hz\n"
            f"\tTime constant:    {r['time_constant_ns']:.1f} ns\n"
            f"\tAmplitude:        {r['amplitude']:.6f}\n"
            f"\tOffset:           {r['offset']:.6f}"
        )
        log_callable(s)
