import logging
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from qualibrate.core import QualibrationNode


@dataclass
class FitParameters:
    """Stores exponential decay fit results for one trace or one aggregated element fit."""

    amplitude: float
    time_constant_ns: float
    cutoff_frequency_Hz: float
    offset: float
    success: bool
    n_sensors_used: int = 1
    tau_std_ns: Optional[float] = None


def _exponential_decay_model(t: np.ndarray, A: float, tau: float, B: float) -> np.ndarray:
    """Exponential decay: S(t) = A * exp(-t / τ) + B.

    After a voltage step through a bias tee (high-pass RC), the signal
    decays from A + B back to B with time constant τ = RC.
    """
    return A * np.exp(-t / tau) + B


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
    time_ns: np.ndarray,
    signal: np.ndarray,
    p0: list,
) -> tuple:
    """Attempt a single curve_fit and return (popt, r_squared) or None on failure."""
    try:
        popt, _ = curve_fit(
            _exponential_decay_model,
            time_ns,
            signal,
            p0=p0,
            bounds=([-np.inf, 1.0, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=10_000,
        )
        r2 = _r_squared(signal, _exponential_decay_model(time_ns, *popt))
        if popt[1] > 0:
            return popt, r2
    except (RuntimeError, ValueError):
        pass
    return None


def _fit_individual(
    time_ns: np.ndarray,
    signal: np.ndarray,
    estimated_tau_ns: float = None,
) -> FitParameters:
    """Fit a single amplitude-vs-time trace with an exponential decay."""
    A_pos = float(signal[0] - signal[-1])
    A_neg = -A_pos
    B_start = float(signal[-1])
    B_end = float(signal[0])
    t_span = float(time_ns[-1] - time_ns[0])

    tau_guesses = [t_span / 10, t_span / 3, t_span, t_span * 3]
    if estimated_tau_ns is not None:
        tau_guesses.insert(0, float(estimated_tau_ns))

    best_popt = None
    best_r2 = -np.inf

    for tau_g in tau_guesses:
        for A_g, B_g in [(A_pos, B_start), (A_neg, B_end)]:
            result = _try_fit(time_ns, signal, [A_g, tau_g, B_g])
            if result is not None and result[1] > best_r2:
                best_popt, best_r2 = result

    if best_popt is not None and best_r2 > 0.5:
        A, tau, B = best_popt
        success = True
    else:
        A = A_pos
        tau = t_span / 3
        B = B_start
        success = False

    f_c = 1e9 / (2 * np.pi * tau) if tau > 0 else 0.0
    return FitParameters(
        amplitude=float(A),
        time_constant_ns=float(tau),
        cutoff_frequency_Hz=float(f_c),
        offset=float(B),
        success=success,
    )


def _aggregate_element_fit_results(element_name: str, sensor_names: list[str], fit_results: Dict[str, FitParameters]) -> FitParameters:
    """Aggregate successful per-sensor fits into one per-element fit result."""
    successful_results = []
    for sensor_name in sensor_names:
        fit_key = f"{element_name}_{sensor_name}"
        fit_result = fit_results.get(fit_key)
        if fit_result is not None and fit_result.success:
            successful_results.append(fit_result)

    if not successful_results:
        return FitParameters(
            amplitude=float("nan"),
            time_constant_ns=float("nan"),
            cutoff_frequency_Hz=float("nan"),
            offset=float("nan"),
            success=False,
            n_sensors_used=0,
        )

    taus = np.array([result.time_constant_ns for result in successful_results], dtype=float)
    tau_ns = float(np.median(taus))
    tau_std_ns = float(np.std(taus)) if len(taus) > 1 else 0.0
    cutoff_frequency_hz = 1e9 / (2 * np.pi * tau_ns) if tau_ns > 0 else 0.0

    return FitParameters(
        amplitude=float(np.mean([result.amplitude for result in successful_results])),
        time_constant_ns=tau_ns,
        cutoff_frequency_Hz=float(cutoff_frequency_hz),
        offset=float(np.mean([result.offset for result in successful_results])),
        success=True,
        n_sensors_used=len(successful_results),
        tau_std_ns=tau_std_ns,
    )


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit exponential decay to all element/sensor pairs."""
    elements = node.namespace["elements"]
    sensors = node.namespace["sensors"]
    sensor_names = [sensor.name for sensor in sensors]

    time_ns = ds.time.values
    ds_fit = ds.copy()
    fit_results: Dict[str, FitParameters] = {}

    for el in elements:
        for i, sensor in enumerate(sensors):
            key = f"{el.name}_{sensor.name}"
            amp_key = f"amplitude_{el.name}_{i + 1}"

            if amp_key not in ds:
                continue

            signal = ds[amp_key].values
            fit_params = _fit_individual(
                time_ns,
                signal,
                estimated_tau_ns=node.parameters.estimated_bias_tee_tau_ns,
            )
            fit_results[key] = fit_params

            if fit_params.success:
                fit_curve = _exponential_decay_model(
                    time_ns,
                    fit_params.amplitude,
                    fit_params.time_constant_ns,
                    fit_params.offset,
                )
                fit_key = f"fit_{el.name}_{i + 1}"
                ds_fit = ds_fit.assign({fit_key: xr.DataArray(fit_curve, dims=["time"])})

                correction = fit_params.amplitude * (1 - np.exp(-time_ns / fit_params.time_constant_ns))
                corrected = signal + correction
                corr_key = f"amplitude_corrected_{el.name}_{i + 1}"
                ds_fit = ds_fit.assign({corr_key: xr.DataArray(corrected, dims=["time"])})

        fit_results[el.name] = _aggregate_element_fit_results(el.name, sensor_names, fit_results)

    return ds_fit, fit_results


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log the fitted exponential decay parameters for all element and sensor fits."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for key, result in fit_results.items():
        if isinstance(result, FitParameters):
            result = asdict(result)
        status = "SUCCESS" if result["success"] else "FAIL"
        tau_ns = result["time_constant_ns"]
        tau_us = tau_ns / 1e3 if np.isfinite(tau_ns) else float("nan")
        lines = [
            f"Results for {key}: {status}!",
            f"\tTime constant:    {tau_ns:.1f} ns ({tau_us:.2f} µs)" if np.isfinite(tau_ns) else "\tTime constant:    n/a",
            f"\tCutoff frequency: {result['cutoff_frequency_Hz']:.1f} Hz"
            if np.isfinite(result["cutoff_frequency_Hz"])
            else "\tCutoff frequency: n/a",
            f"\tAmplitude:        {result['amplitude']:.6f}" if np.isfinite(result["amplitude"]) else "\tAmplitude:        n/a",
            f"\tOffset:           {result['offset']:.6f}" if np.isfinite(result["offset"]) else "\tOffset:           n/a",
        ]
        if "n_sensors_used" in result and result["n_sensors_used"] != 1:
            lines.append(f"\tSensors used:     {result['n_sensors_used']}")
        if result.get("tau_std_ns") is not None and np.isfinite(result["tau_std_ns"]):
            lines.append(f"\tTau std:          {result['tau_std_ns']:.1f} ns")
        log_callable("\n".join(lines))
