from __future__ import annotations

"""Lorentzian fitting for sensor dot Coulomb peak tuning.

The standard Lorentzian used here is

    L(x) = (γ / 2π) / ((x − x0)² + (γ / 2)²) + offset

The inflection points — where the slope |dL/dx| is maximised, giving the
highest charge sensitivity — sit at  x0 ± γ / (2√3).
"""


import logging
from dataclasses import dataclass
from typing import Tuple, Dict, NamedTuple, Optional
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from qualibrate.core import QualibrationNode
from qualibration_libs.analysis import peaks_dips

__all__ = [
    "LorentzianFitResult",
    "lorentzian",
    "optimal_operating_point",
    "fit_lorentzian",
]


class LorentzianFitResult(NamedTuple):
    """Container for the fitted Lorentzian parameters."""

    x0: float
    gamma: float
    amplitude: float
    offset: float
    optimal_voltage: float


def lorentzian(x: np.ndarray, x0: float, gamma: float, amplitude: float, offset: float) -> np.ndarray:
    """Lorentzian peak with free amplitude and offset.

    L(x) = amplitude * (γ/2)² / ((x − x0)² + (γ/2)²) + offset

    This parameterisation keeps the peak height = amplitude + offset at x = x0,
    which maps naturally to the sensor Coulomb peak signal.
    """
    half_gamma_sq = (gamma / 2) ** 2
    return amplitude * half_gamma_sq / ((x - x0) ** 2 + half_gamma_sq) + offset


def optimal_operating_point(x0: float, gamma: float, side: str = "right") -> float:
    """Return the voltage at the Lorentzian inflection point.

    The inflection points of the standard Lorentzian sit at
        x0 ± γ / (2√3)
    which is where |dL/dx| is maximised.
    """
    delta = gamma / (2 * np.sqrt(3))
    if side == "right":
        return x0 + delta
    elif side == "left":
        return x0 - delta
    raise ValueError(f"side must be 'left' or 'right', got '{side}'")


def fit_lorentzian(
    v: np.ndarray,
    signal: np.ndarray,
    side: str = "right",
    *,
    x0_guess: Optional[float] = None,
    gamma_guess: Optional[float] = None,
    amp_guess: Optional[float] = None,
    offset_guess: Optional[float] = None,
) -> LorentzianFitResult:
    """Fit a Lorentzian peak to a 1D sensor sweep.

    Parameters
    ----------
    v : np.ndarray
        Voltage values (1D).
    signal : np.ndarray
        Measured sensor signal (1D, same length as *v*).
    side : str
        Which inflection point to use ('left' or 'right').

    Returns
    -------
    LorentzianFitResult
        Fitted parameters and the derived optimal voltage.
    """
    v = np.asarray(v, dtype=float)
    signal = np.asarray(signal, dtype=float)

    if offset_guess is None:
        offset_guess = float(np.nanmedian(signal))

    if x0_guess is None:
        peak_idx = int(np.nanargmax(signal))
        x0_guess = float(v[peak_idx])

    if amp_guess is None:
        idx0 = int(np.nanargmin(np.abs(v - x0_guess)))
        amp_guess = float(signal[idx0] - offset_guess)

    if gamma_guess is None:
        gamma_guess = float((v[-1] - v[0]) / 10) if len(v) > 1 else 1.0

    popt, pcov = curve_fit(
        lorentzian,
        v,
        signal,
        p0=[x0_guess, gamma_guess, amp_guess, offset_guess],
        maxfev=10_000,
    )
    x0_fit, gamma_fit, amp_fit, offset_fit = popt
    gamma_fit = abs(gamma_fit)

    return LorentzianFitResult(
        x0=x0_fit,
        gamma=gamma_fit,
        amplitude=amp_fit,
        offset=offset_fit,
        optimal_voltage=optimal_operating_point(x0_fit, gamma_fit, side),
    )


@dataclass
class FitParameters:
    """Stores the relevant sensor gate sweep experiment fit parameters for a single sensor"""

    peak_position: float
    peak_amplitude: float
    peak_width: float
    lorentzian_x0: float
    lorentzian_gamma: float
    lorentzian_amplitude: float
    lorentzian_offset: float
    optimal_bias: float
    max_gradient: float
    max_gradient_bias: float
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all sensors from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all sensors.
    log_callable : callable, optional
        Callable for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_sensor = f"Results for sensor {q}: "
        s_peak = f"\tPeak position: {fit_results[q]['peak_position']:.4f} V | "
        s_gamma = f"Lorentzian FWHM (gamma): {fit_results[q]['lorentzian_gamma']:.4e} V | "
        s_grad = f"Max gradient bias: {fit_results[q]['max_gradient_bias']:.4f} V | "
        s_grad_val = f"Max gradient: {fit_results[q]['max_gradient']:.4e}"
        if fit_results[q]["success"]:
            s_sensor += " SUCCESS!\n"
        else:
            s_sensor += " FAIL!\n"
        log_callable(s_sensor + s_peak + s_gamma + s_grad + s_grad_val)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Process raw dataset to add amplitude and phase information.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw I and Q quadrature data.
    node : QualibrationNode
        The calibration node containing parameters and sensors.

    Returns:
    --------
    xr.Dataset
        Processed dataset with amplitude and phase added.
    """
    amplitude = np.sqrt(ds.I**2 + ds.Q**2)
    ds = ds.assign({"amplitude": amplitude})
    ds.amplitude.attrs = {"long_name": "IQ amplitude", "units": "V"}

    phase = np.arctan2(ds.Q, ds.I)
    ds = ds.assign({"phase": phase})
    ds.phase.attrs = {"long_name": "IQ phase", "units": "rad"}

    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Find the sensor response peak/dip via peaks_dips, fit a Lorentzian, and
    locate the bias point of maximum gradient from the analytical inflection point.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data with amplitude variable.
    node : QualibrationNode
        The QUAlibrate node.

    Returns:
    --------
    Tuple[xr.Dataset, dict[str, FitParameters]]
        - Dataset containing the fit results
        - Dictionary of FitParameters for each sensor
    """
    peak_results = xr.concat(
        [
            peaks_dips(ds.amplitude.sel(sensors=sensor.name), "bias_offsets").expand_dims(sensors=[sensor.name])
            for sensor in node.namespace["sensors"]
        ],
        dim="sensors",
    )

    side = getattr(node.parameters, "peak_fit_side", "left")
    ds_fit, fit_results = _extract_relevant_fit_parameters(peak_results, ds, node, side=side)
    return ds_fit, fit_results


def _lorentzian_gradient(x: np.ndarray, x0: float, gamma: float, amplitude: float) -> np.ndarray:
    """Analytical derivative of the Lorentzian peak.

    dL/dx = -amplitude * (γ/2)² * 2(x - x0) / ((x - x0)² + (γ/2)²)²
    """
    half_gamma_sq = (gamma / 2) ** 2
    return -amplitude * half_gamma_sq * 2 * (x - x0) / ((x - x0) ** 2 + half_gamma_sq) ** 2


def _extract_relevant_fit_parameters(peak_ds: xr.Dataset, ds: xr.Dataset, node: QualibrationNode, side: str = "left"):
    """Use peaks_dips results for validation, fit a Lorentzian per sensor, and
    derive the max-gradient point analytically.

    Parameters:
    -----------
    peak_ds : xr.Dataset
        Dataset returned by peaks_dips with position, width, amplitude, base_line.
    ds : xr.Dataset
        Processed dataset containing the amplitude variable.
    node : QualibrationNode
        The calibration node.

    Returns:
    --------
    Tuple[xr.Dataset, dict[str, FitParameters]]
    """
    sensors = node.namespace["sensors"]
    bias_offsets = ds.bias_offsets.values

    lor_x0_list = []
    lor_gamma_list = []
    lor_amp_list = []
    lor_offset_list = []
    optimal_bias_list = []
    max_grad_bias_list = []
    max_grad_value_list = []
    fitted_curve_list = []
    gradient_curve_list = []
    success_list = []

    log_callable = getattr(node, "log", None)
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for sensor in sensors:
        peak_pos = peak_ds.position.sel(sensors=sensor.name).values
        peak_detected = not np.isnan(peak_pos)

        sensor_amp = ds.amplitude.sel(sensors=sensor.name).values

        if peak_detected:
            try:
                peak_width = float(peak_ds.width.sel(sensors=sensor.name).values)
                gamma_guess = peak_width if np.isfinite(peak_width) and peak_width > 0 else None

                lor_result = fit_lorentzian(
                    bias_offsets,
                    sensor_amp,
                    side=side,
                    x0_guess=float(peak_pos),
                    gamma_guess=gamma_guess,
                )

                # Guardrails: ensure the derived optimum is meaningful for this scan.
                finite_fit = (
                    np.isfinite(lor_result.x0)
                    and np.isfinite(lor_result.gamma)
                    and np.isfinite(lor_result.optimal_voltage)
                )
                in_span = (bias_offsets.min() <= lor_result.optimal_voltage) and (
                    lor_result.optimal_voltage <= bias_offsets.max()
                )
                if not finite_fit:
                    raise ValueError("Non-finite Lorentzian fit parameters.")
                if not in_span:
                    raise ValueError(
                        "Optimal bias outside scanned range " f"[{bias_offsets.min():.4g}, {bias_offsets.max():.4g}] V."
                    )

                fitted = lorentzian(
                    bias_offsets,
                    lor_result.x0,
                    lor_result.gamma,
                    lor_result.amplitude,
                    lor_result.offset,
                )
                grad = _lorentzian_gradient(bias_offsets, lor_result.x0, lor_result.gamma, lor_result.amplitude)
                max_grad_idx = int(np.argmin(np.abs(bias_offsets - lor_result.optimal_voltage)))

                lor_x0_list.append(lor_result.x0)
                lor_gamma_list.append(lor_result.gamma)
                lor_amp_list.append(lor_result.amplitude)
                lor_offset_list.append(lor_result.offset)
                optimal_bias_list.append(lor_result.optimal_voltage)
                max_grad_bias_list.append(float(bias_offsets[max_grad_idx]))
                max_grad_value_list.append(float(grad[max_grad_idx]))
                fitted_curve_list.append(fitted)
                gradient_curve_list.append(grad)
                success_list.append(True)
                continue
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                log_callable(
                    f"[fit] Sensor '{sensor.name}': feature detected at {float(peak_pos):.4g} V "
                    f"but fit rejected ({type(exc).__name__}: {exc}). Marking as failed."
                )

        lor_x0_list.append(np.nan)
        lor_gamma_list.append(np.nan)
        lor_amp_list.append(np.nan)
        lor_offset_list.append(np.nan)
        optimal_bias_list.append(np.nan)
        max_grad_bias_list.append(np.nan)
        max_grad_value_list.append(np.nan)
        fitted_curve_list.append(np.full_like(bias_offsets, np.nan))
        gradient_curve_list.append(np.full_like(bias_offsets, np.nan))
        success_list.append(False)

    sensor_names = [s.name for s in sensors]
    fit = peak_ds.copy().rename({"amplitude": "peak_amplitude"})

    fit = fit.assign_coords(lorentzian_x0=("sensors", lor_x0_list))
    fit.lorentzian_x0.attrs = {"long_name": "Lorentzian center", "units": "V"}
    fit = fit.assign_coords(lorentzian_gamma=("sensors", lor_gamma_list))
    fit.lorentzian_gamma.attrs = {"long_name": "Lorentzian FWHM", "units": "V"}
    fit = fit.assign_coords(lorentzian_amplitude=("sensors", lor_amp_list))
    fit.lorentzian_amplitude.attrs = {"long_name": "Lorentzian amplitude", "units": "V"}
    fit = fit.assign_coords(lorentzian_offset=("sensors", lor_offset_list))
    fit.lorentzian_offset.attrs = {"long_name": "Lorentzian offset", "units": "V"}
    fit = fit.assign_coords(optimal_bias=("sensors", optimal_bias_list))
    fit.optimal_bias.attrs = {
        "long_name": "Optimal bias (inflection point)",
        "units": "V",
    }
    fit = fit.assign_coords(max_gradient_bias=("sensors", max_grad_bias_list))
    fit.max_gradient_bias.attrs = {"long_name": "Bias at max gradient", "units": "V"}
    fit = fit.assign_coords(max_gradient=("sensors", max_grad_value_list))
    fit.max_gradient.attrs = {"long_name": "Maximum gradient value", "units": "V/V"}
    fit = fit.assign_coords(success=("sensors", success_list))

    fitted_da = xr.DataArray(
        fitted_curve_list,
        dims=["sensors", "bias_offsets"],
        coords={"sensors": ds.sensors, "bias_offsets": ds.bias_offsets},
        attrs={"long_name": "Lorentzian fit", "units": "V"},
    )
    gradient_da = xr.DataArray(
        gradient_curve_list,
        dims=["sensors", "bias_offsets"],
        coords={"sensors": ds.sensors, "bias_offsets": ds.bias_offsets},
        attrs={"long_name": "dL/d(bias)", "units": "V/V"},
    )
    fit = xr.merge([fit, fitted_da.rename("fitted_curve"), gradient_da.rename("gradient")])
    if "amplitude" in ds and "phase" in ds:
        fit = xr.merge([fit, ds[["amplitude", "phase"]]])

    fit_results = {
        sensor.name: FitParameters(
            peak_position=float(peak_ds.position.sel(sensors=sensor.name).values),
            peak_amplitude=float(peak_ds.amplitude.sel(sensors=sensor.name).values),
            peak_width=float(peak_ds.width.sel(sensors=sensor.name).values),
            lorentzian_x0=lor_x0_list[i],
            lorentzian_gamma=lor_gamma_list[i],
            lorentzian_amplitude=lor_amp_list[i],
            lorentzian_offset=lor_offset_list[i],
            optimal_bias=optimal_bias_list[i],
            max_gradient=max_grad_value_list[i],
            max_gradient_bias=max_grad_bias_list[i],
            success=success_list[i],
        )
        for i, sensor in enumerate(sensors)
    }
    return fit, fit_results
