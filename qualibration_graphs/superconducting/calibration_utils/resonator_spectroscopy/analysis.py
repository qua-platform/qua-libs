import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibration_libs.analysis import peaks_dips
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from scipy.optimize import curve_fit

from qualibrate import QualibrationNode

_MIN_R2 = 0.3
_MAX_CENTER_ERROR_FRACTION = 0.15


def _signed_lorentzian(x, amplitude, center, width, offset):
    """Lorentzian with a signed contrast: positive is a peak, negative is a dip."""

    return offset + amplitude / (1.0 + ((x - center) / width) ** 2)


@dataclass
class FitParameters:
    """Stores the relevant resonator spectroscopy experiment fit parameters for a single qubit."""

    frequency: float
    fwhm: float
    success: bool
    fit_mse: float
    fit_rmse: float
    fit_r2: float
    feature_is_peak: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Logs fitted results and fit-quality metrics for all qubits."""

    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        result = fit_results[q]
        feature = "peak" if result["feature_is_peak"] else "dip"
        s_qubit = f"Results for qubit {q}: "
        s_freq = f"\tResonator frequency: {1e-9 * result['frequency']:.3f} GHz | "
        s_fwhm = f"FWHM: {1e-3 * result['fwhm']:.1f} kHz | "
        s_quality = (
            f"Feature: {feature} | R²: {result['fit_r2']:.3f} | "
            f"RMSE: {1e3 * result['fit_rmse']:.3f} mV | "
        )
        if result["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_freq + s_fwhm + s_quality)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    full_freq = np.array([ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """Fit a Lorentzian peak or dip to each qubit's resonator spectroscopy trace."""

    detuning = ds.detuning.values
    per_qubit = []
    for qubit_name in ds.qubit.values:
        y = ds.IQ_abs.sel(qubit=qubit_name).values
        per_qubit.append(_fit_single_qubit(detuning, y))

    fit = xr.Dataset(
        {
            "position": ("qubit", [item["center"] for item in per_qubit]),
            "width": ("qubit", [item["hwhm"] for item in per_qubit]),
            "amplitude": ("qubit", [item["amplitude"] for item in per_qubit]),
            "offset": ("qubit", [item["offset"] for item in per_qubit]),
            "fit_mse": ("qubit", [item["mse"] for item in per_qubit]),
            "fit_rmse": ("qubit", [item["rmse"] for item in per_qubit]),
            "fit_r2": ("qubit", [item["r2"] for item in per_qubit]),
            "feature_is_peak": ("qubit", [item["feature_type"] == "peak" for item in per_qubit]),
            "fit_curve": (("qubit", "detuning"), np.stack([item["fitted_y"] for item in per_qubit])),
        },
        coords={"qubit": ds.qubit.values, "detuning": ds.detuning.values},
    )
    fit_data, fit_results = _extract_relevant_fit_parameters(fit, node)
    return fit_data, fit_results


def _fit_single_qubit(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit either sign of the resonance, without allowing a one-point needle fit."""

    try:
        return _fit_with_model(x, y)
    except (RuntimeError, ValueError, TypeError):
        return _fallback_fit(x, y)


def _fit_with_model(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    model = _signed_lorentzian
    center0 = float(x[np.argmax(np.abs(y - np.median(y)))])
    offset0 = float(np.median(y))
    amp0 = float(y[np.argmin(np.abs(x - center0))] - offset0)

    span = max(float(np.ptp(x)), 1.0)
    steps = np.diff(np.unique(x))
    step = float(np.min(steps)) if steps.size else span
    # A width below half a frequency step is not identifiable and lets curve_fit
    # explain a single bad sample with a vertical needle.
    min_width = max(step / 2.0, span / 1000.0, 1.0)
    width0 = max(span * 0.05, min_width)
    contrast = max(float(np.ptp(y)), np.finfo(float).eps)
    lower = [-2.0 * contrast, float(x.min()) - step, min_width, float(np.min(y) - contrast)]
    upper = [2.0 * contrast, float(x.max()) + step, span, float(np.max(y) + contrast)]

    amplitude, center, width, offset = curve_fit(
        model,
        x,
        y,
        p0=[amp0, center0, width0, offset0],
        bounds=(lower, upper),
        maxfev=20_000,
    )[0]
    fitted_y = model(x, amplitude, center, width, offset)
    mse, rmse, r2 = _fit_quality_metrics(y, fitted_y)
    return {
        "feature_type": "peak" if amplitude >= 0 else "dip",
        "center": float(center),
        "hwhm": float(width),
        "amplitude": float(amplitude),
        "offset": float(offset),
        "fitted_y": fitted_y,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def _fallback_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """Use peaks_dips for an initial center, then retry bounded Lorentzian fits."""

    da = xr.DataArray(y, coords={"detuning": x}, dims=["detuning"])
    detected = peaks_dips(da, "detuning")
    center = float(detected.position.values)
    hwhm = max(float(abs(detected.width.values)) / 2.0, 1.0)
    amplitude = max(float(abs(detected.amplitude.values)), 1e-12)
    offset = float(detected.base_line.mean().values)

    model = _signed_lorentzian
    amplitude = amplitude if detected.amplitude.values >= 0 else -amplitude
    fitted_y = model(x, amplitude, center, hwhm, offset)
    mse, rmse, r2 = _fit_quality_metrics(y, fitted_y)
    return {
        "feature_type": "peak" if amplitude >= 0 else "dip",
        "center": center,
        "hwhm": hwhm,
        "amplitude": amplitude,
        "offset": offset,
        "fitted_y": fitted_y,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def _fit_quality_metrics(y: np.ndarray, fitted_y: np.ndarray) -> tuple[float, float, float]:
    residual = y - fitted_y
    mse = float(np.mean(residual**2))
    rmse = float(np.sqrt(mse))
    total_var = float(np.var(y))
    r2 = 1.0 - mse / total_var if total_var > 0 else 0.0
    return mse, rmse, r2


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""

    fit.attrs = {"long_name": "frequency", "units": "Hz"}
    full_freq = np.array([q.resonator.RF_frequency for q in node.namespace["qubits"]])
    res_freq = fit.position + full_freq
    fit = fit.assign_coords(res_freq=("qubit", res_freq.data))
    fit.res_freq.attrs = {"long_name": "resonator frequency", "units": "Hz"}

    fwhm = 2.0 * np.abs(fit.width)
    fit = fit.assign_coords(fwhm=("qubit", fwhm.data))
    fit.fwhm.attrs = {"long_name": "resonator fwhm", "units": "Hz"}

    span_hz = node.parameters.frequency_span_in_mhz * 1e6
    center_error = np.abs(fit.position.data) / max(span_hz, 1.0)
    freq_success = np.isfinite(res_freq.data) & (center_error <= _MAX_CENTER_ERROR_FRACTION)
    fwhm_success = np.isfinite(fwhm.data) & (fwhm.data > 0) & (fwhm.data < span_hz)
    quality_success = fit.fit_r2.data >= _MIN_R2
    success_criteria = freq_success & fwhm_success & quality_success
    fit = fit.assign_coords(success=("qubit", success_criteria))

    fit_results = {
        q: FitParameters(
            frequency=fit.sel(qubit=q).res_freq.values.__float__(),
            fwhm=fit.sel(qubit=q).fwhm.values.__float__(),
            success=fit.sel(qubit=q).success.values.__bool__(),
            fit_mse=fit.sel(qubit=q).fit_mse.values.__float__(),
            fit_rmse=fit.sel(qubit=q).fit_rmse.values.__float__(),
            fit_r2=fit.sel(qubit=q).fit_r2.values.__float__(),
            feature_is_peak=bool(fit.sel(qubit=q).feature_is_peak.values),
        )
        for q in fit.qubit.values
    }
    return fit, fit_results
