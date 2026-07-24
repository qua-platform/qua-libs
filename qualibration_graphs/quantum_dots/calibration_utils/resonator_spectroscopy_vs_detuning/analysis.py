import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from qualibrate.core import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase


@dataclass
class FitParameters:
    """Fitted resonator spectroscopy vs detuning results for a single sensor (02c)."""

    success: bool
    """True if the fit passed sanity checks and is safe for the state update."""

    resonator_frequency: float
    """Absolute readout frequency at the PCA signal peak, in Hz."""

    frequency_shift: float
    """Fitted readout frequency offset at the PCA peak, in Hz."""

    optimal_detuning: float
    """QD pair gate voltage at the PCA signal peak, in V."""

    peak_pca_signal: float
    """PCA signal amplitude at the peak (arbitrary units)."""


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log fitted results for all sensors.

    Parameters
    ----------
    fit_results : dict
        ``fit_results[sensor_name]`` mapping to the fitted values.
    log_callable : callable, optional
        Logging function (typically ``node.log``). Defaults to the module logger.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for sensor_name, result in fit_results.items():
        if result["success"]:
            msg = (
                f"[{sensor_name}] SUCCESS | "
                f"resonator_frequency = {1e-9 * result['resonator_frequency']:.6f} GHz | "
                f"frequency_shift = {1e-6 * result['frequency_shift']:.2f} MHz | "
                f"optimal_detuning = {result['optimal_detuning']:.4f} V | "
                f"peak_pca_signal = {result['peak_pca_signal']:.3e}"
            )
        else:
            msg = f"[{sensor_name}] FAIL | fit did not pass sanity checks"
        log_callable(msg)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Add derived IQ fields and absolute readout-frequency coordinate."""
    ds = add_amplitude_and_phase(ds, "frequency", subtract_slope_flag=True)
    full_freq = np.array(
        [ds.frequency + sensor.readout_resonator.intermediate_frequency for sensor in node.namespace["sensors"]]
    )
    ds = ds.assign_coords(full_freq=(["sensor", "frequency"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """Compute a 2D PCA signal map on the processed dataset and extract the peak point per sensor."""
    ds_fit = ds
    pca_signal = np.full(ds.I.shape, np.nan, dtype=float)
    pca_signal_abs = np.full(ds.I.shape, np.nan, dtype=float)

    frequency_shift = np.full(len(ds.sensor), np.nan, dtype=float)
    optimal_detuning = np.full(len(ds.sensor), np.nan, dtype=float)
    peak_pca_signal = np.full(len(ds.sensor), np.nan, dtype=float)

    for i, sensor_name in enumerate(ds.sensor.values):
        i_map = np.asarray(ds.I.sel(sensor=sensor_name).values, dtype=float)
        q_map = np.asarray(ds.Q.sel(sensor=sensor_name).values, dtype=float)

        i_map = i_map - np.nanmean(i_map, axis=1, keepdims=True)
        q_map = q_map - np.nanmean(q_map, axis=1, keepdims=True)

        i_flat = i_map.ravel()
        q_flat = q_map.ravel()
        finite_mask = np.isfinite(i_flat) & np.isfinite(q_flat)
        if np.count_nonzero(finite_mask) < 2:
            continue

        x = np.column_stack([i_flat[finite_mask], q_flat[finite_mask]])
        x_centered = x - np.mean(x, axis=0, keepdims=True)
        cov = np.cov(x_centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        pc1 = eigvecs[:, int(np.argmax(eigvals))]

        projection = x_centered @ pc1
        if np.abs(np.nanmin(projection)) > np.abs(np.nanmax(projection)):
            projection = -projection

        proj_full = np.full(i_flat.shape, np.nan, dtype=float)
        proj_full[finite_mask] = projection
        proj_map = proj_full.reshape(i_map.shape)

        abs_map = np.abs(proj_map)
        pca_signal[i] = proj_map
        pca_signal_abs[i] = abs_map

        if np.any(np.isfinite(abs_map)):
            max_index = np.unravel_index(np.nanargmax(abs_map), abs_map.shape)
            freq_idx, det_idx = max_index
            frequency_shift[i] = float(ds.frequency.values[freq_idx])
            optimal_detuning[i] = float(ds.detuning.values[det_idx])
            peak_pca_signal[i] = float(abs_map[max_index])

    ds_fit["pca_signal"] = xr.DataArray(pca_signal, dims=["sensor", "frequency", "detuning"], coords=ds.coords)
    ds_fit["pca_signal_abs"] = xr.DataArray(pca_signal_abs, dims=["sensor", "frequency", "detuning"], coords=ds.coords)

    ds_fit = ds_fit.assign_coords(
        {
            "frequency_shift": ("sensor", frequency_shift),
            "optimal_detuning": ("sensor", optimal_detuning),
            "peak_pca_signal": ("sensor", peak_pca_signal),
        }
    )
    ds_fit.frequency_shift.attrs = {"long_name": "readout frequency offset from IF", "units": "Hz"}
    return _extract_relevant_fit_parameters(ds_fit, node)


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add final coordinates and build fit-result dictionary."""
    intermediate_freq = np.array(
        [sensor.readout_resonator.intermediate_frequency for sensor in node.namespace["sensors"]]
    )
    res_freq = fit.frequency_shift.data + intermediate_freq
    fit = fit.assign_coords(res_freq=("sensor", res_freq))
    fit.res_freq.attrs = {"long_name": "resonator frequency", "units": "Hz"}

    span_hz = node.parameters.frequency_span_in_mhz * 1e6
    freq_success = np.abs(fit.frequency_shift.data) <= span_hz / 2.0
    finite_success = np.isfinite(fit.frequency_shift.data) & np.isfinite(fit.optimal_detuning.data)
    success_criteria = freq_success & finite_success
    fit = fit.assign_coords(success=("sensor", success_criteria))

    fit_results = {
        s: FitParameters(
            success=bool(fit.sel(sensor=s).success.values),
            resonator_frequency=float(fit.res_freq.sel(sensor=s).values),
            frequency_shift=float(fit.frequency_shift.sel(sensor=s).values),
            optimal_detuning=float(fit.optimal_detuning.sel(sensor=s).values),
            peak_pca_signal=float(fit.peak_pca_signal.sel(sensor=s).values),
        )
        for s in fit.sensor.values
    }
    return fit, fit_results
