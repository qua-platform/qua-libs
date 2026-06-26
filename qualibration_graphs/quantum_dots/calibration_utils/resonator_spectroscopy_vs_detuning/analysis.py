import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from qualibrate.core import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase


@dataclass
class FitParameters:
    """Relevant fit outputs for resonator spectroscopy vs detuning."""

    success: bool
    resonator_frequency: float
    frequency_shift: float
    optimal_detuning: float
    peak_pca_signal: float


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log fitted PCA-map optimal points for all sensors."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for sensor_name, result in fit_results.items():
        status = "SUCCESS!" if result["success"] else "FAIL!"
        msg = (
            f"Results for sensor {sensor_name}: {status}\n"
            f"Peak PCA signal: {result['peak_pca_signal']:.3e} | "
            f"Optimal detuning: {result['optimal_detuning']:.4f} V | "
            f"Resonator frequency: {1e-9 * result['resonator_frequency']:.3f} GHz "
            f"(shift of {1e-6 * result['frequency_shift']:.2f} MHz)"
        )
        log_callable(msg)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Add derived IQ fields and absolute readout-frequency coordinate."""
    ds = add_amplitude_and_phase(ds, "frequency", subtract_slope_flag=True)
    full_freq = np.array(
        [
            ds.frequency + sensor.readout_resonator.intermediate_frequency
            for sensor in node.namespace["sensors"]
        ]
    )
    ds = ds.assign_coords(full_freq=(["sensor", "frequency"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """Compute a 2D PCA signal map and extract the peak point per sensor."""
    ds_fit = xr.Dataset(coords=ds.coords)
    pca_signal = np.full(ds.I.shape, np.nan, dtype=float)
    pca_signal_abs = np.full(ds.I.shape, np.nan, dtype=float)

    optimal_frequency_shift = np.full(len(ds.sensor), np.nan, dtype=float)
    optimal_detuning = np.full(len(ds.sensor), np.nan, dtype=float)
    peak_pca_signal = np.full(len(ds.sensor), np.nan, dtype=float)

    for i, sensor_name in enumerate(ds.sensor.values):
        i_map = np.asarray(ds.I.sel(sensor=sensor_name).values, dtype=float)
        q_map = np.asarray(ds.Q.sel(sensor=sensor_name).values, dtype=float)

        i_map = i_map - np.nanmean(i_map, axis = 1, keepdims=True)
        q_map = q_map - np.nanmean(q_map, axis = 1, keepdims=True)

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
            optimal_frequency_shift[i] = float(ds.frequency.values[freq_idx])
            optimal_detuning[i] = float(ds.detuning.values[det_idx])
            peak_pca_signal[i] = float(abs_map[max_index])

    ds_fit["pca_signal"] = xr.DataArray(
        pca_signal, dims=["sensor", "frequency", "detuning"], coords=ds.coords
    )
    ds_fit["pca_signal_abs"] = xr.DataArray(
        pca_signal_abs, dims=["sensor", "frequency", "detuning"], coords=ds.coords
    )

    ds_fit = ds_fit.assign_coords(
        {
            "optimal_frequency_shift": ("sensor", optimal_frequency_shift),
            "optimal_detuning": ("sensor", optimal_detuning),
            "peak_pca_signal": ("sensor", peak_pca_signal),
        }
    )
    return _extract_relevant_fit_parameters(ds_fit, node)


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add final coordinates and build fit-result dictionary."""
    intermediate_freq = np.array(
        [sensor.readout_resonator.intermediate_frequency for sensor in node.namespace["sensors"]]
    )
    res_freq = fit.optimal_frequency_shift.data + intermediate_freq
    fit = fit.assign_coords(res_freq=("sensor", res_freq))
    fit.res_freq.attrs = {"long_name": "resonator frequency", "units": "Hz"}

    span_hz = node.parameters.frequency_span_in_mhz * 1e6
    freq_success = np.abs(fit.optimal_frequency_shift.data) <= span_hz / 2.0
    finite_success = np.isfinite(fit.optimal_frequency_shift.data) & np.isfinite(
        fit.optimal_detuning.data
    )
    success_criteria = freq_success & finite_success
    fit = fit.assign_coords(success=("sensor", success_criteria))

    fit_results = {
        sensor_name: FitParameters(
            success=bool(fit.sel(sensor=sensor_name).success.values),
            resonator_frequency=float(fit.res_freq.sel(sensor=sensor_name).values),
            frequency_shift=float(
                fit.optimal_frequency_shift.sel(sensor=sensor_name).values
            ),
            optimal_detuning=float(fit.optimal_detuning.sel(sensor=sensor_name).values),
            peak_pca_signal=float(fit.peak_pca_signal.sel(sensor=sensor_name).values),
        )
        for sensor_name in fit.sensor.values
    }
    return fit, fit_results
