from scipy.signal import savgol_filter
import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate.core import QualibrationNode


@dataclass
class FitParameters:
    """Stores the MW-FEM time-of-flight fit parameters for a single sensor."""

    tof_to_add: int
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all sensors from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all sensors.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for q in fit_results.keys():
        s_qubit = f"Results for sensor {q}: "
        s_tof = f"\tTime of flight to add: {fit_results[q]['tof_to_add']:.0f} ns\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_tof)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    # Convert raw ADC traces into volts
    ds = ds.assign({key: -ds[key] / 2**12 for key in ("adcI", "adcQ", "adc_single_runI", "adc_single_runQ")})
    # Add the IQ amplitude to the dataset
    ds = ds.assign({"IQ_abs": np.sqrt(ds["adcI"] ** 2 + ds["adcQ"] ** 2)})
    ds.IQ_abs.attrs = {"long_name": "IQ amplitude", "units": "V"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the time-of-flight delay for each sensor from the IQ amplitude trace.

    Parameters:
    -----------
    ds : xr.Dataset
        Processed dataset (volts) containing ``adcI`` / ``adcQ`` / ``IQ_abs``.
    node : QualibrationNode
        Calibration node (provides ``machine`` and ``sensors`` for controller mapping).

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results (copy; does not mutate ``ds``).
    dict[str, FitParameters]
        Per-sensor fit parameters.
    """
    ds_fit = ds.copy(deep=True)
    # Filter the data to get the pulse arrival time
    ds_fit["filtered_adc"] = xr.apply_ufunc(_filter_adc_signal, ds_fit.IQ_abs)
    # Detect the pulse arrival times
    ds_fit["threshold"] = (
        ds_fit["filtered_adc"][:, 100:].mean("readout_time") + ds_fit["filtered_adc"][:, :-100].mean("readout_time")
    ) / 2
    ds_fit["delay"] = (ds_fit["filtered_adc"] > ds_fit["threshold"]).where(True).idxmax("readout_time")
    ds_fit["delay"] = np.round(ds_fit["delay"] / 4) * 4
    ds_fit.delay.attrs = {"long_name": "TOF to add", "units": "ns"}
    ds_fit = ds_fit.assign_coords(
        {
            "con": (
                ["sensor"],
                [
                    node.machine.sensor_dots[q.name].readout_resonator.opx_input.controller_id
                    for q in node.namespace["sensors"]
                ],
            )
        }
    )

    # Assess whether the fit was successful or not
    nan_success = np.isnan(ds_fit.delay.data)
    success_criteria = ~nan_success
    ds_fit = ds_fit.assign_coords(success=("sensor", success_criteria))
    # Populate the FitParameters class with fitted values
    fit_results = {
        q: FitParameters(
            tof_to_add=int(ds_fit.sel(sensor=q).delay),
            success=bool(ds_fit.sel(sensor=q).success.values),
        )
        for q in ds_fit.sensor.values
    }

    return ds_fit, fit_results


def _extract_relevant_fit_parameters(ds_fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    pass
    # return fit, fit_results


def _filter_adc_signal(data, window_length=11, polyorder=3):
    """
    Applies a Savitzky-Golay filter to smooth the absolute IQ signal in the dataset.
    """
    return savgol_filter(data, window_length, polyorder)
