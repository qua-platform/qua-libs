from scipy.signal import savgol_filter
import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode


@dataclass
class FitParameters:
    """Stores the relevant sensor spectroscopy experiment fit parameters for a single sensor"""

    tof_to_add: int
    offset_to_add: float
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
        s_sensor= f"Results for sensor {q}: "
        s_tof = f"\tTime of flight to add: {fit_results[q]['tof_to_add']:.0f} ns\n"
        s_offsets = f"\tOffset to add: {fit_results[q]['offset_to_add'] * 1e3:.1f} mV\n"
        if fit_results[q]["success"]:
            s_sensor += " SUCCESS!\n"
        else:
            s_sensor += " FAIL!\n"
        log_callable(s_sensor + s_tof + s_offsets)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    # Convert raw ADC traces into volts
    ds = ds.assign({key: -ds[key] / 2**12 for key in ("adc", "adc_single_run")})
    # Add the IQ amplitude to the dataset
    ds = ds.assign({"adc_abs": np.abs(ds["adc"])})
    ds.adc_abs.attrs = {"long_name": "IQ amplitude", "units": "V"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the sensor frequency and FWHM for each sensor in the dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node_parameters : Parameters
        Parameters related to the node, including whether state discrimination is used.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    """
    ds_fit = ds
    # Filter the data to get the pulse arrival time
    ds_fit["filtered_adc"] = xr.apply_ufunc(_filter_adc_signal, ds_fit.adc_abs)
    # Detect the pulse arrival times
    ds_fit["threshold"] = (
        ds_fit["filtered_adc"][:, 100:].mean("readout_time") + ds_fit["filtered_adc"][:, :-100].mean("readout_time")
    ) / 2
    ds_fit["delay"] = (ds_fit["filtered_adc"] > ds_fit["threshold"]).where(True).idxmax("readout_time")
    ds_fit["delay"] = np.round(ds_fit["delay"] / 4) * 4
    ds_fit.delay.attrs = {"long_name": "TOF to add", "units": "ns"}
    # Get the controller of each sensor
    ds_fit = ds_fit.assign_coords(
        {
            "con": (
                ["sensor"],
                [node.machine.sensor_dots[s.name].readout_resonator.opx_input.controller_id for s in node.namespace["sensors"]],
            )
        }
    )

    ds_fit = ds_fit.assign_coords(
        {
            "offset_combined": (ds_fit.adc.mean(dim = "readout_time")) 
        }
    )
    mean_offset = {}
    for con in np.unique(ds_fit.con.values):
        mean_offset[con] = ds_fit.where(ds_fit.con == con).offset_combined.mean(dim = "sensor").values

    offsets_list = []
    for s in ds_fit.sensor.values: 
        offsets_list.append(mean_offset[str(ds_fit.sel(sensor = s).con.values)])
    ds_fit = ds_fit.assign({"offset_mean": xr.DataArray(offsets_list, coords=dict(sensor=ds_fit.sensor.data))})
    ds_fit.offset_mean.attrs = {"long_name": "Mean offset", "units": "V"}
    # Assess whether the fit was successful or not
    nan_success = np.isnan(ds_fit.delay.data) & np.isnan(ds_fit.offset_mean.data)
    offset_success = np.abs(ds_fit.offset_mean) < 0.5
    
    success_criteria = ~nan_success & offset_success.data
    ds_fit = ds_fit.assign_coords(success=("sensor", success_criteria))
    # Populate the FitParameters class with fitted values
    fit_results = {
        s: FitParameters(
            offset_to_add=float(ds_fit.sel(sensor=s).offset_mean),
            tof_to_add=int(ds_fit.sel(sensor=s).delay),
            success=bool(ds_fit.sel(sensor=s).success.values),
        )
        for s in ds_fit.sensor.values
    }
    node.outcomes = {s: "successful" if fit_results[s].success else "fail" for s in ds_fit.sensor.values}

    return ds_fit, fit_results

def _filter_adc_signal(data, window_length=11, polyorder=3):
    """
    Applies a Savitzky-Golay filter to smooth the absolute IQ signal in the dataset.
    """
    return savgol_filter(data, window_length, polyorder)


