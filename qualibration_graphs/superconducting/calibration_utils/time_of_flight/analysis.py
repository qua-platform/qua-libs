from scipy.signal import savgol_filter
import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    tof_to_add: int
    offset_I_to_add: float
    offset_Q_to_add: float
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_tof = f"\tTime of flight to add: {fit_results[q]['tof_to_add']:.0f} ns\n"
        s_offsets = f"\tOffsets to add for 'I': {fit_results[q]['offset_I_to_add'] * 1e3:.1f} mV & for Q: {fit_results[q]['offset_Q_to_add'] * 1e3:.1f} mV\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_tof + s_offsets)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    # Convert raw ADC traces into volts
    ds = ds.assign({key: -ds[key] / 2**12 for key in ("adcI", "adcQ", "adc_single_runI", "adc_single_runQ")})
    # Add the IQ amplitude to the dataset
    ds = ds.assign({"IQ_abs": np.sqrt(ds["adcI"] ** 2 + ds["adcQ"] ** 2)})
    ds.IQ_abs.attrs = {"long_name": "IQ amplitude", "units": "V"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the qubit frequency and FWHM for each qubit in the dataset.

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
    ds_fit["filtered_adc"] = xr.apply_ufunc(_filter_adc_signal, ds_fit.IQ_abs)
    # Detect the pulse arrival times
    ds_fit["threshold"] = (
        ds_fit["filtered_adc"][:, 100:].mean("readout_time") + ds_fit["filtered_adc"][:, :-100].mean("readout_time")
    ) / 2
    ds_fit["delay"] = (ds_fit["filtered_adc"] > ds_fit["threshold"]).where(True).idxmax("readout_time")
    ds_fit["delay"] = np.round(ds_fit["delay"] / 4) * 4
    ds_fit.delay.attrs = {"long_name": "TOF to add", "units": "ns"}
    # Get the controller of each qubit
    _check_resonator_inputs(node)
    ds_fit = ds_fit.assign_coords(
        {
            "con": (
                ["qubit"],
                [node.machine.qubits[q.name].resonator.opx_input_I.controller_id for q in node.namespace["qubits"]],
            )
        }
    )
    # TODO: to test
    # Get the offsets
    ds_fit = ds_fit.assign_coords(
        {
            "offsets_I": ds_fit.adcI.mean(dim="readout_time"),
            "offsets_Q": ds_fit.adcQ.mean(dim="readout_time"),
        }
    )
    # Get the averaged offsets for each controller
    mean_offset_I = {}
    mean_offset_Q = {}
    for con in np.unique(ds_fit.con.values):
        mean_offset_I[con] = ds_fit.where(ds_fit.con == con).offsets_I.mean(dim="qubit").values
        mean_offset_Q[con] = ds_fit.where(ds_fit.con == con).offsets_Q.mean(dim="qubit").values
    # Assign the averaged offsets to each qubit based on their controller
    test_I = []
    test_Q = []
    for q in ds_fit.qubit.values:
        test_I.append(mean_offset_I[str(ds_fit.sel(qubit=q).con.values)])
        test_Q.append(mean_offset_Q[str(ds_fit.sel(qubit=q).con.values)])
    ds_fit = ds_fit.assign({"offsets_I_mean": xr.DataArray(test_I, coords=dict(qubit=ds_fit.qubit.data))})
    ds_fit = ds_fit.assign({"offsets_Q_mean": xr.DataArray(test_Q, coords=dict(qubit=ds_fit.qubit.data))})
    ds_fit.offsets_I_mean.attrs = {"long_name": "Mean offset 'I'", "units": "V"}
    ds_fit.offsets_Q_mean.attrs = {"long_name": "Mean offset 'Q'", "units": "V"}
    # Assess whether the fit was successful or not
    nan_success = (
        np.isnan(ds_fit.delay.data) & np.isnan(ds_fit.offsets_I_mean.data) & np.isnan(ds_fit.offsets_Q_mean.data)
    )
    offset_success = (np.abs(ds_fit.offsets_I_mean) < 0.5) & (np.abs(ds_fit.offsets_Q_mean) < 0.5)
    success_criteria = ~nan_success & offset_success.data
    ds_fit = ds_fit.assign_coords(success=("qubit", success_criteria))
    # Populate the FitParameters class with fitted values
    fit_results = {
        q: FitParameters(
            offset_I_to_add=float(ds_fit.sel(qubit=q).offsets_I_mean),
            offset_Q_to_add=float(ds_fit.sel(qubit=q).offsets_Q_mean),
            tof_to_add=int(ds_fit.sel(qubit=q).delay),
            success=bool(ds_fit.sel(qubit=q).success.values),
        )
        for q in ds_fit.qubit.values
    }
    node.outcomes = {q: "successful" if fit_results[q].success else "fail" for q in ds_fit.qubit.values}

    return ds_fit, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    pass
    # return fit, fit_results


def _filter_adc_signal(data, window_length=11, polyorder=3):
    """
    Applies a Savitzky-Golay filter to smooth the absolute IQ signal in the dataset.
    """
    return savgol_filter(data, window_length, polyorder)


def _check_resonator_inputs(node: QualibrationNode):
    for q in node.namespace["qubits"]:
        con_I = node.machine.qubits[q.name].resonator.opx_input_I.controller_id
        con_Q = node.machine.qubits[q.name].resonator.opx_input_Q.controller_id
        if con_I != con_Q:
            raise RuntimeError(f"{q.resonator.name} doesn't have its two outputs connected to the same controller.")
