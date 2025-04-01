from typing import List, Tuple

from scipy.signal import savgol_filter
import numpy as np
import xarray as xr

from qm import QmJob
from qualang_tools.config import ReadoutResonator

from quam_builder.architecture.superconducting.qubit import AnyTransmon
from qualibration_libs.save_utils import fetch_results_as_xarray
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.qua_datasets import add_amplitude_and_phase, convert_IQ_to_V
from quam_experiments.analysis.fit import peaks_dips
from quam_config.instrument_limits import instrument_limits


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    tof_to_add: int
    success: bool


def log_fitted_results(fit_results: Dict, logger=None):
    """
    Logs the node-specific fitted results for all qubits from the fit xarray Dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the fitted results for all qubits.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_tof = f"\tTime of flight to add: {fit_results[q]['tof_to_add']:.0f} ns\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        logger.info(s_qubit + s_tof)


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
    ds_fit = ds_fit.assign_coords(
        {
            "con": (
                ["qubit"],
                [node.machine.qubits[q.name].resonator.opx_input.controller_id for q in node.namespace["qubits"]],
            )
        }
    )

    # Assess whether the fit was successful or not
    nan_success = np.isnan(ds_fit.delay.data)
    success_criteria = ~nan_success
    ds_fit = ds_fit.assign_coords(success=("qubit", success_criteria))
    # Populate the FitParameters class with fitted values
    fit_results = {
        q: FitParameters(
            tof_to_add=int(ds_fit.sel(qubit=q).delay),
            success=bool(ds_fit.sel(qubit=q).success.values),
        )
        for q in ds_fit.qubit.values
    }
    node.outcomes = {q: "successful" if fit_results[q].success else "fail" for q in ds_fit.qubit.values}

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
