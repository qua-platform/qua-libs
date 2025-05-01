import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from qualibration_libs.analysis import fit_oscillation_decay_exp


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    freq_offset: float
    decay: float
    decay_error: float
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
        s_detuning = f"\tDetuning to correct: {1e-6 * fit_results[q]['freq_offset']:.3f} MHz | "
        s_T2 = f"T2*: {1e6 * fit_results[q]['decay']:.1f} µs\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_detuning + s_T2)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the frequency detuning and T2 decay of the Ramsey oscillations for each qubit.

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
    if node.parameters.use_state_discrimination:
        fit = fit_oscillation_decay_exp(ds.state, "idle_time")
    else:
        fit = fit_oscillation_decay_exp(ds.I, "idle_time")

    ds_fit = xr.merge([ds, fit.rename("fit")])

    ds_fit, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return ds_fit, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    # Add calculated metadata to the dataset
    frequency = fit.sel(fit_vals="f")
    frequency.attrs = {"long_name": "frequency", "units": "MHz"}
    frequency = frequency.where(frequency > 0, drop=True)

    decay = fit.sel(fit_vals="decay")
    decay.attrs = {"long_name": "decay", "units": "nSec"}

    decay_res = fit.sel(fit_vals="decay_decay")
    decay_res.attrs = {"long_name": "decay residual", "units": "nSec"}

    tau = 1 / decay
    tau.attrs = {"long_name": "T2*", "units": "uSec"}

    tau_error = tau * (np.sqrt(decay_res) / decay)
    tau_error.attrs = {"long_name": "T2* error", "units": "uSec"}

    detuning = int(node.parameters.frequency_detuning_in_mhz * 1e6)

    freq_offset, decay, decay_error = calculate_fit_results(frequency, tau, tau_error, fit, detuning)
    # Assess whether the fit was successful or not
    nan_success = np.isnan(freq_offset.fit) | np.isnan(decay.fit)
    success_criteria = ~nan_success
    fit = fit.assign({"success": success_criteria})
    # Populate the FitParameters class with fitted values
    fit_results = {
        q: FitParameters(
            freq_offset=1e9 * float(freq_offset.sel(qubit=q).fit),
            decay=float(decay.sel(qubit=q).fit),
            decay_error=float(decay_error.sel(qubit=q).fit),
            success=bool(fit.sel(qubit=q).success.values),
        )
        for q in fit.qubit.values
    }
    return fit, fit_results


def calculate_fit_results(frequency, tau, tau_error, fit, detuning):
    """
    Calculate fit results such as frequency offset, decay, and decay error.

    Parameters:
        frequency (xarray.DataArray): Frequency data.
        tau (xarray.DataArray): Tau values.
        tau_error (xarray.DataArray): Tau error values.
        fit (xarray.DataArray): Fit results.
        detuning (float): Detuning parameter in Hz.

    Returns:
        tuple: Frequency offset, decay, and decay error.
    """
    within_detuning = (1e9 * frequency < 2 * detuning).mean(dim="detuning_signs") == 1
    positive_shift = frequency.sel(detuning_signs=1) > frequency.sel(detuning_signs=-1)
    freq_offset = (
        within_detuning * (frequency * fit.detuning_signs).mean(dim="detuning_signs")
        + ~within_detuning * positive_shift * frequency.mean(dim="detuning_signs")
        - ~within_detuning * ~positive_shift * frequency.mean(dim="detuning_signs")
    )

    decay = 1e-9 * tau.mean(dim="detuning_signs")
    decay_error = 1e-9 * tau_error.mean(dim="detuning_signs")

    return freq_offset, decay, decay_error
