import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from qualibration_libs.analysis import fit_oscillation
from quam_config.instrument_limits import instrument_limits


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    opt_amp_prefactor: float
    opt_amp: float
    operation: str
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
        s_amp = f"The calibrated {fit_results[q]['operation']} amplitude: {1e3 * fit_results[q]['opt_amp']:.2f} mV (x{fit_results[q]['opt_amp_prefactor']:.2f})\n "
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_amp)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    full_amp = np.array(
        [ds.amp_prefactor * q.xy.operations[node.parameters.operation].amplitude for q in node.namespace["qubits"]]
    )
    ds = ds.assign_coords(full_amp=(["qubit", "amp_prefactor"], full_amp))
    ds.full_amp.attrs = {"long_name": "pulse amplitude", "units": "V"}
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
    if node.parameters.max_number_pulses_per_sweep == 1:
        ds_fit = ds.sel(nb_of_pulses=1)
        # Fit the power Rabi oscillations
        if node.parameters.use_state_discrimination:
            fit_vals = fit_oscillation(ds_fit.state, "amp_prefactor")
        else:
            fit_vals = fit_oscillation(ds_fit.I, "amp_prefactor")

        ds_fit = xr.merge([ds, fit_vals.rename("fit")])
    else:
        ds_fit = ds
        # Get the average along the number of pulses axis to identify the best pulse amplitude
        if node.parameters.use_state_discrimination:
            ds_fit["data_mean"] = ds.state.mean(dim="nb_of_pulses")
        else:
            ds_fit["data_mean"] = ds.I.mean(dim="nb_of_pulses")
        if (ds.nb_of_pulses.data[0] % 2 == 0 and node.parameters.operation == "x180") or (
            ds.nb_of_pulses.data[0] % 2 != 0 and node.parameters.operation != "x180"
        ):
            ds_fit["opt_amp_prefactor"] = ds_fit["data_mean"].idxmin(dim="amp_prefactor")
        else:
            ds_fit["opt_amp_prefactor"] = ds_fit["data_mean"].idxmax(dim="amp_prefactor")

    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    limits = [instrument_limits(q.xy) for q in node.namespace["qubits"]]
    if node.parameters.max_number_pulses_per_sweep == 1:
        # Process the fit parameters to get the right amplitude
        phase = fit.fit.sel(fit_vals="phi") - np.pi * (fit.fit.sel(fit_vals="phi") > np.pi / 2)
        factor = (np.pi - phase) / (2 * np.pi * fit.fit.sel(fit_vals="f"))
        fit = fit.assign({"opt_amp_prefactor": factor})
        fit.opt_amp_prefactor.attrs = {
            "long_name": "factor to get a pi pulse",
            "units": "Hz",
        }
        current_amps = xr.DataArray(
            [q.xy.operations[node.parameters.operation].amplitude for q in node.namespace["qubits"]],
            coords=dict(qubit=fit.qubit.data),
        )
        opt_amp = factor * current_amps
        fit = fit.assign({"opt_amp": opt_amp})
        fit.opt_amp.attrs = {"long_name": "x180 pulse amplitude", "units": "V"}

    else:
        current_amps = xr.DataArray(
            [q.xy.operations[node.parameters.operation].amplitude for q in node.namespace["qubits"]],
            coords=dict(qubit=fit.qubit.data),
        )
        fit = fit.assign({"opt_amp": fit.opt_amp_prefactor * current_amps})
        fit.opt_amp.attrs = {
            "long_name": f"{node.parameters.operation} pulse amplitude",
            "units": "V",
        }

    # Assess whether the fit was successful or not
    nan_success = np.isnan(fit.opt_amp_prefactor) | np.isnan(fit.opt_amp)
    amp_success = fit.opt_amp < limits[0].max_x180_wf_amplitude
    success_criteria = ~nan_success & amp_success
    fit = fit.assign({"success": success_criteria})
    # Populate the FitParameters class with fitted values
    fit_results = {
        q: FitParameters(
            opt_amp_prefactor=fit.sel(qubit=q).opt_amp_prefactor.values.__float__(),
            opt_amp=fit.sel(qubit=q).opt_amp.values.__float__(),
            operation=node.parameters.operation,
            success=fit.sel(qubit=q).success.values.__bool__(),
        )
        for q in fit.qubit.values
    }
    return fit, fit_results
