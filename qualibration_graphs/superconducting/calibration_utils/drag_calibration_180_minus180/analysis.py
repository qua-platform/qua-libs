import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    alpha: float
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

    Returns:
    --------
    None
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_alpha = f"\tDRAG coefficient alpha: {fit_results[q]['alpha']:.2f}\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_alpha)
    pass


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    # Add the DRAG coefficient alpha to the dataset
    alpha = np.array(
        [q.xy.operations[node.parameters.operation].alpha * ds.alpha_prefactor for q in node.namespace["qubits"]]
    )
    ds = ds.assign_coords(alpha=(["qubit", "alpha_prefactor"], alpha))
    ds.alpha.attrs = {"long_name": "DRAG coefficient alpha"}
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
    # Get the average along the number of pulses axis to identify the best DRAG coefficient
    if node.parameters.use_state_discrimination:
        ds_fit["averaged_data"] = ds.state.mean(dim="nb_of_pulses")
    else:
        ds_fit["averaged_data"] = ds.I.mean(dim="nb_of_pulses")
    ds_fit["optimal_alpha"] = ds_fit["averaged_data"].alpha[:, ds_fit["averaged_data"].argmin("alpha_prefactor")]

    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)

    return ds_fit, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    # Assess whether the fit was successful or not
    nan_success = np.isnan(fit.optimal_alpha)
    snr_success = (
        np.abs(
            (fit["averaged_data"].min("alpha_prefactor") - fit["averaged_data"].mean("alpha_prefactor"))
            / fit["averaged_data"].std("alpha_prefactor")
        )
        > 2
    )
    success_criteria = ~nan_success & snr_success
    fit = fit.assign({"success": success_criteria})
    fit_results = {
        q: FitParameters(
            alpha=float(fit.sel(qubit=q)["optimal_alpha"]),
            success=bool(fit.sel(qubit=q).success),
        )
        for q in fit.qubit.values
    }
    node.outcomes = {q: "successful" if fit_results[q].success else "fail" for q in fit.qubit.values}
    return fit, fit_results
