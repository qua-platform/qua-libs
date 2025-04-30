import logging
import numpy as np
import xarray as xr
from dataclasses import dataclass
from typing import Tuple
from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from qualibration_libs.analysis import fit_decay_exp


@dataclass
class T1Fit:
    """Stores the relevant T1 experiment fit parameters for a single qubit"""

    t1: float
    t1_error: float
    success: bool


def log_fitted_results(ds: xr.Dataset, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit xarray Dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the fitted results for all qubits.
        Expected variables: 'tau', 'tau_error', 'success'.
        Expected coordinate: 'qubit'.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.

    Returns:
    --------
    None

    Example:
    --------
        >>> log_fitted_results(ds)
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in ds.qubit.values:
        if ds.sel(qubit=q).success.values:
            log_callable(
                f"T1 for qubit {q} : {1e-3 * ds.sel(qubit=q).tau.values:.2f} +/- {1e-3 * ds.sel(qubit=q).tau_error.values:.2f} us --> SUCCESS!"
            )
        else:
            log_callable(
                f"T1 for qubit {q} : {1e-3 * ds.sel(qubit=q).tau.values:.2f} +/- {1e-3 * ds.sel(qubit=q).tau_error.values:.2f} us --> FAIL!"
            )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, T1Fit]]:
    """
    Fit the T1 relaxation time for each qubit according to ``a * np.exp(t * decay) + offset``.

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

    # Fit the exponential decay
    if node.parameters.use_state_discrimination:
        fit_data = fit_decay_exp(ds.state, "idle_time")
    else:
        fit_data = fit_decay_exp(ds.I, "idle_time")

    ds_fit = xr.merge([ds, fit_data.rename("fit_data")])
    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit)

    return fit_data, fit_results


def _fit_t1_with_exponential_decay(ds, use_state_discrimination):
    """Perform the fitting process based on the state discrimination flag."""
    if use_state_discrimination:
        fit = fit_decay_exp(ds.state, "idle_time")
    else:
        fit = fit_decay_exp(ds.I, "idle_time")
    return fit


def _extract_relevant_fit_parameters(fit: xr.Dataset):
    """Add metadata to the dataset and fit results."""
    # Add metadata to fit results
    fit.attrs = {"long_name": "time", "units": "ns"}
    # Get the fitted T1
    tau = -1 / fit.fit_data.sel(fit_vals="decay")
    fit = fit.assign_coords(tau=("qubit", tau.data))
    fit.tau.attrs = {"long_name": "T1", "units": "ns"}
    # Get the error on T1
    tau_error = -tau * (np.sqrt(fit.fit_data.sel(fit_vals="decay_decay")) / fit.fit_data.sel(fit_vals="decay"))
    fit = fit.assign_coords(tau_error=("qubit", tau_error.data))
    fit.tau_error.attrs = {"long_name": "T1 error", "units": "ns"}
    # Assess whether the fit was successful or not
    success_criteria = (tau.data > 16) & (tau_error.data / tau.data < 1)
    fit = fit.assign_coords(success=("qubit", success_criteria))

    fit_results = {
        q: T1Fit(
            t1=fit.sel(qubit=q).tau.values.__float__(),
            t1_error=fit.sel(qubit=q).tau_error.values.__float__(),
            success=fit.sel(qubit=q).success.values.__bool__(),
        )
        for q in fit.qubit.values
    }
    return fit, fit_results
