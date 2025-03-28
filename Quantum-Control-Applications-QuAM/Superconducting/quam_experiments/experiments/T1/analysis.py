import logging
import numpy as np
import xarray as xr
from dataclasses import dataclass
from typing import Optional, Tuple
from quam_experiments.experiments.T1.parameters import Parameters
from quam_experiments.analysis.fit import fit_decay_exp


@dataclass
class T1Fit:
    """Stores the relevant T1 experiment fit parameters for a single qubit"""

    t1: float
    t1_error: float
    success: bool
    qubit_name: Optional[str] = ""


def log_fitted_results(ds: xr.Dataset, logger=None):
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
    if logger is None:
        logger = logging.getLogger(__name__)
    for q in ds.qubit.values:
        if ds.sel(qubit=q).success.values:
            logger.info(
                f"T1 for qubit {q} : {1e-3 * ds.sel(qubit=q).tau.values:.2f} +/- {1e-3 * ds.sel(qubit=q).tau_error.values:.2f} us --> SUCCESS!"
            )
        else:
            logger.error(
                f"T1 for qubit {q} : {1e-3 * ds.sel(qubit=q).tau.values:.2f} +/- {1e-3 * ds.sel(qubit=q).tau_error.values:.2f} us --> FAIL!"
            )


def fit_t1_decay(ds: xr.Dataset, node_parameters: Parameters) -> Tuple[xr.Dataset, dict[str, T1Fit]]:
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
    fit_results = _fit_t1_with_exponential_decay(ds, node_parameters.use_state_discrimination)
    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(fit_results.to_dataset(name="ds_fit"))

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
    tau = -1 / fit.sel(fit_vals="decay")
    fit = fit.assign_coords(tau=("qubit", tau.ds_fit.data))
    fit.tau.attrs = {"long_name": "T1", "units": "ns"}
    # Get the error on T1
    tau_error = -tau * (np.sqrt(fit.sel(fit_vals="decay_decay")) / fit.sel(fit_vals="decay"))
    fit = fit.assign_coords(tau_error=("qubit", tau_error.ds_fit.data))
    fit.tau_error.attrs = {"long_name": "T1 error", "units": "ns"}
    # Assess whether the fit was successful or not
    success_criteria = (tau.ds_fit.data > 16) & (tau_error.ds_fit.data / tau.ds_fit.data < 1)
    fit = fit.assign_coords(success=("qubit", success_criteria))

    fit_results = {
        q: T1Fit(
            qubit_name=q.item(),
            t1=fit.sel(qubit=q).tau.values.__float__(),
            t1_error=fit.sel(qubit=q).tau_error.values.__float__(),
            success=fit.sel(qubit=q).success.values.__bool__(),
        )
        for q in fit.qubit.values
    }
    return fit, fit_results
