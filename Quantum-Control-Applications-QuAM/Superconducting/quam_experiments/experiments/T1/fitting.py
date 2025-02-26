import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import xarray as xr

from quam_experiments.experiments.T1.parameters import Parameters
from quam_experiments.analysis.fit import fit_decay_exp


@dataclass
class T1Fit:
    """Stores the relevant Ramsey experiment fit parameters for a single qubit"""

    freq_offset: float
    decay: float
    decay_error: float

    qubit_name: Optional[str] = ""
    raw_fit_results: Optional[xr.Dataset] = None

    def log_frequency_offset(self, logger=None):
        if logger is None:
            logger = logging.getLogger(__name__)
        logger.info(f"Frequency offset for qubit {self.qubit_name} : {self.freq_offset / 1e6:.2f} MHz ")

    def log_t2(self, logger=None):
        if logger is None:
            logger = logging.getLogger(__name__)
        logger.info(f"T2* for qubit {self.qubit_name} : {1e6 * self.decay:.2f} us")


def fit_t1_decay(ds: xr.Dataset, node_parameters: Parameters) -> dict[str, T1Fit]:
    """
    Fit the frequency detuning and T2 decay of the Ramsey oscillations for each qubit.

    Returns:
        dict: Dictionary containing fit results.
    """

    # Fit the exponential decay
    fit_results = fit_t1_with_exponential_decay(ds, node_parameters.use_state_discrimination)
    # Extract the relevant fitted parameters
    fit_results, tau, tau_error = extract_relevant_fit_parameters(fit_results)

    return fit_results.to_dataset(name="fit_results")

def fit_t1_with_exponential_decay(ds, use_state_discrimination):
    """
    Perform the fitting process based on the state discrimination flag.

    Parameters:
        ds (xarray.Dataset): Input dataset.
        use_state_discrimination (bool): Flag indicating whether to use state discrimination.

    Returns:
        xarray.DataArray: Fit results.
    """

    if use_state_discrimination:
        fit = fit_decay_exp(ds.state, "idle_time")
    else:
        fit = fit_decay_exp(ds.I, "idle_time")
    return fit


def extract_relevant_fit_parameters(fit):
    """
    Add metadata to the dataset and fit results.

    Parameters:
        fit (xarray.DataArray): Fit results.

    Returns:
        tuple: Processed frequency, decay, tau, and tau_error.
    """
    # Add metadata to fit results
    fit.attrs = {"long_name": "time", "units": "ns"}
    # Get the fitted T1
    tau = -1 / fit.sel(fit_vals="decay")
    fit = fit.assign_coords(tau=("qubit", tau.data))
    fit.tau.attrs = {"long_name": "T1", "units": "ns"}
    # Get the error on T1
    tau_error = -tau * (np.sqrt(fit.sel(fit_vals="decay_decay")) / fit.sel(fit_vals="decay"))
    fit = fit.assign_coords(tau_error=("qubit", tau_error.data))
    fit.tau_error.attrs = {"long_name": "T1 error", "units": "ns"}
    # Assess whether the fit was successful or not
    success_criteria = (tau.values > 0) & (tau_error.values / tau.values < 1)
    fit = fit.assign_coords(success=("qubit", success_criteria))

    return fit, tau, tau_error