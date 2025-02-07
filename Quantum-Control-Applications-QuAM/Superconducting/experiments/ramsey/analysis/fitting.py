import logging
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import xarray as xr

from quam_libs.components import Transmon
from quam_libs.experiments.ramsey.parameters import Parameters
from quam_libs.lib.fit import fit_oscillation_decay_exp


@dataclass
class RamseyFit:
    """ Stores the relevant Ramsey experiment fit parameters for a single qubit"""
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


def fit_frequency_detuning_and_t2_decay(ds: xr.Dataset, node_parameters: Parameters) \
        -> dict[str, RamseyFit]:
    """
    Fit the frequency detuning and T2 decay of the Ramsey oscillations for each qubit.

    Returns:
        dict: Dictionary containing fit results.
    """
    fit = fit_ramsey_oscillations_with_exponential_decay(ds, node_parameters.use_state_discrimination)

    frequency, decay, tau, tau_error = extract_relevant_fit_parameters(fit)

    detuning = int(node_parameters.frequency_detuning_in_mhz * 1e6)

    freq_offset, decay, decay_error = calculate_fit_results(
        frequency, tau, tau_error, fit, detuning
    )

    fits = {
        q.name: RamseyFit(
            qubit_name = q.item(),
            freq_offset=1e9 * freq_offset.loc[q].values,
            decay=decay.loc[q].values,
            decay_error=decay_error.loc[q].values,
            raw_fit_results=fit.to_dataset(name="fit")
        )

        for q in freq_offset.qubit
    }

    return fits

def fit_ramsey_oscillations_with_exponential_decay(ds, use_state_discrimination):
    """
    Perform the fitting process based on the state discrimination flag.

    Parameters:
        ds (xarray.Dataset): Input dataset.
        use_state_discrimination (bool): Flag indicating whether to use state discrimination.

    Returns:
        xarray.DataArray: Fit results.
    """
    if use_state_discrimination:
        fit = fit_oscillation_decay_exp(ds.state, "time")
    else:
        fit = fit_oscillation_decay_exp(ds.I, "time")
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
    fit.attrs = {"long_name": "time", "units": "µs"}

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

    return frequency, decay, tau, tau_error


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
    within_detuning = (1e9 * frequency < 2 * detuning).mean(dim="sign") == 1
    positive_shift = frequency.sel(sign=1) > frequency.sel(sign=-1)
    freq_offset = (
        within_detuning * (frequency * fit.sign).mean(dim="sign")
        + ~within_detuning * positive_shift * frequency.mean(dim="sign")
        - ~within_detuning * ~positive_shift * frequency.mean(dim="sign")
    )

    decay = 1e-9 * tau.mean(dim="sign")
    decay_error = 1e-9 * tau_error.mean(dim="sign")

    return freq_offset, decay, decay_error
