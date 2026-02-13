"""Time Rabi chevron parity difference calibration.

Fits 2D chevron (parity diff vs duration × frequency) to extract resonant frequency
and π-time via generalized Rabi formula. Backends: scipy curve_fit or NumPyro MCMC.
"""

from calibration_utils.time_rabi_chevron_parity_diff.parameters import Parameters
from calibration_utils.time_rabi_chevron_parity_diff.analysis import (
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
)
from calibration_utils.time_rabi_chevron_parity_diff.plotting import plot_raw_data_with_fit

__all__ = [
    "Parameters",
    "process_raw_dataset",
    "fit_raw_data",
    "log_fitted_results",
    "plot_raw_data_with_fit",
]
