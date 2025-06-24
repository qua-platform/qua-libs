import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from quam_config.instrument_limits import instrument_limits
from qualibration_libs.analysis import fit_oscillation_decay_exp, oscillation_decay_exp, peaks_dips


@dataclass
class FitParameters:
    """Stores the relevant Ramsey vs flux experiment fit parameters for a single qubit"""

    success: bool
    quad_term: float
    flux_offset: float
    freq_offset: float
    flux_offset: float
    t2_star: np.ndarray


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
    pass


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
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
    # # TODO: explain the data analysis
    fit_data = fit_oscillation_decay_exp(ds.state, "idle_times")
    fit_data.attrs = {"long_name": "time", "units": "µs"}
    fitted = oscillation_decay_exp(
        ds.state.idle_times,
        fit_data.sel(fit_vals="a"),
        fit_data.sel(fit_vals="f"),
        fit_data.sel(fit_vals="phi"),
        fit_data.sel(fit_vals="offset"),
        fit_data.sel(fit_vals="decay"),
    )

    frequency = fit_data.sel(fit_vals="f")
    frequency.attrs = {"long_name": "frequency", "units": "MHz"}

    decay = fit_data.sel(fit_vals="decay")
    decay.attrs = {"long_name": "decay", "units": "nSec"}

    tau = 1 / fit_data.sel(fit_vals="decay")
    tau.attrs = {"long_name": "T2*", "units": "uSec"}

    frequency = frequency.where(frequency > 0, drop=True)

    fitvals = frequency.polyfit(dim="flux_bias", deg=2)
    flux = frequency.flux_bias

    a = {}
    flux_offset = {}
    freq_offset = {}
    t2_star = {}

    qubits = ds.qubit.values

    for q in qubits:
        a[q] = float(-1e6 * fitvals.sel(qubit=q, degree=2).polyfit_coefficients.values)
        flux_offset[q] = float(
            (
                -0.5
                * fitvals.sel(qubit=q, degree=1).polyfit_coefficients
                / fitvals.sel(qubit=q, degree=2).polyfit_coefficients
            ).values
        )
        freq_offset[q] = 1e6 * (
            flux_offset[q] ** 2 * float(fitvals.sel(qubit=q, degree=2).polyfit_coefficients.values)
            + flux_offset[q] * float(fitvals.sel(qubit=q, degree=1).polyfit_coefficients.values)
            + float(fitvals.sel(qubit=q, degree=0).polyfit_coefficients.values)
        )
        t2_star[q] = tau.sel(qubit=q).values

    ds_fit = ds.merge(fit_data.rename("fit_results"))

    # Add a, flux_offset, and freq_offset as data variables in the dataset
    ds_fit["quad_term"] = xr.DataArray([a[q] for q in qubits], dims=["qubit"], coords={"qubit": qubits})
    ds_fit["flux_offset"] = xr.DataArray([flux_offset[q] for q in qubits], dims=["qubit"], coords={"qubit": qubits})
    ds_fit["freq_offset"] = xr.DataArray([freq_offset[q] for q in qubits], dims=["qubit"], coords={"qubit": qubits})
    ds_fit["artifitial_detuning"] = xr.DataArray(
        node.parameters.frequency_detuning_in_mhz, dims=["qubit"], coords={"qubit": qubits}
    )

    fit_results = {
        q: FitParameters(
            success=True,
            quad_term=a[q],
            flux_offset=flux_offset[q],
            freq_offset=freq_offset[q],
            t2_star=t2_star[q],
        )
        for q in ds_fit.qubit.values
    }

    return ds_fit, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    pass
