import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from qualibrate import QualibrationNode
from qualibration_libs.analysis import fit_oscillation
from qualibration_libs.data import convert_IQ_to_V
from quam_config.instrument_limits import instrument_limits
from scipy.optimize import curve_fit


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    success: bool
    flux_delay: int


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
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])

    data = "state" if hasattr(ds, "state") else "I"

    difference = ds[data].sel(init_state="e") - ds[data].sel(init_state="g")

    if data == "I":
        difference -= difference.mean()

    ds["difference"] = difference
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

    dfit = ds.groupby("qubit").apply(fit_routine, node=node)

    fit_results = _extract_relevant_fit_parameters(dfit, node)

    return dfit, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    fit_results = {}
    for q in fit.qubit.data:
        fit_results[q] = FitParameters(
            success=fit.success.sel(qubit=q).data, flux_delay=fit.flux_delay.sel(qubit=q).data
        )
    return fit_results


def fit_routine(da, node):

    x = da.relative_time.data
    y = da.difference.data[0]

    try:
        sign_changes = np.sign(y)
        crossings = np.where(np.diff(sign_changes) != 0)[0]
        assert len(crossings) == 2, "Expected exactly two sign change points in the data."
        flux_delay = x[(crossings[1] + crossings[0]) // 2]
        da = da.assign(flux_delay=flux_delay)
        da = da.assign(success=True)

    except AssertionError as e:
        print(f"Error processing {da.qubit.data}: {e}")
        flux_delay = 0
        da = da.assign(flux_delay=flux_delay)
        da = da.assign(success=False)
        return da

    print(f"Flux delay for {da.qubit.data}: {flux_delay:.2f} ns")

    return da
