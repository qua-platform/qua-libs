"""Analysis utilities for two-qubit randomized benchmarking experiments.

This module provides functions for processing and analyzing raw RB data,
including dataset processing and result logging.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode

# @dataclass
# class FitResults:
#     """Stores the relevant fit parameters for a single qubit pair in an RB experiment"""

#     optimal_amplitude: float
#     success: bool


def log_fitted_results(fit_results: Dict[str, float], log_callable=None):
    """
    Logs the node-specific fitted results for all qubit pairs.

    Parameters:
    -----------
    fit_results : Dict[str, float]
        Dictionary containing floats for each qubit pair.
    log_callable : callable, optional
        Logger for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qp_name, fit_result in fit_results.items():
        s_qubit = f"Results for qubit pair {qp_name}: "

        s_alpha = f"\tFitted alpha: {fit_result['alpha']:.6f} a.u."
        s_fidelity = f"\tFitted fidelity: {100*fit_result['fidelity']:.6f} %"

        if fit_result["success"]:
            s_qubit += "SUCCESS!\n"
        else:
            s_qubit += "FAIL!\n"

        log_message = s_qubit + s_alpha + s_fidelity

        log_callable(log_message)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """
    Process the raw dataset by adding amplitude and detuning coordinates.

    Parameters:
    -----------
    ds : xr.Dataset
        Raw dataset from the experiment
    node : QualibrationNode
        The calibration node containing qubit pairs information

    Returns:
    --------
    xr.Dataset
        Processed dataset with additional coordinates
    """
    ds = node.results["ds_raw"]
    # Assume ds is your input dataset and ds['state'] is your DataArray
    state = ds["state"]  # shape: (qubit, shots, sequence, depths)

    # Outcome labels for 2-qubit states
    labels = ["00", "01", "10", "11"]

    # Create a list of DataArrays: one for each outcome
    probs = [state == i for i in range(4)]

    # Stack along a new outcome dimension
    probs = xr.concat(probs, dim="outcome")

    # Assign outcome labels
    probs = probs.assign_coords(outcome=("outcome", labels))

    probs_00 = probs.sel(outcome="00")
    probs_00 = probs_00.rename({"shots": "average", "sequence": "repeat", "depths": "circuit_depth"})
    probs_00 = probs_00.transpose("qubit_pair", "repeat", "circuit_depth", "average")

    probs_00 = probs_00.astype(int)

    ds_transposed = ds.rename({"shots": "average", "sequence": "repeat", "depths": "circuit_depth"})
    ds_transposed = ds_transposed.transpose("qubit_pair", "repeat", "circuit_depth", "average")

    return ds_transposed


# def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitResults]]:
#     """
#     Fit the CZ conditional phase data for each qubit pair.

#     Parameters:
#     -----------
#     ds : xr.Dataset
#         Dataset containing the processed data.
#     node : QualibrationNode
#         The calibration node containing parameters and qubit pairs.

#     Returns:
#     --------
#     Tuple[xr.Dataset, Dict[str, FitResults]]
#         Dataset with fit results and dictionary of fit results for each qubit pair.
#     """
#     # For RB analysis, no fitting routine is currently implemented.
#     # Pass the dataset through unchanged, or implement RB-specific fitting here if needed.
#     ds_fit = ds

#     # Extract the relevant fitted parameters
#     ds_fit, fit_results = _extract_relevant_parameters(ds_fit, node)

#     return ds_fit, fit_results


# def _extract_relevant_parameters(
#     ds_fit: xr.Dataset, node: QualibrationNode
# ) -> Tuple[xr.Dataset, Dict[str, FitResults]]:
#     """
#     Extract relevant fit parameters and create FitResults for each qubit pair.

#     Parameters:
#     -----------
#     ds_fit : xr.Dataset
#         Dataset containing the fit results from fit_routine.
#     node : QualibrationNode
#         The calibration node containing parameters and qubit pairs.

#     Returns:
#     --------
#     Tuple[xr.Dataset, Dict[str, FitResults]]
#         Dataset with additional metadata and dictionary of FitResults for each qubit pair.
#     """
#     qubit_pairs = node.namespace["qubit_pairs"]

#     # Add metadata attributes to the dataset
#     if "optimal_amplitude" in ds_fit.data_vars:
#         ds_fit.optimal_amplitude.attrs = {"long_name": "optimal CZ amplitude", "units": "a.u."}
#     if "phase_diff" in ds_fit.data_vars:
#         ds_fit.phase_diff.attrs = {"long_name": "phase difference", "units": "2π"}
#     if "fitted_curve" in ds_fit.data_vars:
#         ds_fit.fitted_curve.attrs = {"long_name": "fitted tanh curve", "units": "2π"}

#     # Create FitResults for each qubit pair
#     fit_results = {}
#     for qp in qubit_pairs:
#         qp_name = qp.name
#         qp_data = ds_fit.sel(qubit_pair=qp_name)

#         fit_results[qp_name] = FitResults(
#             optimal_amplitude=float(qp_data.optimal_amplitude.values),
#             success=bool(qp_data.success.values),
#         )

#     return ds_fit, fit_results
