import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V


@dataclass
class FitParameters:
    """Stores the relevant readout frequency optimization experiment fit parameters for a single qubit"""

    optimal_detuning: float
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
        s_freq = f"\tOptimal frequency shift: {1e-6 * fit_results[q]['optimal_detuning']:.3f} MHz | "
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_freq)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    # Convert IQ data into volts
    ds = convert_IQ_to_V(ds, node.namespace["qubits"], IQ_list=["Ig", "Qg", "Ie", "Qe", "If", "Qf"])
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

    ds = ds.groupby("qubit").apply(fit_routine, node=node)

    ds_fit, fit_results = _extract_relevant_fit_parameters(ds, node)

    return ds_fit, fit_results


def _extract_relevant_fit_parameters(ds_fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""

    fit_results = {}
    for q in node.parameters.qubits:
        if q not in ds_fit.qubit.values:
            logging.warning(f"Qubit {q} not found in the fit results.")
            continue

        fit_results[q] = FitParameters(
            optimal_detuning=ds_fit.optimal_detuning.sel(qubit=q).item(),
            success=ds_fit.success.sel(qubit=q).item(),
        )

    return ds_fit, fit_results


def fit_routine(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """
    Fit routine for a single qubit.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data for a single qubit.
    node : QualibrationNode
        The node containing the parameters and context for the fit.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results for the qubit.
    """

    ds = ds.assign(
        {
            "Dge": np.sqrt((ds.Ig - ds.Ie) ** 2 + (ds.Qg - ds.Qe) ** 2),
            "Def": np.sqrt((ds.Ie - ds.If) ** 2 + (ds.Qe - ds.Qf) ** 2),
            "Dgf": np.sqrt((ds.Ig - ds.If) ** 2 + (ds.Qg - ds.Qf) ** 2),
        }
    )

    ds["Distance"] = ds[["Dge", "Def", "Dgf"]].to_array().min("variable")

    detuning = ds.Distance.rolling({"frequency": 3}).mean("frequency").idxmax("frequency")

    ds = ds.assign(
        {
            "optimal_detuning": detuning,
        }
    )

    ds = ds.assign(
        {
            "success": True,
        }
    )

    return ds
