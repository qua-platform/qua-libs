import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.qua_datasets import convert_IQ_to_V
from scipy.optimize import minimize


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    iw_angle: float
    ge_threshold: float
    rus_threshold: float
    readout_fidelity: float
    confusion_matrix: list
    success: bool


def log_fitted_results(fit_results: Dict, logger=None):
    """
    Logs the node-specific fitted results for all qubits from the fit xarray Dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the fitted results for all qubits.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s = f"IW angle: {fit_results[q]['iw_angle'] * 180 / np.pi:.1f} deg | "
        s += f"ge_threshold: {fit_results[q]['ge_threshold'] * 1e3:.1f} mV | "
        s += f"rus_threshold: {fit_results[q]['rus_threshold'] * 1e3:.1f} mV | "
        s += f"readout fidelity: {fit_results[q]['readout_fidelity']:.1f} % \n "
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        logger.info(s_qubit + s)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    # Fix the structure of ds to avoid tuples
    def extract_value(element):
        if isinstance(element, tuple):
            return element[0]
        return element

    ds = xr.apply_ufunc(
        extract_value,
        ds,
        vectorize=True,  # This ensures the function is applied element-wise
        dask="parallelized",  # This allows for parallel processing
        output_dtypes=[float],  # Specify the output data type
    )
    ds = convert_IQ_to_V(ds, node.namespace["qubits"], IQ_list=["Ig", "Qg", "Ie", "Qe"])
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
    # Condition to have the Q equal for both states:
    angle = np.arctan2(
        ds_fit.Qe.mean(dim="n_runs") - ds_fit.Qg.mean(dim="n_runs"),
        ds_fit.Ig.mean(dim="n_runs") - ds_fit.Ie.mean(dim="n_runs"),
    )
    ds_fit = ds_fit.assign({"iw_angle": xr.DataArray(angle, coords=dict(qubit=ds_fit.qubit.data))})

    C = np.cos(angle)
    S = np.sin(angle)
    # Condition for having e > Ig
    if np.mean((ds_fit.Ig - ds_fit.Ie) * C - (ds_fit.Qg - ds_fit.Qe) * S) > 0:
        angle += np.pi
        C = np.cos(angle)
        S = np.sin(angle)

    ds_fit = ds_fit.assign({"Ig_rot": ds_fit.Ig * C - ds_fit.Qg * S})
    ds_fit = ds_fit.assign({"Qg_rot": ds_fit.Ig * S + ds_fit.Qg * C})
    ds_fit = ds_fit.assign({"Ie_rot": ds_fit.Ie * C - ds_fit.Qe * S})
    ds_fit = ds_fit.assign({"Qe_rot": ds_fit.Ie * S + ds_fit.Qe * C})

    # Get the blobs histogram along the rotated axis
    hist = np.histogram(ds_fit.Ig_rot, bins=100)
    # Get the discriminating threshold along the rotated axis
    rus_threshold = [
        hist[1][1:][np.argmax(np.histogram(ds_fit.Ig_rot.sel(qubit=q.name), bins=100)[0])]
        for q in node.namespace["qubits"]
    ]
    ds_fit = ds_fit.assign({"rus_threshold": xr.DataArray(rus_threshold, coords=dict(qubit=ds_fit.qubit.data))})

    threshold = []
    gg, ge, eg, ee = [], [], [], []
    for q in node.namespace["qubits"]:
        fit = minimize(
            _false_detections,
            0.5 * (np.mean(ds_fit.Ig_rot.sel(qubit=q.name)) + np.mean(ds_fit.Ie_rot.sel(qubit=q.name))),
            (ds_fit.Ig_rot.sel(qubit=q.name), ds_fit.Ie_rot.sel(qubit=q.name)),
            method="Nelder-Mead",
        )
        threshold.append(fit.x[0])
        gg.append(np.sum(ds_fit.Ig_rot.sel(qubit=q.name) < fit.x[0]) / len(ds_fit.Ig_rot.sel(qubit=q.name)))
        ge.append(np.sum(ds_fit.Ig_rot.sel(qubit=q.name) > fit.x[0]) / len(ds_fit.Ig_rot.sel(qubit=q.name)))
        eg.append(np.sum(ds_fit.Ie_rot.sel(qubit=q.name) < fit.x[0]) / len(ds_fit.Ie_rot.sel(qubit=q.name)))
        ee.append(np.sum(ds_fit.Ie_rot.sel(qubit=q.name) > fit.x[0]) / len(ds_fit.Ie_rot.sel(qubit=q.name)))
    ds_fit = ds_fit.assign({"ge_threshold": xr.DataArray(threshold, coords=dict(qubit=ds_fit.qubit.data))})
    ds_fit = ds_fit.assign({"gg": xr.DataArray(gg, coords=dict(qubit=ds_fit.qubit.data))})
    ds_fit = ds_fit.assign({"ge": xr.DataArray(ge, coords=dict(qubit=ds_fit.qubit.data))})
    ds_fit = ds_fit.assign({"eg": xr.DataArray(eg, coords=dict(qubit=ds_fit.qubit.data))})
    ds_fit = ds_fit.assign({"ee": xr.DataArray(ee, coords=dict(qubit=ds_fit.qubit.data))})
    ds_fit = ds_fit.assign(
        {"readout_fidelity": xr.DataArray(100 * (ds_fit.gg + ds_fit.ee) / 2, coords=dict(qubit=ds_fit.qubit.data))}
    )

    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _false_detections(threshold, Ig, Ie):
    if np.mean(Ig) < np.mean(Ie):
        false_detections_var = np.sum(Ig > threshold) + np.sum(Ie < threshold)
    else:
        false_detections_var = np.sum(Ig < threshold) + np.sum(Ie > threshold)
    return false_detections_var


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""

    # Assess whether the fit was successful or not
    nan_success = (
        np.isnan(fit.iw_angle)
        | np.isnan(fit.ge_threshold)
        | np.isnan(fit.rus_threshold)
        | np.isnan(fit.readout_fidelity)
    )
    success_criteria = ~nan_success
    fit = fit.assign({"success": success_criteria})

    fit_results = {
        q: FitParameters(
            iw_angle=float(fit.sel(qubit=q).iw_angle),
            ge_threshold=float(fit.sel(qubit=q).ge_threshold),
            rus_threshold=float(fit.sel(qubit=q).rus_threshold),
            readout_fidelity=float(fit.sel(qubit=q).readout_fidelity),
            confusion_matrix=[
                [float(fit.sel(qubit=q).gg), float(fit.sel(qubit=q).ge)],
                [float(fit.sel(qubit=q).eg), float(fit.sel(qubit=q).ee)],
            ],
            success=fit.sel(qubit=q).success.values.__bool__(),
        )
        for q in fit.qubit.values
    }
    return fit, fit_results
