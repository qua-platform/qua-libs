import logging
import re
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from scipy.optimize import curve_fit, minimize


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit xarray Dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the fitted results for all qubits.
    log_callable : callable, optional
        Callable for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
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
        log_callable(s_qubit + s)


def find_biggest_gaussian(da):
    # Define Gaussian function
    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    # Get histogram data
    hist, bin_edges = np.histogram(da, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit multiple Gaussians
    initial_guess = [(hist.max(), bin_centers[hist.argmax()], (bin_centers[-1] - bin_centers[0]) / 4)]
    popt, _ = curve_fit(gaussian, bin_centers, hist, p0=initial_guess)
    # Find the biggest Gaussian
    biggest_gaussian = {"amp": popt[0], "mu": popt[1], "sigma": popt[2]}

    return biggest_gaussian["mu"]


def fit_gaussian_centers(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """
    Fit the centers of the Gaussian distributions for each qubit in the dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node : QualibrationNode
        Node containing parameters and results.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fitted Gaussian centers.
    """
    fit_results = {}
    for q in ds.qubit.values:
        ds_q = ds.sel(qubit=q)
        I_g_cent = find_biggest_gaussian(ds_q.Ig)
        Q_g_cent = find_biggest_gaussian(ds_q.Qg)
        I_e_cent = find_biggest_gaussian(ds_q.Ie)
        Q_e_cent = find_biggest_gaussian(ds_q.Qe)
        I_f_cent = find_biggest_gaussian(ds_q.If)
        Q_f_cent = find_biggest_gaussian(ds_q.Qf)

        fit_results[q] = {
            "I_g_center": I_g_cent,
            "Q_g_center": Q_g_cent,
            "I_e_center": I_e_cent,
            "Q_e_center": Q_e_cent,
            "I_f_center": I_f_cent,
            "Q_f_center": Q_f_cent,
        }

    # Convert fit results to xarray Dataset
    # Reorganize fit_results to have each parameter as a variable with dimension ["qubit"]
    qubit_names = list(fit_results.keys())
    param_names = list(next(iter(fit_results.values())).keys())
    data_vars = {param: (["qubit"], [fit_results[q][param] for q in qubit_names]) for param in param_names}
    fit_ds = xr.Dataset(data_vars, coords={"qubit": qubit_names})

    fit_ds = xr.merge([ds, fit_ds])

    return fit_ds


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
    ds_fit = fit_gaussian_centers(ds, node)

    # node.results["results"][q.name]["center_matrix"] = np.array(
    #     [[I_g_cent, Q_g_cent], [I_e_cent, Q_e_cent], [I_f_cent, Q_f_cent]]
    # )
    # # Derive the confusion matrix
    # confusion = np.zeros((3, 3))
    # for p, prep_state in enumerate(["g", "e", "f"]):
    #     dist_g = np.sqrt(
    #         (I_g_cent - ds[f"I_{prep_state}"].sel(qubit=q.name)) ** 2
    #         + (Q_g_cent - ds[f"Q_{prep_state}"].sel(qubit=q.name)) ** 2
    #     )
    #     dist_e = np.sqrt(
    #         (I_e_cent - ds[f"I_{prep_state}"].sel(qubit=q.name)) ** 2
    #         + (Q_e_cent - ds[f"Q_{prep_state}"].sel(qubit=q.name)) ** 2
    #     )
    #     dist_f = np.sqrt(
    #         (I_f_cent - ds[f"I_{prep_state}"].sel(qubit=q.name)) ** 2
    #         + (Q_f_cent - ds[f"Q_{prep_state}"].sel(qubit=q.name)) ** 2
    #     )
    #     dist = np.stack([dist_g, dist_e, dist_f], axis=0)
    #     counts = np.argmin(dist, axis=0)
    #     confusion[p][0] = np.sum(counts == 0) / len(counts)
    #     confusion[p][1] = np.sum(counts == 1) / len(counts)
    #     confusion[p][2] = np.sum(counts == 2) / len(counts)
    #     node.results["results"][q.name]["confusion_matrix"] = confusion
    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    # return fit_data, fit_results
    return fit_data, fit_results


def _false_detections(threshold, Ig, Ie):
    if np.mean(Ig) < np.mean(Ie):
        false_detections_var = np.sum(Ig > threshold) + np.sum(Ie < threshold)
    else:
        false_detections_var = np.sum(Ig < threshold) + np.sum(Ie > threshold)
    return false_detections_var


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""

    fit_results = {
        q: FitParameters(
            success=True,
        )
        for q in fit.qubit.values
    }

    fit = fit.groupby("qubit").apply(center_matrix)

    return fit, fit_results


def center_matrix(da: xr.DataArray) -> xr.DataArray:
    center = np.array(
        [
            [da.I_g_center.item(), da.Q_g_center.item()],
            [da.I_e_center.item(), da.Q_e_center.item()],
            [da.I_f_center.item(), da.Q_f_center.item()],
        ]
    )
    return da.assign(center_matrix=(["I", "Q"], center))
