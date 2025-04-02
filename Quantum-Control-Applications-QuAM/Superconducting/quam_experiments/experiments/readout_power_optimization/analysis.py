import logging
from dataclasses import dataclass, replace
from typing import Optional, Tuple, Dict
import numpy as np
import xarray as xr
from sklearn.mixture import GaussianMixture

from qualibrate import QualibrationNode
from qualibration_libs.qua_datasets import convert_IQ_to_V
from quam_experiments.experiments.iq_blobs import fit_raw_data as fit_iq_blobs
from quam_experiments.experiments.iq_blobs.analysis import FitParameters as FitParametersIQblobs


@dataclass
class FitParameters(FitParametersIQblobs):
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    optimal_amplitude: float = 0


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
    # for q in fit_results.keys():
    #     s_qubit = f"Results for qubit {q}: "
    #     s_freq = f"\tQubit frequency: {1e-9 * fit_results[q]['frequency']:.3f} GHz | "
    #     s_fwhm = f"FWHM: {1e-3 * fit_results[q]['fwhm']:.1f} kHz | "
    #     s_angle = (
    #         f"The integration weight angle: {fit_results[q]['iw_angle']:.3f} rad\n "
    #     )
    #     s_saturation = f"To get the desired FWHM, the saturation amplitude is updated to: {1e3 * fit_results[q]['saturation_amp']:.1f} mV | "
    #     s_x180 = f"To get the desired x180 gate, the x180 amplitude is updated to: {1e3 * fit_results[q]['x180_amp']:.1f} mV\n "
    #     if fit_results[q]["success"]:
    #         s_qubit += " SUCCESS!\n"
    #     else:
    #         s_qubit += " FAIL!\n"
    #     logger.info(
    #         s_qubit + s_freq + s_fwhm + s_freq + s_angle + s_saturation + s_x180
    #     )
    pass


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    # Skip if the data has already been processed
    if ~np.all([var in ds.data_vars for var in ["Ig", "Qg", "Ie", "Qe"]]):
        return ds
    ds = convert_IQ_to_V(ds, node.namespace["qubits"], IQ_list=["Ig", "Qg", "Ie", "Qe"])
    # Add the absolute readout power to the dataset
    readout_amplitudes = np.array(
        [ds.amp_prefactor * q.resonator.operations["readout"].amplitude for q in node.namespace["qubits"]]
    )
    ds = ds.assign_coords(readout_amplitude=(["qubit", "amp_prefactor"], readout_amplitudes))
    ds.readout_amplitude.attrs = {"long_name": "readout amplitude", "units": "V"}
    # Rearrange the data to combine I_g and I_e into I, and Q_g and Q_e into Q
    ds_rearranged = xr.Dataset()
    # Combine I_g and I_e into I
    ds_rearranged["I"] = xr.concat([ds.Ig, ds.Ie], dim="state")
    ds_rearranged["I"] = ds_rearranged["I"].assign_coords(state=[0, 1])
    # Combine Q_g and Q_e into Q
    ds_rearranged["Q"] = xr.concat([ds.Qg, ds.Qe], dim="state")
    ds_rearranged["Q"] = ds_rearranged["Q"].assign_coords(state=[0, 1])
    # Copy other coordinates and data variables
    for var in ds.coords:
        if var not in ds_rearranged.coords:
            ds_rearranged[var] = ds[var]

    for var in ds.data_vars:
        if var not in ["Ig", "Ie", "Qg", "Qe"]:
            ds_rearranged[var] = ds[var]

    # Replace the original dataset with the rearranged one
    ds = ds_rearranged
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, xr.Dataset, dict[str, FitParameters]]:
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

    def apply_fit_gmm(I, Q):
        I_mean = np.mean(I, axis=1)
        Q_mean = np.mean(Q, axis=1)
        means_init = [[I_mean[0], Q_mean[0]], [I_mean[1], Q_mean[1]]]
        precisions_init = [1 / ((np.mean(np.var(I, axis=1)) + np.mean(np.var(Q, axis=1))) / 2)] * 2
        clf = GaussianMixture(
            n_components=2,
            covariance_type="spherical",
            means_init=means_init,
            precisions_init=precisions_init,
            tol=1e-5,
            reg_covar=1e-12,
        )
        X = np.array([np.array(I).flatten(), np.array(Q).flatten()]).T
        clf.fit(X)
        meas_fidelity = (
            np.sum(clf.predict(np.array([I[0], Q[0]]).T) == 0) / len(I[0])
            + np.sum(clf.predict(np.array([I[1], Q[1]]).T) == 1) / len(I[1])
        ) / 2
        loglikelihood = clf.score_samples(X)
        max_ll = np.max(loglikelihood)
        outliers = np.sum(loglikelihood > np.log(0.01) + max_ll) / len(X)
        return np.array([meas_fidelity, outliers])

    fit_data = xr.apply_ufunc(
        apply_fit_gmm,
        ds_fit.I,
        ds_fit.Q,
        input_core_dims=[["state", "n_runs"], ["state", "n_runs"]],
        output_core_dims=[["fit_vals"]],
        vectorize=True,
    )
    fit_data = fit_data.assign_coords(fit_vals=["meas_fidelity", "outliers"])
    ds_fit = xr.merge([ds, fit_data.rename("fit_data")])

    # Extract the relevant fitted parameters
    fit_data, fit_results, ds_iq_blobs = _extract_relevant_fit_parameters(ds_fit, node)

    return ds_fit, ds_iq_blobs, fit_results


def _extract_relevant_fit_parameters(ds_fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""

    ds_fit["valid_amps"] = ds_fit.readout_amplitude.where(
        ds_fit.fit_data.sel(fit_vals="outliers") >= node.parameters.outliers_threshold, drop=True
    )
    ds_fit["valid_fidelity"] = ds_fit.fit_data.sel(
        amp_prefactor=ds_fit["valid_amps"].amp_prefactor, fit_vals="meas_fidelity"
    )
    opt_amp = ds_fit["valid_fidelity"].readout_amplitude[:, ds_fit["valid_fidelity"].argmax(dim="amp_prefactor")]
    ds_fit["optimal_amp"] = opt_amp
    ds_fit["best_fidelity"] = ds_fit["valid_fidelity"].sel(amp_prefactor=opt_amp.amp_prefactor)

    best_amp_data = ds_fit.sel(amp_prefactor=opt_amp.amp_prefactor)
    # Extract I and Q data for ground and excited states
    Ig = best_amp_data.I.sel(state=0)
    Qg = best_amp_data.Q.sel(state=0)
    Ie = best_amp_data.I.sel(state=1)
    Qe = best_amp_data.Q.sel(state=1)
    ds_temp = xr.Dataset(
        {"Ig": Ig.drop("state"), "Ie": Ie.drop("state"), "Qg": Qg.drop("state"), "Qe": Qe.drop("state")}
    )
    ds_iq_blobs, _fit_results = fit_iq_blobs(ds_temp, node)

    fit_results = {}
    for q in ds_fit.qubit.values:
        # Create a dictionary of the existing attributes
        params_dict = _fit_results[q].__dict__
        # Add the new field to the dictionary
        params_dict["optimal_amplitude"] = float(ds_fit["optimal_amp"].sel(qubit=q))
        # Instantiate FitParameters using the updated dictionary
        fit_results[q] = FitParameters(**params_dict)
    return ds_fit, fit_results, ds_iq_blobs
