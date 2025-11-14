import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr
import jax.numpy as jnp
from jax import config as jax_config

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V

# Barthel model imports
from .readout_barthel.utils import Barthel1DMetricCurves
from .readout_barthel.calibrate import Barthel1DFromIQ
from .readout_barthel.classify import classify_iq_with_pca_threshold

# Enable 64-bit precision for JAX
jax_config.update("jax_enable_x64", True)


@dataclass
class FitParameters:
    """Stores the relevant quantum_dot_pair spectroscopy experiment fit parameters for a single quantum_dot_pair"""

    iw_angle: float
    ge_threshold: float
    rus_threshold: float
    readout_fidelity: float
    confusion_matrix: list
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all quantum_dot_pairs from the fit xarray Dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the fitted results for all quantum_dot_pairs.
    log_callable : callable, optional
        Callable for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_quantum_dot_pair = f"Results for quantum_dot_pair {q}: "
        s = f"IW angle: {fit_results[q]['iw_angle'] * 180 / np.pi:.1f} deg | "
        s += f"ge_threshold: {fit_results[q]['ge_threshold'] * 1e3:.1f} mV | "
        s += f"rus_threshold: {fit_results[q]['rus_threshold'] * 1e3:.1f} mV | "
        s += f"readout fidelity: {fit_results[q]['readout_fidelity']:.1f} % \n "
        if fit_results[q]["success"]:
            s_quantum_dot_pair += " SUCCESS!\n"
        else:
            s_quantum_dot_pair += " FAIL!\n"
        log_callable(s_quantum_dot_pair + s)


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
    ds = convert_IQ_to_V(ds, node.namespace["quantum_dot_pairs"], IQ_list=["Ig", "Qg", "Ie", "Qe"])
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the quantum dot readout using PCA and Barthel model for each quantum_dot_pair in the dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data with Ig, Qg, Ie, Qe.
    node : QualibrationNode
        Node containing quantum_dot_pair information.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    dict[str, FitParameters]
        Dictionary of fit parameters for each quantum_dot_pair.
    """
    ds_fit = ds

    # Lists to store results for all quantum_dot_pairs
    angles = []
    ge_thresholds = []
    rus_thresholds = []
    gg_list, ge_list, eg_list, ee_list = [], [], [], []

    # Lists to store rotated IQ data for plotting
    Ig_rot_list, Qg_rot_list, Ie_rot_list, Qe_rot_list = [], [], [], []

    for q in node.namespace["quantum_dot_pairs"]:
        # Extract I and Q data for ground (S) and excited (T) states
        Ig_q = ds_fit.Ig.sel(quantum_dot_pair=q.name).values
        Qg_q = ds_fit.Qg.sel(quantum_dot_pair=q.name).values
        Ie_q = ds_fit.Ie.sel(quantum_dot_pair=q.name).values
        Qe_q = ds_fit.Qe.sel(quantum_dot_pair=q.name).values

        # Stack ground and excited state data for combined fitting
        # Shape: (n_samples, 2) where columns are [I, Q]
        X_ground = np.column_stack([Ig_q, Qg_q])
        X_excited = np.column_stack([Ie_q, Qe_q])
        X_combined = np.vstack([X_ground, X_excited])
        X_combined = jnp.array(X_combined)

        # Use excited state (T) as calibration data
        X_calib = jnp.array(X_excited)

        # Fit Barthel model with PCA projection
        # Returns: y (1D projection), proj (projection matrix), normalizer, mcmc, samples, _, calib_res
        y, proj, normalizer, mcmc, samples, _, calib_res = Barthel1DFromIQ.fit(
            X_combined,
            fix_tau_M=1.0,  # Fixed measurement time (can be adjusted based on your setup)
            calib=(X_calib, "T"),  # Excited state is triplet (T)
            prior_strength=0.3,  # Prior strength from calibration
            sigma_scale_default=0.25  # Noise prior scale
        )

        # Extract rotation angle from PCA projection
        # proj is the projection direction, angle is arctan2(proj[1], proj[0])
        angle = np.arctan2(proj[1], proj[0])
        angles.append(angle)

        # Compute fidelity and visibility metrics to get optimal threshold
        fidelity_res = Barthel1DMetricCurves.summarize_metric(
            samples,
            tauM_fixed=1.0,
            use_ppd=True,
            draws=64,
            metric="fidelity",
            return_aligned_curve=True,
        )
        visibility_res = Barthel1DMetricCurves.summarize_metric(
            samples,
            tauM_fixed=1.0,
            use_ppd=True,
            draws=64,
            metric="visibility",
            return_components=True,
        )

        # Get optimal threshold in normalized 1D space
        v_rf_norm = fidelity_res["vrf_opt_aligned"]
        # Convert back to physical space
        v_rf_phys = normalizer.inverse(v_rf_norm)
        ge_thresholds.append(v_rf_phys)

        # Rotate data using PCA projection for rus_threshold calculation and plotting
        C = np.cos(angle)
        S = np.sin(angle)
        Ig_rot = Ig_q * C - Qg_q * S
        Qg_rot = Ig_q * S + Qg_q * C
        Ie_rot = Ie_q * C - Qe_q * S
        Qe_rot = Ie_q * S + Qe_q * C

        # Store rotated data for plotting
        Ig_rot_list.append(Ig_rot)
        Qg_rot_list.append(Qg_rot)
        Ie_rot_list.append(Ie_rot)
        Qe_rot_list.append(Qe_rot)

        # Calculate rus_threshold from histogram of ground state rotated data
        hist = np.histogram(Ig_rot, bins=100)
        rus_thresh = hist[1][1:][np.argmax(hist[0])]
        rus_thresholds.append(rus_thresh)

        # Calculate confusion matrix using Barthel threshold
        # Classify the data using the optimal threshold
        labels_ground, _ = classify_iq_with_pca_threshold(
            jnp.array(X_ground), proj, v_rf_norm, normalizer=normalizer, return_margin=True
        )
        labels_excited, _ = classify_iq_with_pca_threshold(
            jnp.array(X_excited), proj, v_rf_norm, normalizer=normalizer, return_margin=True
        )

        # Convert JAX arrays to numpy for counting
        labels_ground = np.array(labels_ground)
        labels_excited = np.array(labels_excited)

        # Confusion matrix elements
        # For quantum dots: label 0 = S (singlet/ground), label 1 = T (triplet/excited)
        # gg: ground state measured as ground
        # ge: ground state measured as excited
        # eg: excited state measured as ground
        # ee: excited state measured as excited
        gg = np.sum(labels_ground == 0) / len(labels_ground)
        ge = np.sum(labels_ground == 1) / len(labels_ground)
        eg = np.sum(labels_excited == 0) / len(labels_excited)
        ee = np.sum(labels_excited == 1) / len(labels_excited)

        gg_list.append(gg)
        ge_list.append(ge)
        eg_list.append(eg)
        ee_list.append(ee)

    # Assign all results to dataset
    ds_fit = ds_fit.assign({"iw_angle": xr.DataArray(angles, coords=dict(quantum_dot_pair=ds_fit.quantum_dot_pair.data))})
    ds_fit = ds_fit.assign({"ge_threshold": xr.DataArray(ge_thresholds, coords=dict(quantum_dot_pair=ds_fit.quantum_dot_pair.data))})
    ds_fit = ds_fit.assign({"rus_threshold": xr.DataArray(rus_thresholds, coords=dict(quantum_dot_pair=ds_fit.quantum_dot_pair.data))})
    ds_fit = ds_fit.assign({"gg": xr.DataArray(gg_list, coords=dict(quantum_dot_pair=ds_fit.quantum_dot_pair.data))})
    ds_fit = ds_fit.assign({"ge": xr.DataArray(ge_list, coords=dict(quantum_dot_pair=ds_fit.quantum_dot_pair.data))})
    ds_fit = ds_fit.assign({"eg": xr.DataArray(eg_list, coords=dict(quantum_dot_pair=ds_fit.quantum_dot_pair.data))})
    ds_fit = ds_fit.assign({"ee": xr.DataArray(ee_list, coords=dict(quantum_dot_pair=ds_fit.quantum_dot_pair.data))})
    ds_fit = ds_fit.assign(
        {"readout_fidelity": xr.DataArray(100 * (ds_fit.gg + ds_fit.ee) / 2, coords=dict(quantum_dot_pair=ds_fit.quantum_dot_pair.data))}
    )

    # Assign rotated IQ data for plotting
    ds_fit = ds_fit.assign({"Ig_rot": xr.DataArray(Ig_rot_list, dims=["quantum_dot_pair", "n_runs"])})
    ds_fit = ds_fit.assign({"Qg_rot": xr.DataArray(Qg_rot_list, dims=["quantum_dot_pair", "n_runs"])})
    ds_fit = ds_fit.assign({"Ie_rot": xr.DataArray(Ie_rot_list, dims=["quantum_dot_pair", "n_runs"])})
    ds_fit = ds_fit.assign({"Qe_rot": xr.DataArray(Qe_rot_list, dims=["quantum_dot_pair", "n_runs"])})

    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


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
            iw_angle=float(fit.sel(quantum_dot_pair=q).iw_angle),
            ge_threshold=float(fit.sel(quantum_dot_pair=q).ge_threshold),
            rus_threshold=float(fit.sel(quantum_dot_pair=q).rus_threshold),
            readout_fidelity=float(fit.sel(quantum_dot_pair=q).readout_fidelity),
            confusion_matrix=[
                [float(fit.sel(quantum_dot_pair=q).gg), float(fit.sel(quantum_dot_pair=q).ge)],
                [float(fit.sel(quantum_dot_pair=q).eg), float(fit.sel(quantum_dot_pair=q).ee)],
            ],
            success=fit.sel(quantum_dot_pair=q).success.values.__bool__(),
        )
        for q in fit.quantum_dot_pair.values
    }
    return fit, fit_results
