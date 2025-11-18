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
    """Stores the relevant qubit_pair spectroscopy experiment fit parameters for a single qubit_pair"""

    iw_angle: float
    ge_threshold: float
    I_threshold: float
    readout_fidelity: float
    confusion_matrix: list
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubit_pairs from the fit xarray Dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the fitted results for all qubit_pairs.
    log_callable : callable, optional
        Callable for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit_pair = f"Results for qubit_pair {q}: "
        s = f"IW angle: {fit_results[q]['iw_angle'] * 180 / np.pi:.1f} deg | "
        s += f"ge_threshold: {fit_results[q]['ge_threshold'] * 1e3:.1f} mV | "
        s += f"I_threshold: {fit_results[q]['I_threshold'] * 1e3:.1f} mV | "
        s += f"readout fidelity: {fit_results[q]['readout_fidelity']:.1f} % \n "
        if fit_results[q]["success"]:
            s_qubit_pair += " SUCCESS!\n"
        else:
            s_qubit_pair += " FAIL!\n"
        log_callable(s_qubit_pair + s)


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
    # ds = convert_IQ_to_V(ds, node.namespace["qubit_pairs"], IQ_list=["Ig", "Qg", "Ie", "Qe"])
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the quantum dot readout using PCA and Barthel model for each qubit_pair in the dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data with Ig, Qg, Ie, Qe.
    node : QualibrationNode
        Node containing qubit_pair information.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    dict[str, FitParameters]
        Dictionary of fit parameters for each qubit_pair.
    """
    ds_fit = ds

    # Lists to store results for all qubit_pairs
    angles = []
    ge_thresholds = []
    norm_ge_thresholds = []
    I_thresholds = []
    gg_list, ge_list, eg_list, ee_list = [], [], [], []

    # Lists to store rotated IQ data for plotting
    Ig_rot_list, Qg_rot_list, Ie_rot_list, Qe_rot_list = [], [], [], []

    # Lists to store PCA projected data for plotting
    y_pca_list = []  # PCA projected data (1D)

    # Lists to store pre-computed density curves for plotting
    grid_values_list = []  # x-axis values for density plots (normalized)
    total_density_list = []  # total density curve
    S_density_list = []  # S component density
    T_no_density_list = []  # T (no decay) component density
    T_dec_density_list = []  # T (decay) component density
    weights_list = []  # [w_S, w_T_no, w_T_dec]

    # Lists to store fidelity and visibility curves for plotting
    fidelity_vrf_list = []  # threshold values for fidelity curve
    fidelity_curve_list = []  # fidelity values
    fidelity_opt_vrf_list = []  # optimal threshold for fidelity
    fidelity_opt_list = []  # optimal fidelity value
    visibility_vrf_list = []  # threshold values for visibility curve
    visibility_curve_list = []  # visibility values
    visibility_opt_vrf_list = []  # optimal threshold for visibility
    visibility_opt_list = []  # optimal visibility value

    for q in node.namespace["qubit_pairs"]:
        # Extract I and Q data for ground (S) and excited (T) states
        Ig_q = ds_fit.Ig.sel(qubit_pair=q.name).values
        Qg_q = ds_fit.Qg.sel(qubit_pair=q.name).values
        Ie_q = ds_fit.Ie.sel(qubit_pair=q.name).values
        Qe_q = ds_fit.Qe.sel(qubit_pair=q.name).values

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
        proj_dir = proj.pc1 * proj.sign
        angle = jnp.arctan2(proj_dir[1], proj_dir[0])
        angles.append(float(angle))

        # Rotation matrix for rotating by -angle to align with I-axis
        rotation_matrix = jnp.array([
            [jnp.cos(angle), jnp.sin(angle)],
            [-jnp.sin(angle), jnp.cos(angle)]
        ])

        # Apply rotation to data
        Xg_rotated = X_ground @ rotation_matrix.T
        Xe_rotated = X_excited @ rotation_matrix.T

        proj_rotated_mean = proj.mean @ rotation_matrix.T

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
        norm_ge_thresholds.append(v_rf_norm)

        I_threshold = float(v_rf_phys + proj_rotated_mean[0])
        I_thresholds.append(I_threshold)

        # Store fidelity and visibility curves for plotting
        fidelity_vrf_list.append(np.array(fidelity_res["vrf"]))
        fidelity_curve_list.append(np.array(fidelity_res["fidelity"]))
        fidelity_opt_vrf_list.append(float(fidelity_res["vrf_opt"]))
        fidelity_opt_list.append(float(fidelity_res["fidelity_opt"]))

        visibility_vrf_list.append(np.array(visibility_res["vrf_grid"]))
        visibility_curve_list.append(np.array(visibility_res["visibility_curve"]))
        visibility_opt_vrf_list.append(float(visibility_res["vrf"]))
        visibility_opt_list.append(float(visibility_res["visibility"]))

        Ig_rot = Xg_rotated[..., 0]
        Qg_rot = Xg_rotated[..., 1]
        Ie_rot = Xe_rotated[..., 0]
        Qe_rot = Xe_rotated[..., 1]

        # Store rotated data for plotting
        Ig_rot_list.append(Ig_rot)
        Qg_rot_list.append(Qg_rot)
        Ie_rot_list.append(Ie_rot)
        Qe_rot_list.append(Qe_rot)

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

        # Pre-compute density curves for plotting using analytic functions
        from .readout_barthel.analytic import _norm_pdf, triplet_pdf_analytic, decay_inflight_integral

        # Normalize the PCA projected data for density calculation
        y_norm = normalizer.transform(y)

        # Store normalized PCA projected data (for plotting)
        y_pca_list.append(y_norm)
        rng_norm = np.ptp(y_norm) or 1.0
        xs_norm = np.linspace(y_norm.min() - 0.1 * rng_norm, y_norm.max() + 0.1 * rng_norm, 800)

        # Get posterior means for parameters (these are in normalized space)
        mu_S = float(np.asarray(samples["mu_S"]).mean())
        mu_T = float(np.asarray(samples["mu_T"]).mean())
        sigma = float(np.asarray(samples["sigma"]).mean())
        pT_m = float(np.asarray(samples["pT"]).mean())
        T1_m = float(np.asarray(samples["T1"]).mean())
        tauM_m = 1.0  # Fixed measurement time

        # Convert to JAX arrays for analytic functions
        xs_jax = jnp.array(xs_norm)

        # Compute density components analytically
        # Singlet component: (1 - pT) * N(y; mu_S, sigma)
        S_comp = (1 - pT_m) * _norm_pdf(xs_jax, mu_S, sigma)

        # Triplet component breakdown:
        p_no = jnp.exp(-tauM_m / T1_m) if T1_m > 0 else 0.0
        # T (no decay): pT * p_no * N(y; mu_T, sigma)
        T_no_comp = pT_m * p_no * _norm_pdf(xs_jax, mu_T, sigma)
        # T (decay): pT * (1/T1) * integral
        T_dec_comp = pT_m * (1.0 / T1_m) * decay_inflight_integral(xs_jax, mu_S, mu_T, sigma, T1_m, tauM_m)

        # Total density: singlet + triplet (no decay) + triplet (decay)
        total = S_comp + T_no_comp + T_dec_comp

        # Weights
        w_S = 1 - pT_m
        w_T_no = pT_m * float(p_no)
        w_T_dec = pT_m * (1 - float(p_no))

        # Store pre-computed curves and weights (convert JAX arrays to numpy)
        grid_values_list.append(np.array(xs_norm))
        total_density_list.append(np.array(total))
        S_density_list.append(np.array(S_comp))
        T_no_density_list.append(np.array(T_no_comp))
        T_dec_density_list.append(np.array(T_dec_comp))
        weights_list.append([w_S, w_T_no, w_T_dec])

    # Assign all results to dataset
    ds_fit = ds_fit.assign({"iw_angle": xr.DataArray(angles, coords=dict(qubit_pair=ds_fit.qubit_pair.data))})
    ds_fit = ds_fit.assign({"ge_threshold": xr.DataArray(ge_thresholds, coords=dict(qubit_pair=ds_fit.qubit_pair.data))})
    ds_fit = ds_fit.assign({"norm_ge_threshold": xr.DataArray(norm_ge_thresholds, coords=dict(qubit_pair=ds_fit.qubit_pair.data))})
    ds_fit = ds_fit.assign({'I_threshold': xr.DataArray(I_thresholds, coords=dict(qubit_pair=ds_fit.qubit_pair.data))})
    ds_fit = ds_fit.assign({"gg": xr.DataArray(gg_list, coords=dict(qubit_pair=ds_fit.qubit_pair.data))})
    ds_fit = ds_fit.assign({"ge": xr.DataArray(ge_list, coords=dict(qubit_pair=ds_fit.qubit_pair.data))})
    ds_fit = ds_fit.assign({"eg": xr.DataArray(eg_list, coords=dict(qubit_pair=ds_fit.qubit_pair.data))})
    ds_fit = ds_fit.assign({"ee": xr.DataArray(ee_list, coords=dict(qubit_pair=ds_fit.qubit_pair.data))})
    ds_fit = ds_fit.assign(
        {"readout_fidelity": xr.DataArray(100 * (ds_fit.gg + ds_fit.ee) / 2, coords=dict(qubit_pair=ds_fit.qubit_pair.data))}
    )

    # Assign rotated IQ data for plotting
    ds_fit = ds_fit.assign({"Ig_rot": xr.DataArray(Ig_rot_list, dims=["qubit_pair", "n_runs"])})
    ds_fit = ds_fit.assign({"Qg_rot": xr.DataArray(Qg_rot_list, dims=["qubit_pair", "n_runs"])})
    ds_fit = ds_fit.assign({"Ie_rot": xr.DataArray(Ie_rot_list, dims=["qubit_pair", "n_runs"])})
    ds_fit = ds_fit.assign({"Qe_rot": xr.DataArray(Qe_rot_list, dims=["qubit_pair", "n_runs"])})

    # Assign PCA projected data for plotting
    # Note: y_pca combines both ground and excited states, so it has dimension n_samples (2 * n_runs)
    ds_fit = ds_fit.assign({"y_pca": xr.DataArray(y_pca_list, dims=["qubit_pair", "n_samples"])})

    # Assign pre-computed density curves for plotting (all as DataArrays)
    # Note: All lists have same grid size (800 points), so we use a grid_pts dimension
    ds_fit = ds_fit.assign({"density_grid": xr.DataArray(grid_values_list, dims=["qubit_pair", "grid_pts"])})
    ds_fit = ds_fit.assign({"density_total": xr.DataArray(total_density_list, dims=["qubit_pair", "grid_pts"])})
    ds_fit = ds_fit.assign({"density_S": xr.DataArray(S_density_list, dims=["qubit_pair", "grid_pts"])})
    ds_fit = ds_fit.assign({"density_T_no": xr.DataArray(T_no_density_list, dims=["qubit_pair", "grid_pts"])})
    ds_fit = ds_fit.assign({"density_T_dec": xr.DataArray(T_dec_density_list, dims=["qubit_pair", "grid_pts"])})
    ds_fit = ds_fit.assign({"weights": xr.DataArray(weights_list, dims=["qubit_pair", "weight_component"],
                                                      coords={"weight_component": ["w_S", "w_T_no", "w_T_dec"]})})

    # Assign fidelity and visibility curves for plotting (all as DataArrays)
    ds_fit = ds_fit.assign({"fidelity_vrf": xr.DataArray(fidelity_vrf_list, dims=["qubit_pair", "fidelity_pts"])})
    ds_fit = ds_fit.assign({"fidelity_curve": xr.DataArray(fidelity_curve_list, dims=["qubit_pair", "fidelity_pts"])})
    ds_fit = ds_fit.assign({"fidelity_opt_vrf": xr.DataArray(fidelity_opt_vrf_list, coords=dict(qubit_pair=ds_fit.qubit_pair.data))})
    ds_fit = ds_fit.assign({"fidelity_opt": xr.DataArray(fidelity_opt_list, coords=dict(qubit_pair=ds_fit.qubit_pair.data))})

    ds_fit = ds_fit.assign({"visibility_vrf": xr.DataArray(visibility_vrf_list, dims=["qubit_pair", "visibility_pts"])})
    ds_fit = ds_fit.assign({"visibility_curve": xr.DataArray(visibility_curve_list, dims=["qubit_pair", "visibility_pts"])})
    ds_fit = ds_fit.assign({"visibility_opt_vrf": xr.DataArray(visibility_opt_vrf_list, coords=dict(qubit_pair=ds_fit.qubit_pair.data))})
    ds_fit = ds_fit.assign({"visibility_opt": xr.DataArray(visibility_opt_list, coords=dict(qubit_pair=ds_fit.qubit_pair.data))})

    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""

    # Assess whether the fit was successful or not
    nan_success = (
        np.isnan(fit.iw_angle)
        | np.isnan(fit.ge_threshold)
        | np.isnan(fit.I_threshold)
        | np.isnan(fit.readout_fidelity)
    )
    success_criteria = ~nan_success
    fit = fit.assign({"success": success_criteria})

    fit_results = {
        q: FitParameters(
            iw_angle=float(fit.sel(qubit_pair=q).iw_angle),
            ge_threshold=float(fit.sel(qubit_pair=q).ge_threshold),
            I_threshold=float(fit.sel(qubit_pair=q).I_threshold),
            readout_fidelity=float(fit.sel(qubit_pair=q).readout_fidelity),
            confusion_matrix=[
                [float(fit.sel(qubit_pair=q).gg), float(fit.sel(qubit_pair=q).ge)],
                [float(fit.sel(qubit_pair=q).eg), float(fit.sel(qubit_pair=q).ee)],
            ],
            success=fit.sel(qubit_pair=q).success.values.__bool__(),
        )
        for q in fit.qubit_pair.values
    }
    return fit, fit_results
