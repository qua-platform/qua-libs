import logging
from dataclasses import dataclass
from typing import Tuple, Dict, List
import numpy as np
import xarray as xr
import jax.numpy as jnp
from jax import config as jax_config

from qualibrate import QualibrationNode

from calibration_utils.iq_blobs.readout_barthel.utils import Barthel1DMetricCurves
from calibration_utils.iq_blobs.readout_barthel.calibrate import Barthel1DFromIQ
from calibration_utils.iq_blobs.readout_barthel.classify import classify_iq_with_pca_threshold

jax_config.update("jax_enable_x64", True)


@dataclass
class FitParameters:
    """Per-qubit fit parameters for a 1D sweep of IQ-blob readout analysis."""

    sweep_name: str
    sweep_values: List[float]

    optimal_sweep_value: float
    optimal_sweep_index: int

    optimal_sweep_value_fidelity: float
    optimal_sweep_index_fidelity: int
    optimal_sweep_value_visibility: float
    optimal_sweep_index_visibility: int

    iw_angle: float
    I_threshold: float
    ge_threshold: float
    readout_threshold: float
    readout_projector: Dict[str, float]
    readout_fidelity: float
    visibility: float
    confusion_matrix: List[List[float]]
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, r in fit_results.items():
        s_qubit = f"Results for qubit pair {q}:"
        s_qubit += " SUCCESS!\n" if r["success"] else " FAIL!\n"
        s = (
            f"optimal {r['sweep_name']} = {r['optimal_sweep_value']:.4g} "
            f"(metric={r['sweep_name']}) | "
            f"F* @ {r['optimal_sweep_value_fidelity']:.4g}, "
            f"V* @ {r['optimal_sweep_value_visibility']:.4g} | "
            f"F = {r['readout_fidelity']:.1f} % | "
            f"V = {r['visibility']:.3f} | "
            f"IW angle = {r['iw_angle'] * 180 / np.pi:.1f} deg"
        )
        log_callable(s_qubit + s)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Strip tuple wrapping coming out of the data fetcher."""

    def extract_value(element):
        if isinstance(element, tuple):
            return element[0]
        return element

    ds = xr.apply_ufunc(
        extract_value,
        ds,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    return ds


def _fit_single_slice(
    *,
    I: np.ndarray = None,
    Q: np.ndarray = None,
    Ig: np.ndarray = None,
    Qg: np.ndarray = None,
    Ie: np.ndarray = None,
    Qe: np.ndarray = None,
) -> Dict:
    """Fit the Barthel model on a single (qubit, sweep_value) slice of IQ data.

    Two input modes:
      - Labeled: pass Ig, Qg, Ie, Qe (singlet/triplet preps known). A confusion
        matrix is computed against the ground-truth labels.
      - Unlabeled / mixed: pass I, Q only. The Barthel mixture fit still
        recovers mu_S, mu_T, sigma, pT, T1 and the fidelity/visibility curves,
        but the confusion matrix is NaN.

    On failure returns NaNs with success=0 so the sweep loop can continue.
    """
    try:
        labeled = Ig is not None
        if labeled:
            X_ground = np.column_stack([Ig, Qg])
            X_excited = np.column_stack([Ie, Qe])
            X_combined = jnp.array(np.vstack([X_ground, X_excited]))
            calib_arg = (jnp.array(X_excited), "T")
        else:
            X_combined = jnp.array(np.column_stack([I, Q]))
            calib_arg = None

        y, proj, normalizer, mcmc, samples, _, calib_res = Barthel1DFromIQ.fit(
            X_combined,
            fix_tau_M=1.0,
            calib=calib_arg,
            orient="auto",
            prior_strength=0.3,
            sigma_scale_default=0.25,
        )

        proj_dir = proj.pc1 * proj.sign
        angle = float(jnp.arctan2(proj_dir[1], proj_dir[0]))

        rot = jnp.array([[jnp.cos(angle), jnp.sin(angle)], [-jnp.sin(angle), jnp.cos(angle)]])
        proj_rotated_mean = proj.mean @ rot.T

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

        v_rf_norm = fidelity_res["vrf_opt_aligned"]
        v_rf_phys = float(normalizer.inverse(v_rf_norm))
        I_threshold = float(v_rf_phys + proj_rotated_mean[0])

        # Always report the model-based (Barthel) fidelity and visibility.
        # The confusion matrix from labelled data is retained as diagnostic
        # information but is not used to drive the optimisation.
        readout_fidelity = 100.0 * float(fidelity_res["fidelity_opt"])

        if labeled:
            labels_g, _ = classify_iq_with_pca_threshold(
                jnp.array(X_ground), proj, v_rf_norm, normalizer=normalizer, return_margin=True
            )
            labels_e, _ = classify_iq_with_pca_threshold(
                jnp.array(X_excited), proj, v_rf_norm, normalizer=normalizer, return_margin=True
            )
            labels_g = np.array(labels_g)
            labels_e = np.array(labels_e)

            gg = float(np.mean(labels_g == 0))
            ge = float(np.mean(labels_g == 1))
            eg = float(np.mean(labels_e == 0))
            ee = float(np.mean(labels_e == 1))
        else:
            gg = ge = eg = ee = np.nan

        return {
            "iw_angle": angle,
            "ge_threshold": v_rf_phys,
            "I_threshold": I_threshold,
            "readout_fidelity": readout_fidelity,
            "fidelity_opt": float(fidelity_res["fidelity_opt"]),
            "visibility_opt": float(visibility_res["visibility"]),
            "gg": gg,
            "ge": ge,
            "eg": eg,
            "ee": ee,
            "success": 1.0,
        }
    except Exception:
        return {
            "iw_angle": np.nan,
            "ge_threshold": np.nan,
            "I_threshold": np.nan,
            "readout_fidelity": np.nan,
            "fidelity_opt": np.nan,
            "visibility_opt": np.nan,
            "gg": np.nan,
            "ge": np.nan,
            "eg": np.nan,
            "ee": np.nan,
            "success": 0.0,
        }


_METRIC_KEYS = [
    "iw_angle",
    "ge_threshold",
    "I_threshold",
    "readout_fidelity",
    "fidelity_opt",
    "visibility_opt",
    "gg",
    "ge",
    "eg",
    "ee",
    "success",
]


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit Barthel model for every (qubit, sweep_value) slice.

    The sweep coordinate is selected via ``node.parameters.sweep_name`` so the
    same analysis can be reused for detuning, integration-time, or any other
    1D-per-qubit sweep. Expects ``ds`` to contain ``Ig, Qg, Ie, Qe`` with
    dims ``(qubit, n_runs, <sweep>)``.
    """
    sweep_name = node.parameters.sweep_name
    if sweep_name not in ds.coords and sweep_name not in ds.dims:
        raise KeyError(f"sweep_name='{sweep_name}' not found in ds. " f"Available coords: {list(ds.coords)}")
    sweep_values = np.asarray(ds[sweep_name].values)
    n_sweep = len(sweep_values)

    qubit_pairs = node.namespace["qubit_pairs"]
    qubit_pair_names = [qp.name for qp in qubit_pairs]
    n_q = len(qubit_pairs)

    arrays = {k: np.full((n_q, n_sweep), np.nan) for k in _METRIC_KEYS}

    labeled = bool(getattr(node.parameters, "labeled_states", False))
    if labeled and not {"Ig", "Qg", "Ie", "Qe"}.issubset(ds.data_vars):
        raise KeyError("labeled_states=True but ds is missing one of Ig/Qg/Ie/Qe. " f"Found: {list(ds.data_vars)}")
    if not labeled and not {"I", "Q"}.issubset(ds.data_vars):
        raise KeyError("labeled_states=False but ds is missing I/Q. " f"Found: {list(ds.data_vars)}")

    for qi, qp in enumerate(qubit_pairs):
        for si in range(n_sweep):
            sel = {"qubit_pair": qp.name, sweep_name: sweep_values[si]}
            if labeled:
                res = _fit_single_slice(
                    Ig=np.asarray(ds.Ig.sel(**sel).values),
                    Qg=np.asarray(ds.Qg.sel(**sel).values),
                    Ie=np.asarray(ds.Ie.sel(**sel).values),
                    Qe=np.asarray(ds.Qe.sel(**sel).values),
                )
            else:
                res = _fit_single_slice(
                    I=np.asarray(ds.I.sel(**sel).values),
                    Q=np.asarray(ds.Q.sel(**sel).values),
                )
            for k in _METRIC_KEYS:
                arrays[k][qi, si] = float(res[k])

    ds_fit = xr.Dataset(coords={"qubit_pair": qubit_pair_names, sweep_name: sweep_values})
    for k in _METRIC_KEYS:
        ds_fit = ds_fit.assign({k: xr.DataArray(arrays[k], dims=["qubit_pair", sweep_name])})

    metric_choice = node.parameters.optimization_metric
    fit_results: Dict[str, FitParameters] = {}

    opt_val_fid, opt_val_vis, opt_val_selected = [], [], []

    for qi, qname in enumerate(qubit_pair_names):
        fid_curve = arrays["readout_fidelity"][qi]
        vis_curve = arrays["visibility_opt"][qi]

        idx_f, val_f = _argmax_safe(fid_curve, sweep_values)
        idx_v, val_v = _argmax_safe(vis_curve, sweep_values)

        if metric_choice == "fidelity":
            opt_idx, opt_val = idx_f, val_f
        else:
            opt_idx, opt_val = idx_v, val_v

        success = (opt_idx >= 0) and bool(arrays["success"][qi, opt_idx] > 0.5)

        if opt_idx >= 0:
            cm = [
                [float(arrays["gg"][qi, opt_idx]), float(arrays["ge"][qi, opt_idx])],
                [float(arrays["eg"][qi, opt_idx]), float(arrays["ee"][qi, opt_idx])],
            ]
            iw_angle = float(arrays["iw_angle"][qi, opt_idx])
            I_threshold = float(arrays["I_threshold"][qi, opt_idx])
            ge_threshold = float(arrays["ge_threshold"][qi, opt_idx])
            readout_threshold = I_threshold
            readout_projector = {
                "wI": float(np.cos(iw_angle)),
                "wQ": float(np.sin(iw_angle)),
                "offset": 0.0,
            }
            readout_fidelity = float(arrays["readout_fidelity"][qi, opt_idx])
            visibility = float(arrays["visibility_opt"][qi, opt_idx])
        else:
            cm = [[np.nan, np.nan], [np.nan, np.nan]]
            iw_angle = I_threshold = ge_threshold = readout_fidelity = visibility = np.nan
            readout_threshold = np.nan
            readout_projector = {"wI": np.nan, "wQ": np.nan, "offset": np.nan}

        fit_results[qname] = FitParameters(
            sweep_name=sweep_name,
            sweep_values=[float(v) for v in sweep_values],
            optimal_sweep_value=float(opt_val),
            optimal_sweep_index=int(opt_idx),
            optimal_sweep_value_fidelity=float(val_f),
            optimal_sweep_index_fidelity=int(idx_f),
            optimal_sweep_value_visibility=float(val_v),
            optimal_sweep_index_visibility=int(idx_v),
            iw_angle=iw_angle,
            I_threshold=I_threshold,
            ge_threshold=ge_threshold,
            readout_threshold=readout_threshold,
            readout_projector=readout_projector,
            readout_fidelity=readout_fidelity,
            visibility=visibility,
            confusion_matrix=cm,
            success=success,
        )
        opt_val_fid.append(val_f)
        opt_val_vis.append(val_v)
        opt_val_selected.append(opt_val)

    ds_fit = ds_fit.assign(
        {
            "optimal_sweep_value_fidelity": xr.DataArray(opt_val_fid, coords={"qubit_pair": qubit_pair_names}),
            "optimal_sweep_value_visibility": xr.DataArray(opt_val_vis, coords={"qubit_pair": qubit_pair_names}),
            "optimal_sweep_value": xr.DataArray(opt_val_selected, coords={"qubit_pair": qubit_pair_names}),
            "success_overall": xr.DataArray(
                [fit_results[qp].success for qp in qubit_pair_names], coords={"qubit_pair": qubit_pair_names}
            ),
        }
    )
    ds_fit[sweep_name].attrs = ds[sweep_name].attrs

    return ds_fit, fit_results


def _argmax_safe(values: np.ndarray, coords: np.ndarray) -> Tuple[int, float]:
    if np.all(np.isnan(values)):
        return -1, float("nan")
    idx = int(np.nanargmax(values))
    return idx, float(coords[idx])
