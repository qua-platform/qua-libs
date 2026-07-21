import logging
from dataclasses import dataclass
from typing import Tuple, Dict, List
import numpy as np
import xarray as xr
import jax
import jax.numpy as jnp
from jax import config as jax_config

from qualibrate import QualibrationNode

from calibration_utils.iq_blobs.readout_barthel.utils import Barthel1DMetricCurves
from calibration_utils.iq_blobs.readout_barthel.calibrate import Barthel1DFromIQ
from calibration_utils.iq_blobs.readout_barthel.classify import (
    classify_iq_with_pca_threshold,
)

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

        rot = jnp.array(
            [[jnp.cos(angle), jnp.sin(angle)], [-jnp.sin(angle), jnp.cos(angle)]]
        )
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
                jnp.array(X_ground),
                proj,
                v_rf_norm,
                normalizer=normalizer,
                return_margin=True,
            )
            labels_e, _ = classify_iq_with_pca_threshold(
                jnp.array(X_excited),
                proj,
                v_rf_norm,
                normalizer=normalizer,
                return_margin=True,
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


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit Barthel model for every (qubit, sweep_value) slice.

    The sweep coordinate is selected via ``node.parameters.sweep_name`` so the
    same analysis can be reused for detuning, integration-time, or any other
    1D-per-qubit sweep. Expects ``ds`` to contain ``Ig, Qg, Ie, Qe`` with
    dims ``(qubit, n_runs, <sweep>)``.
    """
    sweep_name = node.parameters.sweep_name
    if sweep_name not in ds.coords and sweep_name not in ds.dims:
        raise KeyError(
            f"sweep_name='{sweep_name}' not found in ds. "
            f"Available coords: {list(ds.coords)}"
        )
    sweep_values = np.asarray(ds[sweep_name].values)
    n_sweep = len(sweep_values)

    qubit_pairs = node.namespace["qubit_pairs"]
    qubit_pair_names = [qp.name for qp in qubit_pairs]
    n_q = len(qubit_pairs)

    arrays = {k: np.full((n_q, n_sweep), np.nan) for k in _METRIC_KEYS}

    labeled = bool(getattr(node.parameters, "labeled_states", False))
    if labeled and not {"Ig", "Qg", "Ie", "Qe"}.issubset(ds.data_vars):
        raise KeyError(
            "labeled_states=True but ds is missing one of Ig/Qg/Ie/Qe. "
            f"Found: {list(ds.data_vars)}"
        )
    if not labeled and not {"I", "Q"}.issubset(ds.data_vars):
        raise KeyError(
            "labeled_states=False but ds is missing I/Q. "
            f"Found: {list(ds.data_vars)}"
        )

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

    ds_fit = xr.Dataset(
        coords={"qubit_pair": qubit_pair_names, sweep_name: sweep_values}
    )
    for k in _METRIC_KEYS:
        ds_fit = ds_fit.assign(
            {k: xr.DataArray(arrays[k], dims=["qubit_pair", sweep_name])}
        )

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
            iw_angle = I_threshold = ge_threshold = readout_fidelity = visibility = (
                np.nan
            )
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
            "optimal_sweep_value_fidelity": xr.DataArray(
                opt_val_fid, coords={"qubit_pair": qubit_pair_names}
            ),
            "optimal_sweep_value_visibility": xr.DataArray(
                opt_val_vis, coords={"qubit_pair": qubit_pair_names}
            ),
            "optimal_sweep_value": xr.DataArray(
                opt_val_selected, coords={"qubit_pair": qubit_pair_names}
            ),
            "success_overall": xr.DataArray(
                [fit_results[qp].success for qp in qubit_pair_names],
                coords={"qubit_pair": qubit_pair_names},
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


# ---------------------------------------------------------------------------
# PCA + 2-Gaussian alternative analysis
# ---------------------------------------------------------------------------


def _pca_projector_np(
    I: np.ndarray, Q: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """First PCA component from all IQ samples.

    Returns (w, mu): w is the unit PC1 direction (shape (2,)), mu is the data
    mean (shape (2,)). The direction corresponds to the axis of maximum variance
    across *all* shots and sweep points combined.
    """
    X = np.column_stack([I.ravel(), Q.ravel()]).astype(np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    _, eigvecs = np.linalg.eigh(cov)
    return eigvecs[:, -1], mu  # eigenvec for largest eigenvalue


def _em_two_gaussians_1d(z: jnp.ndarray, n_iter: int = 80) -> Tuple:
    """Fixed-iteration EM for a 2-component Gaussian mixture on 1D data.

    Initialises cluster centres and widths from the lower and upper halves of
    the sorted data, then runs ``n_iter`` E/M steps via ``jax.lax.scan``.  The
    fixed iteration count makes the function vmappable.

    Sigma is initialised per-cluster from within-cluster variance of the
    sorted split, which gives a much tighter starting point than using the
    global std — particularly important when the blobs are well-separated.

    Returns ``(mu1, sigma1, mu2, sigma2)`` with ``mu1 <= mu2`` enforced.
    """
    n = z.shape[0]
    z_sorted = jnp.sort(z)
    half = n // 2
    lower = z_sorted[:half]
    upper = z_sorted[half:]
    mu1_init = jnp.mean(lower)
    mu2_init = jnp.mean(upper)
    # Per-cluster sigma from within-cluster spread; falls back gracefully when
    # the distribution is unimodal (lower/upper halves of a single Gaussian).
    sigma1_init = jnp.std(lower) + 1e-12
    sigma2_init = jnp.std(upper) + 1e-12
    pi_init = jnp.array(0.5)

    def _step(carry, _):
        mu1, mu2, s1, s2, pi1 = carry
        # E-step in log-space for numerical stability
        log_p1 = jnp.log(pi1 + 1e-30) - 0.5 * ((z - mu1) / s1) ** 2 - jnp.log(s1)
        log_p2 = jnp.log(1.0 - pi1 + 1e-30) - 0.5 * ((z - mu2) / s2) ** 2 - jnp.log(s2)
        shift = jnp.maximum(log_p1, log_p2)
        r1 = jnp.exp(log_p1 - shift)
        r2 = jnp.exp(log_p2 - shift)
        r1 = r1 / (r1 + r2)
        r2 = 1.0 - r1
        # M-step
        N1 = jnp.sum(r1) + 1e-30
        N2 = jnp.sum(r2) + 1e-30
        mu1_new = jnp.dot(r1, z) / N1
        mu2_new = jnp.dot(r2, z) / N2
        s1_new = jnp.sqrt(jnp.dot(r1, (z - mu1_new) ** 2) / N1 + 1e-12)
        s2_new = jnp.sqrt(jnp.dot(r2, (z - mu2_new) ** 2) / N2 + 1e-12)
        pi1_new = N1 / n
        return (mu1_new, mu2_new, s1_new, s2_new, pi1_new), None

    init = (mu1_init, mu2_init, sigma1_init, sigma2_init, pi_init)
    (mu1_f, mu2_f, s1_f, s2_f, _), _ = jax.lax.scan(_step, init, None, length=n_iter)

    swap = mu1_f > mu2_f
    mu1 = jnp.where(swap, mu2_f, mu1_f)
    mu2 = jnp.where(swap, mu1_f, mu2_f)
    s1 = jnp.where(swap, s2_f, s1_f)
    s2 = jnp.where(swap, s1_f, s2_f)
    return mu1, s1, mu2, s2


# Compiled vmap over the sweep axis: input Z shape (n_sweep, n_runs).
_vmap_em_two_gaussians = jax.jit(jax.vmap(_em_two_gaussians_1d))


def _two_gaussian_fidelity_visibility(
    mu1: jnp.ndarray,
    sigma1: jnp.ndarray,
    mu2: jnp.ndarray,
    sigma2: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Analytical fidelity and visibility for two equal-weight 1D Gaussians.

    Threshold placed at the midpoint t* = (mu1+mu2)/2, which is the optimal
    decision boundary when the two components have equal weights and equal
    widths (a good approximation for readout shot noise).

    Error probability:
        P_err = (erfc(Δ / (2·σ₁·√2)) + erfc(Δ / (2·σ₂·√2))) / 4
    where Δ = |mu2 - mu1|.

    Fidelity F = 1 − P_err  (ranges 0.5 → 1).
    Visibility V = 2·F − 1  (ranges 0 → 1).
    """
    delta = jnp.abs(mu2 - mu1)
    sqrt2 = jnp.sqrt(jnp.array(2.0))
    p_err = (
        jax.scipy.special.erfc(delta / (2.0 * sigma1 * sqrt2))
        + jax.scipy.special.erfc(delta / (2.0 * sigma2 * sqrt2))
    ) / 4.0
    fidelity = 1.0 - p_err
    visibility = 2.0 * fidelity - 1.0
    return fidelity, visibility


_PCA_GMM_KEYS = [
    "mu1",
    "mu2",
    "sigma1",
    "sigma2",
    "fidelity_opt",
    "visibility_opt",
    "readout_fidelity",
    "iw_angle",
    "I_threshold",
    "ge_threshold",
    "success",
]


def fit_raw_data_pca_gaussian(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """PCA-projection + analytical two-Gaussian alternative to :func:`fit_raw_data`.

    For each qubit pair:

    1. Compute a **global** PCA projector from all IQ shots across the entire
       sweep — one direction captures the axis of maximum readout contrast.
    2. Project every sweep slice from 2D IQ to 1D (vectorised numpy).
    3. ``vmap`` a fixed-iteration EM algorithm over the sweep axis to fit two
       1D Gaussians at each detuning/sweep point (no Python loop over sweep).
    4. Compute fidelity and visibility analytically from the Gaussian means and
       widths — no MCMC required.

    Returns the same ``(ds_fit, Dict[str, FitParameters])`` format as
    :func:`fit_raw_data` so it can be used as a drop-in replacement.
    """
    sweep_name = node.parameters.sweep_name
    if sweep_name not in ds.coords and sweep_name not in ds.dims:
        raise KeyError(
            f"sweep_name='{sweep_name}' not found in ds. "
            f"Available coords: {list(ds.coords)}"
        )
    sweep_values = np.asarray(ds[sweep_name].values)
    n_sweep = len(sweep_values)

    qubit_pairs = node.namespace["qubit_pairs"]
    qubit_pair_names = [qp.name for qp in qubit_pairs]
    n_q = len(qubit_pairs)

    arrays = {k: np.full((n_q, n_sweep), np.nan) for k in _PCA_GMM_KEYS}

    for qi, qp in enumerate(qubit_pairs):
        # I, Q: (n_runs, n_sweep)
        I = np.asarray(ds.I.sel(qubit_pair=qp.name).values, dtype=np.float64)
        Q = np.asarray(ds.Q.sel(qubit_pair=qp.name).values, dtype=np.float64)

        z_sweep = []
        z_offset = []
        iw_angles_per_slice = []
        w_prev = None
        for i, q in zip(I.T, Q.T):
            # 1. Per-slice PCA projector.
            w, mu_iq = _pca_projector_np(i, q)
            # Enforce sign continuity across slices: eigh eigenvector sign is arbitrary.
            if w_prev is not None and np.dot(w, w_prev) < 0:
                w = -w
            w_prev = w
            iw_angles_per_slice.append(float(np.arctan2(w[1], w[0])))

            offset = float(mu_iq[0] * w[0] + mu_iq[1] * w[1])
            z = jnp.array(((i - mu_iq[0]) * w[0] + (q - mu_iq[1]) * w[1]).T)

            z_offset.append(offset)
            z_sweep.append(z)

        z_sweep = np.array(z_sweep)
        z_offset = np.array(z_offset)
        # 3. vmap EM across sweep axis; each output has shape (n_sweep,).

        mu1, sigma1, mu2, sigma2 = _vmap_em_two_gaussians(z_sweep)

        # 4. Analytical fidelity and visibility from Gaussian parameters.
        fidelity, visibility = _two_gaussian_fidelity_visibility(
            mu1, sigma1, mu2, sigma2
        )

        # Threshold in the un-centred projected IQ frame (I·wI + Q·wQ > threshold).
        t_star = np.asarray((mu1 + mu2) / 2.0) + z_offset

        fidelity_np = np.asarray(fidelity)
        visibility_np = np.asarray(visibility)
        # Threshold > 0.5 avoids false positives: F >= 0.5 by construction
        # even for zero separation, so 0.5 would mark everything as success.
        success_vec = np.where(
            np.isfinite(fidelity_np) & (fidelity_np > 0.6), 1.0, 0.0
        )

        # mu1/mu2 are in the centered per-slice PCA space; shift by z_offset so
        # they sit in the same uncentered projected frame as the plotted shots
        # (I·cos(iw_angle) + Q·sin(iw_angle)), matching the sweep summary imshow.
        arrays["mu1"][qi] = np.asarray(mu1) + z_offset
        arrays["mu2"][qi] = np.asarray(mu2) + z_offset
        arrays["sigma1"][qi] = np.asarray(sigma1)
        arrays["sigma2"][qi] = np.asarray(sigma2)
        arrays["fidelity_opt"][qi] = fidelity_np
        arrays["visibility_opt"][qi] = visibility_np
        arrays["readout_fidelity"][qi] = 100.0 * fidelity_np
        arrays["iw_angle"][qi] = np.array(iw_angles_per_slice)
        arrays["I_threshold"][qi] = t_star
        arrays["ge_threshold"][qi] = t_star
        arrays["success"][qi] = success_vec

    ds_fit = xr.Dataset(
        coords={"qubit_pair": qubit_pair_names, sweep_name: sweep_values}
    )
    for k in _PCA_GMM_KEYS:
        ds_fit = ds_fit.assign(
            {k: xr.DataArray(arrays[k], dims=["qubit_pair", sweep_name])}
        )

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
            iw_angle = float(arrays["iw_angle"][qi, opt_idx])
            I_threshold = float(arrays["I_threshold"][qi, opt_idx])
            ge_threshold = float(arrays["ge_threshold"][qi, opt_idx])
            readout_fidelity = float(arrays["readout_fidelity"][qi, opt_idx])
            visibility = float(arrays["visibility_opt"][qi, opt_idx])
        else:
            iw_angle = I_threshold = ge_threshold = readout_fidelity = visibility = np.nan

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
            readout_threshold=I_threshold,
            readout_projector={
                "wI": float(np.cos(iw_angle)) if np.isfinite(iw_angle) else np.nan,
                "wQ": float(np.sin(iw_angle)) if np.isfinite(iw_angle) else np.nan,
                "offset": 0.0,
            },
            readout_fidelity=readout_fidelity,
            visibility=visibility,
            confusion_matrix=[[np.nan, np.nan], [np.nan, np.nan]],
            success=success,
        )
        opt_val_fid.append(val_f)
        opt_val_vis.append(val_v)
        opt_val_selected.append(opt_val)

    ds_fit = ds_fit.assign(
        {
            "optimal_sweep_value_fidelity": xr.DataArray(
                opt_val_fid, coords={"qubit_pair": qubit_pair_names}
            ),
            "optimal_sweep_value_visibility": xr.DataArray(
                opt_val_vis, coords={"qubit_pair": qubit_pair_names}
            ),
            "optimal_sweep_value": xr.DataArray(
                opt_val_selected, coords={"qubit_pair": qubit_pair_names}
            ),
            "success_overall": xr.DataArray(
                [fit_results[qp].success for qp in qubit_pair_names],
                coords={"qubit_pair": qubit_pair_names},
            ),
        }
    )
    ds_fit[sweep_name].attrs = ds[sweep_name].attrs

    return ds_fit, fit_results
