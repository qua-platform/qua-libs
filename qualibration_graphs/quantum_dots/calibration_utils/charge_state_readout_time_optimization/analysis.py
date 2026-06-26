import logging
from dataclasses import dataclass
from typing import Tuple, Dict, List, Any
import numpy as np
import xarray as xr

from qualibrate.core import QualibrationNode


_ELONGATION_THRESHOLD = 1  # aspect ratio above which a double Gaussian is fitted to (0,2)


@dataclass
class FitParameters:
    """Stores the SNR analysis results for a single sensor / dot-pair combination."""

    snr_values: List[float]
    integration_times: List[float]
    optimal_integration_time: float
    threshold_snr: float
    iw_angle: float          # rotation angle for integration weights; (1,1) maps to higher I
    I_threshold: float       # discrimination threshold in the rotated I frame
    aspect_ratio_02: float   # elongation of the (0,2) blob; > _ELONGATION_THRESHOLD triggers double-Gaussian
    used_double_gaussian: bool
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for key, r in fit_results.items():
        status = "SUCCESS!" if r["success"] else "FAIL (threshold not reached)!"
        dg = " [double-Gaussian (0,2)]" if r.get("used_double_gaussian") else ""
        log_callable(
            f"Results for {key}: {status}\n"
            f"\tSNR threshold: {r['threshold_snr']:.1f} | "
            f"Optimal t_int = {r['optimal_integration_time']:.0f} ns\n"
            f"\tIW angle = {np.degrees(r['iw_angle']):.2f} deg | "
            f"I_threshold = {r['I_threshold']:.6f} | "
            f"(0,2) aspect ratio = {r['aspect_ratio_02']:.2f}{dg}"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    return ds


def _unit_axis(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-20:
        return np.array([1.0, 0.0], dtype=float)
    return v / n


def _find_threshold_crossing(times, snr, threshold):
    """Return the integration time at which SNR first crosses *threshold* (linear interpolation)."""
    above = np.where(snr >= threshold)[0]
    if len(above) == 0:
        return None
    first_above = above[0]
    if first_above == 0:
        return float(times[0])
    t0, t1 = float(times[first_above - 1]), float(times[first_above])
    s0, s1 = float(snr[first_above - 1]), float(snr[first_above])
    frac = (threshold - s0) / (s1 - s0) if s1 != s0 else 0.0
    return t0 + frac * (t1 - t0)


def _em_1d_with_resp(z: np.ndarray, n_iter: int = 60) -> Dict[str, Any]:
    """Fit a 2-component 1D Gaussian mixture via EM and return responsibilities."""
    n = len(z)
    z_sorted = np.sort(z)
    half = n // 2
    mu1 = float(np.mean(z_sorted[:half]))
    mu2 = float(np.mean(z_sorted[half:]))
    s1 = float(np.std(z_sorted[:half]) + 1e-12)
    s2 = float(np.std(z_sorted[half:]) + 1e-12)
    pi1 = 0.5
    r1 = np.full_like(z, 0.5, dtype=float)

    for _ in range(n_iter):
        p1 = pi1 * np.exp(-0.5 * ((z - mu1) / s1) ** 2) / (s1 + 1e-30)
        p2 = (1.0 - pi1) * np.exp(-0.5 * ((z - mu2) / s2) ** 2) / (s2 + 1e-30)
        denom = p1 + p2 + 1e-30
        r1 = p1 / denom
        r2 = 1.0 - r1
        N1 = r1.sum() + 1e-30
        N2 = r2.sum() + 1e-30
        mu1 = float(np.dot(r1, z) / N1)
        mu2 = float(np.dot(r2, z) / N2)
        s1 = float(np.sqrt(np.dot(r1, (z - mu1) ** 2) / N1 + 1e-12))
        s2 = float(np.sqrt(np.dot(r2, (z - mu2) ** 2) / N2 + 1e-12))
        pi1 = float(N1 / n)

    if mu1 <= mu2:
        comp_a = {"mu": mu1, "sigma": s1, "weight": pi1, "resp": r1}
        comp_b = {"mu": mu2, "sigma": s2, "weight": 1.0 - pi1, "resp": 1.0 - r1}
    else:
        comp_a = {"mu": mu2, "sigma": s2, "weight": 1.0 - pi1, "resp": 1.0 - r1}
        comp_b = {"mu": mu1, "sigma": s1, "weight": pi1, "resp": r1}
    return {"components": [comp_a, comp_b]}


def _fit_02_components_rotated(
    I_02_rot: np.ndarray,
    Q_02_rot: np.ndarray,
    mu_11_rot: float,
) -> Dict[str, Any]:
    """Fit one or two components for (0,2) in rotated coordinates."""
    cov_02 = np.cov(I_02_rot, Q_02_rot)
    eigvals = np.linalg.eigvalsh(cov_02)
    aspect_ratio = float(np.sqrt(max(float(eigvals[-1]), 1e-30) / max(float(eigvals[0]), 1e-30)))
    used_double_gaussian = bool(aspect_ratio > _ELONGATION_THRESHOLD)

    if used_double_gaussian:
        em = _em_1d_with_resp(I_02_rot)
        components = em["components"]
        ref_idx = int(np.argmax([abs(c["mu"] - mu_11_rot) for c in components]))
    else:
        components = [
            {
                "mu": float(np.mean(I_02_rot)),
                "sigma": float(np.std(I_02_rot)),
                "weight": 1.0,
                "resp": np.ones_like(I_02_rot),
            }
        ]
        ref_idx = 0

    return {
        "aspect_ratio": aspect_ratio,
        "used_double_gaussian": used_double_gaussian,
        "components": components,
        "ref_idx": ref_idx,
    }


def _compute_snr_single_time(
    I_11: np.ndarray,
    Q_11: np.ndarray,
    I_02: np.ndarray,
    Q_02: np.ndarray,
) -> Dict[str, Any]:
    """Compute SNR at one integration time using far (0,2) component when split."""
    I_11 = np.asarray(I_11, dtype=float).ravel()
    Q_11 = np.asarray(Q_11, dtype=float).ravel()
    I_02 = np.asarray(I_02, dtype=float).ravel()
    Q_02 = np.asarray(Q_02, dtype=float).ravel()

    mu_11 = np.array([float(np.mean(I_11)), float(np.mean(Q_11))], dtype=float)
    mu_02 = np.array([float(np.mean(I_02)), float(np.mean(Q_02))], dtype=float)
    axis_raw = _unit_axis(mu_02 - mu_11)
    iw_angle = float(np.arctan2(-axis_raw[1], -axis_raw[0]))
    c, s = np.cos(iw_angle), np.sin(iw_angle)

    I_11_rot = I_11 * c + Q_11 * s
    I_02_rot = I_02 * c + Q_02 * s
    Q_02_rot = -I_02 * s + Q_02 * c
    mu_11_rot = float(np.mean(I_11_rot))

    sigma_11 = float(np.std(I_11_rot))
    fit_02 = _fit_02_components_rotated(I_02_rot, Q_02_rot, mu_11_rot)
    comp_ref = fit_02["components"][fit_02["ref_idx"]]
    mu_02_ref = float(comp_ref["mu"])
    sigma_02 = float(comp_ref["sigma"])
    delta_rf = float(abs(mu_02_ref - mu_11_rot))
    denom = sigma_11**2 + sigma_02**2
    snr = float(delta_rf**2 / denom) if denom > 0 else 0.0

    return {
        "snr": snr,
        "delta_rf": delta_rf,
        "sigma_11": sigma_11,
        "sigma_02": sigma_02,
        "mu_11": mu_11,
        "mu_02": mu_02,
        "axis": axis_raw,
        "iw_angle": iw_angle,
        "aspect_ratio": float(fit_02["aspect_ratio"]),
        "used_double_gaussian": bool(fit_02["used_double_gaussian"]),
    }


def _fit_components_for_plot(
    I_11: np.ndarray,
    Q_11: np.ndarray,
    I_02: np.ndarray,
    Q_02: np.ndarray,
) -> Dict[str, Any]:
    """Return Gaussian components in IQ for plotting (2 or 3 total)."""
    I_11 = np.asarray(I_11, dtype=float).ravel()
    Q_11 = np.asarray(Q_11, dtype=float).ravel()
    I_02 = np.asarray(I_02, dtype=float).ravel()
    Q_02 = np.asarray(Q_02, dtype=float).ravel()

    mu_11 = np.array([float(np.mean(I_11)), float(np.mean(Q_11))], dtype=float)
    cov_11 = np.cov(I_11, Q_11)
    mu_02 = np.array([float(np.mean(I_02)), float(np.mean(Q_02))], dtype=float)
    axis_raw = _unit_axis(mu_02 - mu_11)
    iw_angle = float(np.arctan2(-axis_raw[1], -axis_raw[0]))
    c, s = np.cos(iw_angle), np.sin(iw_angle)

    I_11_rot = I_11 * c + Q_11 * s
    I_02_rot = I_02 * c + Q_02 * s
    Q_02_rot = -I_02 * s + Q_02 * c
    mu_11_rot = float(np.mean(I_11_rot))
    fit_02 = _fit_02_components_rotated(I_02_rot, Q_02_rot, mu_11_rot)

    components_02 = []
    for idx, comp in enumerate(fit_02["components"]):
        resp = np.asarray(comp["resp"], dtype=float)
        w = resp / (float(np.sum(resp)) + 1e-30)
        mu_i = np.array([float(np.dot(w, I_02)), float(np.dot(w, Q_02))], dtype=float)
        dI = I_02 - mu_i[0]
        dQ = Q_02 - mu_i[1]
        cov_i = np.array(
            [
                [float(np.dot(w, dI * dI)), float(np.dot(w, dI * dQ))],
                [float(np.dot(w, dQ * dI)), float(np.dot(w, dQ * dQ))],
            ],
            dtype=float,
        )
        cov_i = (cov_i + cov_i.T) * 0.5 + np.eye(2) * 1e-18
        components_02.append(
            {
                "mu_iq": mu_i,
                "cov_iq": cov_i,
                "weight": float(comp["weight"]),
                "is_ref": idx == fit_02["ref_idx"],
            }
        )

    mu_ref = components_02[fit_02["ref_idx"]]["mu_iq"]
    return {
        "snr": _compute_snr_single_time(I_11, Q_11, I_02, Q_02)["snr"],
        "mu_11": mu_11,
        "cov_11": cov_11,
        "mu_ref": mu_ref,
        "components_02": components_02,
        "used_double_gaussian": bool(fit_02["used_double_gaussian"]),
    }


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    For each (dot_pair, sensor), compute SNR vs integration time, then:
    1. Find the optimal integration time from the SNR threshold crossing.
    2. Derive the integration weight angle from the blob separation at max integration time
       (best statistics for direction), orienting so (1,1) maps to higher rotated I.
    3. Rotate the data at the optimal integration time and characterise the (0,2) blob:
       - If its aspect ratio along the rotated I axis exceeds _ELONGATION_THRESHOLD, fit a
         double Gaussian and use the component furthest from (1,1) as the true (0,2) reference
         (T1-limited readout).
    4. Set the threshold at the midpoint between (1,1) and the (0,2) reference in rotated I.
    """
    all_sensors = node.namespace["all_sensors"]
    quantum_dot_pairs = node.namespace["quantum_dot_pairs"]
    integration_times = ds.integration_time.values

    fit_results = {}
    ds_fit_vars = {}

    for dp in quantum_dot_pairs:
        for sensor in all_sensors[dp.name]:
            key = f"{dp.name}_{sensor.name}"

            I_11_var = ds[f"I_11_{key}"]
            Q_11_var = ds[f"Q_11_{key}"]
            I_02_var = ds[f"I_02_{key}"]
            Q_02_var = ds[f"Q_02_{key}"]

            # --- SNR sweep: use far (0,2) component when split ---
            snr_list = []
            delta_rf_list = []
            result = None
            for t_idx in range(len(integration_times)):
                result = _compute_snr_single_time(
                    I_11_var.isel(integration_time=t_idx).values.flatten(),
                    Q_11_var.isel(integration_time=t_idx).values.flatten(),
                    I_02_var.isel(integration_time=t_idx).values.flatten(),
                    Q_02_var.isel(integration_time=t_idx).values.flatten(),
                )
                snr_list.append(result["snr"])
                delta_rf_list.append(result["delta_rf"])

            snr_arr = np.array(snr_list)
            threshold = node.parameters.threshold_SNR

            ds_fit_vars[f"snr_{key}"] = xr.DataArray(
                snr_arr,
                dims=["integration_time"],
                coords={"integration_time": integration_times},
                attrs={"long_name": f"SNR ({key})", "units": ""},
            )
            ds_fit_vars[f"delta_rf_{key}"] = xr.DataArray(
                delta_rf_list,
                dims=["integration_time"],
                coords={"integration_time": integration_times},
                attrs={"long_name": f"Peak separation ({key})", "units": "V"},
            )

            optimal_time = _find_threshold_crossing(integration_times, snr_arr, threshold)
            reached_threshold = optimal_time is not None
            if not reached_threshold:
                optimal_time = float(integration_times[-1])
            optimal_time = int(np.round(optimal_time / 4) * 4)

            # Integration-weight angle from max-time data
            iw_angle = float(result["iw_angle"])

            # --- Threshold from optimal-time data rotated by iw_angle ---
            t_opt_idx = int(np.argmin(np.abs(integration_times - optimal_time)))
            I_11_opt = I_11_var.isel(integration_time=t_opt_idx).values.flatten()
            Q_11_opt = Q_11_var.isel(integration_time=t_opt_idx).values.flatten()
            I_02_opt = I_02_var.isel(integration_time=t_opt_idx).values.flatten()
            Q_02_opt = Q_02_var.isel(integration_time=t_opt_idx).values.flatten()

            # Rotate the measured IQ by the calculated rotation angle
            cos_a, sin_a = np.cos(iw_angle), np.sin(iw_angle)
            I_11_rot = I_11_opt * cos_a + Q_11_opt * sin_a
            I_02_rot = I_02_opt * cos_a + Q_02_opt * sin_a
            Q_02_rot = -I_02_opt * sin_a + Q_02_opt * cos_a

            mu_11_rot = float(np.mean(I_11_rot))
            fit_02 = _fit_02_components_rotated(I_02_rot, Q_02_rot, mu_11_rot)
            comp_ref = fit_02["components"][fit_02["ref_idx"]]
            mu_02_ref = float(comp_ref["mu"])
            aspect_ratio = float(fit_02["aspect_ratio"])
            used_double_gaussian = bool(fit_02["used_double_gaussian"])

            I_threshold = (mu_11_rot + mu_02_ref) / 2.0

            fit_results[key] = FitParameters(
                snr_values=[float(s) for s in snr_arr],
                integration_times=[float(t) for t in integration_times],
                optimal_integration_time=float(optimal_time),
                threshold_snr=float(threshold),
                iw_angle=iw_angle,
                I_threshold=float(I_threshold),
                aspect_ratio_02=aspect_ratio,
                used_double_gaussian=used_double_gaussian,
                success=reached_threshold,
            )

    return xr.Dataset(ds_fit_vars), fit_results
