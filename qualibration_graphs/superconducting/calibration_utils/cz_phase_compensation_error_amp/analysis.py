"""Analysis functions for CZ phase compensation with error amplification calibration.

For each qubit (control / target), the signal is averaged over the number of CZ
operations and fit with a sinc model to locate the optimal compensation frame:

    f(frame) = B + A * sinc(w * (frame - frame_0) / pi)

When state discrimination is used the peak should approach 1 at the optimal
compensation point. A parabolic-refinement fallback is used if the sinc fit fails.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from scipy.optimize import curve_fit


@dataclass
class FitResults:
    """Stores fit results for a single qubit pair."""

    fitted_control_phase: float
    fitted_target_phase: float
    control_mean_at_peak: float
    target_mean_at_peak: float
    success: bool
    control_fit_method: str = "sinc"
    target_fit_method: str = "sinc"
    control_sinc_params: Dict[str, float] = field(default_factory=dict)
    target_sinc_params: Dict[str, float] = field(default_factory=dict)


def _sinc_model(x: np.ndarray, A: float, B: float, x_0: float, w: float) -> np.ndarray:
    """Sinc model used to localise the central peak of the averaged signal."""
    return B + A * np.sinc(w * (x - x_0) / np.pi)


def _parabolic_refine(y: np.ndarray, i: int) -> float:
    """Three-point parabolic refinement around an extremum index, returning fractional index."""
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    y1, y2, y3 = y[i - 1], y[i], y[i + 1]
    denom = y1 - 2.0 * y2 + y3
    if denom == 0:
        return float(i)
    delta = 0.5 * (y1 - y3) / denom
    return float(i) + float(np.clip(delta, -0.5, 0.5))


def _argmax_with_refine(x_values: np.ndarray, y: np.ndarray) -> float:
    """Argmax of ``y`` over ``x_values`` with 3-point parabolic refinement."""
    i = int(np.argmax(y))
    i_star = _parabolic_refine(y, i)
    return float(np.interp(i_star, np.arange(len(x_values)), x_values))


def _estimate_fwhm_around(x_values: np.ndarray, y: np.ndarray, i_max: int) -> float:
    """Estimate the full-width at half-maximum of the central peak around ``i_max``."""
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return float("nan")
    y_max = float(y[i_max])
    y_min = float(np.min(finite))
    half = y_max - (y_max - y_min) / 2.0
    left = i_max
    while left > 0 and y[left] > half:
        left -= 1
    right = i_max
    while right < len(y) - 1 and y[right] > half:
        right += 1
    fwhm = float(x_values[right] - x_values[left])
    if fwhm > 0:
        return fwhm
    return float(x_values[-1] - x_values[0]) / 4.0


def _fit_sinc_peak(
    x_values: np.ndarray, y: np.ndarray
) -> Tuple[float, float, bool, str, Dict[str, float], np.ndarray]:
    """Fit a sinc model to a 1D curve and return the peak position."""
    if len(x_values) != len(y) or len(x_values) == 0:
        return float("nan"), float("nan"), False, "none", {}, np.full_like(x_values, np.nan, dtype=float)

    y_avg = np.asarray(y, dtype=float)
    if not np.any(np.isfinite(y_avg)):
        return float("nan"), float("nan"), False, "none", {}, np.full_like(x_values, np.nan, dtype=float)

    y_finite = np.where(np.isfinite(y_avg), y_avg, -np.inf)
    i_max = int(np.argmax(y_finite))

    A_init = float(np.nanmax(y_avg) - np.nanmedian(y_avg))
    if not np.isfinite(A_init) or A_init <= 0:
        A_init = 0.5
    B_init = float(np.nanmedian(y_avg))
    if not np.isfinite(B_init):
        B_init = 0.5
    x_seed_init = float(x_values[i_max])
    fwhm = _estimate_fwhm_around(x_values, y_avg, i_max)
    if np.isfinite(fwhm) and fwhm > 0:
        w_init = 3.79 / fwhm
    else:
        w_init = 3.79 / max(float(x_values[-1] - x_values[0]) / 4.0, 1e-9)

    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))

    try:
        popt, _ = curve_fit(
            _sinc_model,
            x_values,
            y_avg,
            p0=[A_init, B_init, x_seed_init, w_init],
            bounds=(
                [0.0, -0.5, x_min, w_init / 50.0],
                [2.0, 1.5, x_max, w_init * 50.0],
            ),
            maxfev=10000,
        )
        A_fit, B_fit, x_0_fit, w_fit = map(float, popt)
        if not x_min <= x_0_fit <= x_max:
            raise RuntimeError(f"Fitted frame = {x_0_fit} outside swept window.")
        fit_curve = _sinc_model(x_values, A_fit, B_fit, x_0_fit, w_fit)
        return (
            x_0_fit,
            B_fit + A_fit,
            True,
            "sinc",
            {"A": A_fit, "B": B_fit, "frame_0": x_0_fit, "w": w_fit},
            fit_curve,
        )
    except Exception:  # pylint: disable=broad-except
        x_refined = _argmax_with_refine(x_values, y_finite)
        y_at_peak = float(np.interp(x_refined, x_values, y_avg))
        return (
            x_refined,
            y_at_peak,
            bool(np.isfinite(x_refined)),
            "parabolic",
            {},
            np.full_like(x_values, np.nan, dtype=float),
        )


def _find_phase_from_sinc_fit(
    signal: xr.DataArray,
) -> Tuple[float, float, bool, str, Dict[str, float], np.ndarray, np.ndarray]:
    """Average over operations and fit a sinc model to locate the optimal frame."""
    mean_vs_frame = signal.mean(dim="number_of_operations")
    frames = mean_vs_frame.frame.values
    values = mean_vs_frame.values
    return _fit_sinc_peak(frames, values) + (values,)


def log_fitted_results(fit_results: Dict[str, FitResults], log_callable=None):
    """Log fit results for each qubit pair."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qp_name, fit_result in fit_results.items():
        status = "SUCCESS" if fit_result.success else "FAIL"
        log_callable(
            f"{qp_name} [{status}] | "
            f"control_phase={fit_result.fitted_control_phase:.6f} ({fit_result.control_fit_method}), "
            f"target_phase={fit_result.fitted_target_phase:.6f} ({fit_result.target_fit_method}), "
            f"control_peak_mean={fit_result.control_mean_at_peak:.4f}, "
            f"target_peak_mean={fit_result.target_mean_at_peak:.4f}"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Convert IQ data to volts when state discrimination is disabled."""
    if "I_control" in ds.data_vars:
        ds = convert_IQ_to_V(
            ds,
            qubit_pairs=node.namespace["qubit_pairs"],
            IQ_list=["I_control", "Q_control", "I_target", "Q_target"],
        )
    return ds


def _get_signals(qp_data: xr.Dataset):
    """Extract control and target signal arrays from a single qubit-pair slice."""
    if "state_control" in qp_data.data_vars and "state_target" in qp_data.data_vars:
        return qp_data["state_control"], qp_data["state_target"]

    signal_control = qp_data["I_control"]
    signal_target = qp_data["I_target"]
    if "control_target" in signal_control.dims:
        signal_control = signal_control.sel(control_target="c")
    if "control_target" in signal_target.dims:
        signal_target = signal_target.sel(control_target="t")
    return signal_control, signal_target


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitResults]]:
    """Find optimal phase compensation by fitting a sinc model to the averaged signal."""
    frames = ds.frame.values
    fit_datasets = []
    fit_results: Dict[str, FitResults] = {}

    for qp in node.namespace["qubit_pairs"]:
        qp_name = qp.name
        qp_data = ds.sel(qubit_pair=qp_name)

        try:
            signal_control, signal_target = _get_signals(qp_data)

            (
                control_phase,
                control_peak_mean,
                control_success,
                control_method,
                control_params,
                control_sinc_fit,
                control_mean_curve,
            ) = _find_phase_from_sinc_fit(signal_control)
            (
                target_phase,
                target_peak_mean,
                target_success,
                target_method,
                target_params,
                target_sinc_fit,
                target_mean_curve,
            ) = _find_phase_from_sinc_fit(signal_target)

            success = control_success and target_success

            fit_ds = xr.Dataset(
                {
                    "control_mean_vs_frame": xr.DataArray(
                        control_mean_curve, dims=("frame",), coords={"frame": frames}
                    ),
                    "target_mean_vs_frame": xr.DataArray(
                        target_mean_curve, dims=("frame",), coords={"frame": frames}
                    ),
                    "control_sinc_fit": xr.DataArray(
                        control_sinc_fit, dims=("frame",), coords={"frame": frames}
                    ),
                    "target_sinc_fit": xr.DataArray(
                        target_sinc_fit, dims=("frame",), coords={"frame": frames}
                    ),
                    "fitted_control_phase": xr.DataArray(control_phase),
                    "fitted_target_phase": xr.DataArray(target_phase),
                    "control_mean_at_peak": xr.DataArray(control_peak_mean),
                    "target_mean_at_peak": xr.DataArray(target_peak_mean),
                    "control_fit_method": xr.DataArray(control_method),
                    "target_fit_method": xr.DataArray(target_method),
                    "success": xr.DataArray(success),
                }
            )
            fit_results[qp_name] = FitResults(
                fitted_control_phase=control_phase,
                fitted_target_phase=target_phase,
                control_mean_at_peak=control_peak_mean,
                target_mean_at_peak=target_peak_mean,
                success=success,
                control_fit_method=control_method,
                target_fit_method=target_method,
                control_sinc_params=control_params,
                target_sinc_params=target_params,
            )
        except Exception:
            fit_ds = xr.Dataset({"success": xr.DataArray(False)})
            fit_results[qp_name] = FitResults(
                fitted_control_phase=np.nan,
                fitted_target_phase=np.nan,
                control_mean_at_peak=np.nan,
                target_mean_at_peak=np.nan,
                success=False,
                control_fit_method="none",
                target_fit_method="none",
            )

        fit_datasets.append(fit_ds.expand_dims(qubit_pair=[qp_name]))

    ds_fit = xr.concat(fit_datasets, dim="qubit_pair")
    return ds_fit, fit_results
