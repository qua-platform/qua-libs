"""Analysis functions for CZ phase compensation with error amplification calibration.

For each qubit (control / target), the fitted phase is the frame value whose
vertical linecut (across all number_of_operations) has the highest mean signal.
When state discrimination is used this mean should approach 1 at the optimal
compensation point.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V


@dataclass
class FitResults:
    """Stores fit results for a single qubit pair."""

    fitted_control_phase: float
    fitted_target_phase: float
    control_mean_at_peak: float
    target_mean_at_peak: float
    success: bool


def _find_phase_from_max_mean(signal: xr.DataArray) -> Tuple[float, float, np.ndarray]:
    """Find the frame value whose vertical linecut has the highest mean across operations.

    Returns (best_frame, best_mean, mean_curve_as_numpy).
    """
    mean_vs_frame = signal.mean(dim="number_of_operations")
    frames = mean_vs_frame.frame.values
    values = mean_vs_frame.values
    best_idx = int(np.argmax(values))
    return float(frames[best_idx]), float(values[best_idx]), values


def log_fitted_results(fit_results: Dict[str, FitResults], log_callable=None):
    """Log fit results for each qubit pair."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qp_name, fit_result in fit_results.items():
        status = "SUCCESS" if fit_result.success else "FAIL"
        log_callable(
            f"{qp_name} [{status}] | "
            f"control_phase={fit_result.fitted_control_phase:.6f}, "
            f"target_phase={fit_result.fitted_target_phase:.6f}, "
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
    """Find the optimal phase compensation by locating the frame with highest mean signal."""
    frames = ds.frame.values
    fit_datasets = []
    fit_results: Dict[str, FitResults] = {}

    for qp in node.namespace["qubit_pairs"]:
        qp_name = qp.name
        qp_data = ds.sel(qubit_pair=qp_name)

        try:
            signal_control, signal_target = _get_signals(qp_data)

            control_phase, control_peak_mean, control_mean_curve = _find_phase_from_max_mean(signal_control)
            target_phase, target_peak_mean, target_mean_curve = _find_phase_from_max_mean(signal_target)

            fit_ds = xr.Dataset(
                {
                    "control_mean_vs_frame": xr.DataArray(control_mean_curve, dims=("frame",), coords={"frame": frames}),
                    "target_mean_vs_frame": xr.DataArray(target_mean_curve, dims=("frame",), coords={"frame": frames}),
                    "fitted_control_phase": xr.DataArray(control_phase),
                    "fitted_target_phase": xr.DataArray(target_phase),
                    "control_mean_at_peak": xr.DataArray(control_peak_mean),
                    "target_mean_at_peak": xr.DataArray(target_peak_mean),
                    "success": xr.DataArray(True),
                }
            )
            fit_results[qp_name] = FitResults(
                fitted_control_phase=control_phase,
                fitted_target_phase=target_phase,
                control_mean_at_peak=control_peak_mean,
                target_mean_at_peak=target_peak_mean,
                success=True,
            )
        except Exception:
            fit_ds = xr.Dataset({"success": xr.DataArray(False)})
            fit_results[qp_name] = FitResults(
                fitted_control_phase=np.nan,
                fitted_target_phase=np.nan,
                control_mean_at_peak=np.nan,
                target_mean_at_peak=np.nan,
                success=False,
            )

        fit_datasets.append(fit_ds.expand_dims(qubit_pair=[qp_name]))

    ds_fit = xr.concat(fit_datasets, dim="qubit_pair")
    return ds_fit, fit_results
