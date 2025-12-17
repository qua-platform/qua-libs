import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import fit_oscillation, oscillation
from scipy.ndimage import uniform_filter1d


@dataclass
class FitResults:
    """Stores the relevant CZ conditional phase experiment fit parameters for a single qubit pair"""

    optimal_amplitude: float
    success: bool


def fix_oscillation_phi_2pi(fit_data):
    """Extract the phase parameter from oscillation fit data."""
    # Extract the phase parameter from the fit results
    phase = fit_data.sel(fit_vals="phi")
    # Normalize phase to [0, 1] range (representing 0 to 2π)
    phase = (phase / (2 * np.pi)) % 1
    return phase


def log_fitted_results(fit_results: Dict[str, FitResults], log_callable=None):
    """
    Logs the node-specific fitted results for all qubit pairs.

    Parameters:
    -----------
    fit_results : Dict[str, FitResults]
        Dictionary containing FitResults for each qubit pair.
    log_callable : callable, optional
        Logger for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qp_name, fit_result in fit_results.items():
        s_qubit = f"Results for qubit pair {qp_name}: "
        s_amp = f"\tOptimal CZ amplitude: {fit_result.optimal_amplitude:.6f} a.u."

        if fit_result.success:
            s_qubit += "SUCCESS!\n"
        else:
            s_qubit += "FAIL!\n"

        log_message = s_qubit + s_amp

        log_callable(log_message)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """
    Process the raw dataset by adding amplitude and detuning coordinates.

    Parameters:
    -----------
    ds : xr.Dataset
        Raw dataset from the experiment
    node : QualibrationNode
        The calibration node containing qubit pairs information

    Returns:
    --------
    xr.Dataset
        Processed dataset with additional coordinates
    """
    qubit_pairs = node.namespace["qubit_pairs"]

    operation = node.parameters.operation

    # def abs_amp(qp, amp):
    #     return amp * 1

    # def detuning(qp, amp):
    #     amplitude_squared = (amp * 1) ** 2
    #     return -amplitude_squared * qp.qubit_control.freq_vs_flux_01_quad_term

    # ds = ds.assign_coords({"amp": (["qubit_pair", "amp"], np.array([abs_amp(qp, ds.amp) for qp in qubit_pairs]))})
    # ds = ds.assign_coords({"detuning": (["qubit_pair", "amp"], np.array([detuning(qp, ds.amp) for qp in qubit_pairs]))})

    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitResults]]:
    """
    Fit the CZ conditional phase data for each qubit pair.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the processed data.
    node : QualibrationNode
        The calibration node containing parameters and qubit pairs.

    Returns:
    --------
    Tuple[xr.Dataset, Dict[str, FitResults]]
        Dataset with fit results and dictionary of fit results for each qubit pair.
    """
    ds_fit = ds.groupby("qubit_pair").apply(fit_routine)

    # Extract the relevant fitted parameters
    ds_fit, fit_results = _extract_relevant_parameters(ds_fit, node)

    return ds_fit, fit_results


def fit_routine(da):

    if hasattr(da, "state_target"):
        data = "state_target"
    else:
        data = "I_target"
    # Fit oscillation for each control state and amplitude
    fit_data = fit_oscillation(da[data], "frame")

    # Add fitted oscillation curves to the dataset
    da = da.assign(
        {
            "fitted": oscillation(
                da.frame,
                fit_data.sel(fit_vals="a"),
                fit_data.sel(fit_vals="f"),
                fit_data.sel(fit_vals="phi"),
                fit_data.sel(fit_vals="offset"),
            )
        }
    )

    # Extract phase and calculate phase difference
    phase = fix_oscillation_phi_2pi(fit_data)
    phase_diff = (phase.sel(control_axis=0) - phase.sel(control_axis=1)) % 1

    # Robust analysis: find amplitude closest to π phase difference (0.5 in normalized units)
    # Apply smoothing to make it more robust to noisy data

    # Apply smoothing if we have enough data points
    if len(phase_diff.amp) > 5:
        # Smooth the phase difference data with a rolling window
        window_size = min(5, len(phase_diff.amp) // 3)  # Adaptive window size

        # Handle potentially multidimensional data by flattening along amp axis
        phase_values = phase_diff.values.flatten() if phase_diff.values.ndim > 1 else phase_diff.values
        smoothed_phase_diff = uniform_filter1d(phase_values, size=window_size, mode="nearest")

        # Find points near 0.5 (within some tolerance)
        tolerance = 0.1  # Allow 10% tolerance around 0.5
        distances = np.abs(smoothed_phase_diff - 0.5)

        # Find candidates within tolerance
        candidates_mask = distances <= tolerance

        if np.any(candidates_mask):
            # Use weighted average of good candidates, weighted by inverse distance
            amp_values = phase_diff.amp.values
            candidate_amps = (
                amp_values[candidates_mask] if amp_values.ndim == 1 else amp_values.flatten()[candidates_mask]
            )
            candidate_distances = distances[candidates_mask]

            # Avoid division by zero by adding small epsilon
            weights = 1 / (candidate_distances + 1e-10)
            optimal_amp = float(np.average(candidate_amps, weights=weights))
            success = True
        else:
            # Fallback: find the closest point to 0.5
            min_idx = np.argmin(distances)
            amp_values = phase_diff.amp.values
            optimal_amp = float(amp_values[min_idx] if amp_values.ndim == 1 else amp_values.flatten()[min_idx])
            success = False  # Mark as failed since no good candidates found
    else:
        # Not enough data points for smoothing, use simple approach
        optimal_amp = float(np.abs(phase_diff - 0.5).idxmin("amp"))
        success = len(phase_diff.amp) >= 3  # Need at least 3 points for meaningful result

    fitted_curve = xr.full_like(phase_diff, np.nan)  # No curve fitting, create xarray with same structure

    da = da.assign(
        optimal_amplitude=optimal_amp,
        phase_diff=phase_diff,
        fitted_curve=fitted_curve,
        success=success,
    )

    return da


def _extract_relevant_parameters(
    ds_fit: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitResults]]:
    """
    Extract relevant fit parameters and create FitResults for each qubit pair.

    Parameters:
    -----------
    ds_fit : xr.Dataset
        Dataset containing the fit results from fit_routine.
    node : QualibrationNode
        The calibration node containing parameters and qubit pairs.

    Returns:
    --------
    Tuple[xr.Dataset, Dict[str, FitResults]]
        Dataset with additional metadata and dictionary of FitResults for each qubit pair.
    """
    qubit_pairs = node.namespace["qubit_pairs"]

    # Add metadata attributes to the dataset
    if "optimal_amplitude" in ds_fit.data_vars:
        ds_fit.optimal_amplitude.attrs = {"long_name": "optimal CZ amplitude", "units": "a.u."}
    if "phase_diff" in ds_fit.data_vars:
        ds_fit.phase_diff.attrs = {"long_name": "phase difference", "units": "2π"}
    if "fitted_curve" in ds_fit.data_vars:
        ds_fit.fitted_curve.attrs = {"long_name": "fitted tanh curve", "units": "2π"}

    # Create FitResults for each qubit pair
    fit_results = {}
    for qp in qubit_pairs:
        qp_name = qp.name
        qp_data = ds_fit.sel(qubit_pair=qp_name)

        fit_results[qp_name] = FitResults(
            optimal_amplitude=float(qp_data.optimal_amplitude.values),
            success=bool(qp_data.success.values),
        )

    return ds_fit, fit_results
