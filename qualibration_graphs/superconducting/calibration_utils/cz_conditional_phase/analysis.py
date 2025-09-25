import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import fit_oscillation, oscillation
from scipy.optimize import curve_fit


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


def tanh_fit(x, a, b, c, d):
    """Tanh fitting function for phase difference vs amplitude."""
    return a * np.tanh(b * x + c) + d


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

    def abs_amp(qp, amp):
        return amp * qp.macros[operation].flux_pulse_control.amplitude

    def detuning(qp, amp):
        amplitude_squared = (amp * qp.macros[operation].flux_pulse_control.amplitude) ** 2
        return -amplitude_squared * qp.qubit_control.freq_vs_flux_01_quad_term

    ds = ds.assign_coords({"amp_full": (["qubit_pair", "amp"], np.array([abs_amp(qp, ds.amp) for qp in qubit_pairs]))})
    ds = ds.assign_coords({"detuning": (["qubit_pair", "amp"], np.array([detuning(qp, ds.amp) for qp in qubit_pairs]))})

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

    # Fit tanh curve to find optimal amplitude
    try:
        # The initial guess for the tanh fit is important and needs to be adjusted based on the amplitude range
        amp_range = np.max(phase_diff.amp_full.values) - np.min(phase_diff.amp_full.values)
        p0 = [
            -0.5,  # a
            1 / amp_range,  # b
            -np.mean(phase_diff.amp_full.values) / amp_range,  # c
            0.5,  # d
        ]
        fit_params, _ = curve_fit(tanh_fit, phase_diff.amp_full.values[0], phase_diff.values[0], p0=p0)
        optimal_amp = (np.arctanh((0.5 - fit_params[3]) / fit_params[0]) - fit_params[2]) / fit_params[1]
        fitted_curve = tanh_fit(phase_diff.amp_full, *fit_params)
        success = True

    except Exception as e:
        # Fallback: find amplitude closest to π phase difference (0.5 in normalized units)
        optimal_amp = float(np.abs(phase_diff - 0.5).idxmin("amp_full"))
        fitted_curve = np.full_like(phase_diff.values, np.nan)
        success = False

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
