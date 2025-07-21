import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import fit_oscillation, oscillation
from scipy.optimize import curve_fit


@dataclass
class CzConditionalPhaseFit:
    """Stores the relevant CZ conditional phase experiment fit parameters for a single qubit pair"""

    optimal_amplitude: float
    phase_diff: xr.DataArray
    fitted_curve: np.ndarray
    leakage: xr.DataArray
    success: bool


def fix_oscillation_phi_2pi(fit_data):
    """Extract and fix the phase parameter from oscillation fit data."""
    # Extract the phase parameter from the fit results
    phase = fit_data.sel(fit_vals="phi")
    # Normalize phase to [0, 1] range (representing 0 to 2π)
    phase = (phase / (2 * np.pi)) % 1
    return phase


def tanh_fit(x, a, b, c, d):
    """Tanh fitting function for phase difference vs amplitude."""
    return a * np.tanh(b * x + c) + d


def log_fitted_results(optimal_amps: Dict[str, float], log_callable=None):
    """
    Logs the node-specific fitted results for all qubit pairs.

    Parameters:
    -----------
    optimal_amps : Dict[str, float]
        Dictionary containing optimal amplitudes for each qubit pair.
    log_callable : callable, optional
        Logger for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qp_name, amp in optimal_amps.items():
        log_callable(f"Optimal CZ amplitude for {qp_name}: {amp:.6f} V --> SUCCESS!")


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

    def abs_amp(qp, amp):
        return amp * qp.macros["cz_unipolar"].flux_pulse_control.amplitude

    def detuning(qp, amp):
        return (
            -((amp * qp.macros["cz_unipolar"].flux_pulse_control.amplitude) ** 2)
            * qp.qubit_control.freq_vs_flux_01_quad_term
        )

    ds = ds.assign_coords({"amp_full": (["qubit_pair", "amp"], np.array([abs_amp(qp, ds.amp) for qp in qubit_pairs]))})
    ds = ds.assign_coords({"detuning": (["qubit_pair", "amp"], np.array([detuning(qp, ds.amp) for qp in qubit_pairs]))})

    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[Dict[str, CzConditionalPhaseFit], Dict[str, float]]:
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
    Tuple[Dict[str, CzConditionalPhaseFit], Dict[str, float]]
        Dictionary of fit results for each qubit pair and optimal amplitudes.
    """
    qubit_pairs = node.namespace["qubit_pairs"]
    fit_results = {}
    optimal_amps = {}

    for qp in qubit_pairs:
        ds_qp = ds.sel(qubit_pair=qp.name)

        # Fit oscillation for each control state and amplitude
        fit_data = fit_oscillation(ds_qp.state_target, "frame")

        # Add fitted oscillation curves to the dataset
        ds_qp = ds_qp.assign(
            {
                "fitted": oscillation(
                    ds_qp.frame,
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
            p0 = [
                -0.5,  # a
                1 / (np.max(phase_diff.amp_full) - np.min(phase_diff.amp_full)),  # b
                -np.mean(phase_diff.amp_full) / (np.max(phase_diff.amp_full) - np.min(phase_diff.amp_full)),  # c
                0.5,  # d
            ]
            fit_params, _ = curve_fit(tanh_fit, phase_diff.amp_full, phase_diff, p0=p0)
            optimal_amp = (np.arctanh((0.5 - fit_params[3]) / fit_params[0]) - fit_params[2]) / fit_params[1]
            fitted_curve = tanh_fit(phase_diff.amp_full, *fit_params)
            success = True
        except Exception as e:
            node.log(f"Fitting failed for {qp.name}: {e}")
            optimal_amp = float(np.abs(phase_diff - 0.5).idxmin("amp_full"))
            fitted_curve = np.full_like(phase_diff.values, np.nan)
            success = False

        optimal_amps[qp.name] = optimal_amp

        # Calculate leakage if measured
        leakage = None
        if node.parameters.measure_leak:
            all_counts = (ds_qp.state_control < 3).sum(dim="n").sel(control_axis=1).sum(dim="frame")
            leak_counts = (ds_qp.state_control == 2).sum(dim="n").sel(control_axis=1).sum(dim="frame")
            leakage = leak_counts / all_counts

        # Store fit results
        fit_results[qp.name] = CzConditionalPhaseFit(
            optimal_amplitude=optimal_amp,
            phase_diff=phase_diff,
            fitted_curve=fitted_curve,
            leakage=leakage,
            success=success,
        )

    return fit_results, optimal_amps
