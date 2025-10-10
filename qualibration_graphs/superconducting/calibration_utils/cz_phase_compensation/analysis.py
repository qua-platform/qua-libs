import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import fit_oscillation, oscillation
from qualibration_libs.data import convert_IQ_to_V
from scipy.optimize import curve_fit


@dataclass
class FitResults:
    """Stores the relevant CZ conditional phase experiment fit parameters for a single qubit pair"""

    control_phase_correction: float
    target_phase_correction: float
    success: bool


def fix_oscillation_phi_2pi(fit_data):
    """Extract and fix the phase parameter from oscillation fit data."""
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
        s_control = f"\tControl qubit phase correction: {fit_results[qp_name]["control_phase_correction"]:.6f} a.u."
        s_target = f"\tTarget qubit phase correction: {fit_results[qp_name]["target_phase_correction"]:.6f} a.u."

        if fit_results[qp_name]["success"]:
            s_qubit += "SUCCESS!\n"
        else:
            s_qubit += "FAIL!\n"

        log_message = s_qubit + s_control + s_target

        log_callable(log_message)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    # Convert IQ data into volts
    if hasattr(ds, "I_control"):
        ds = convert_IQ_to_V(
            ds, qubit_pairs=node.namespace["qubit_pairs"], IQ_list=["I_control", "Q_control", "I_target", "Q_target"]
        )
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
        data_control = "state_control"
        data_target = "state_target"
    else:
        data_control = "I_control"
        data_target = "I_target"

    try:
        # Fit oscillation for each control state and amplitude
        if data_control == "state_control":
            fit_control = fit_oscillation(da[data_control], "frame")
            fit_target = fit_oscillation(da[data_target], "frame")
        else:
            fit_control = fit_oscillation(da[data_control].sel(control_target="c"), "frame")
            fit_target = fit_oscillation(da[data_target].sel(control_target="t"), "frame")
        # Add fitted oscillation curves to the dataset
        da = da.assign(
            {
                "fitted_control": oscillation(
                    da.frame,
                    fit_control.sel(fit_vals="a"),
                    fit_control.sel(fit_vals="f"),
                    fit_control.sel(fit_vals="phi"),
                    fit_control.sel(fit_vals="offset"),
                )
            }
        )

        da = da.assign(
            {
                "fitted_target": oscillation(
                    da.frame,
                    fit_target.sel(fit_vals="a"),
                    fit_target.sel(fit_vals="f"),
                    fit_target.sel(fit_vals="phi"),
                    fit_target.sel(fit_vals="offset"),
                )
            }
        )

        # Extract phase and calculate phase difference
        phase_control = fix_oscillation_phi_2pi(fit_control)
        phase_target = fix_oscillation_phi_2pi(fit_target)

        da = da.assign(
            {
                "fitted_control_phase": phase_control,
            }
        )
        da = da.assign(
            {
                "fitted_target_phase": phase_target,
            }
        )

        da = da.assign(
            {
                "success": True,
            }
        )
    except Exception as e:
        print(f"Error fitting data: {e}")

        da = da.assign(
            {
                "success": False,
            }
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

    # Create FitResults for each qubit pair
    fit_results = {}
    for qp in qubit_pairs:
        qp_name = qp.name
        qp_data = ds_fit.sel(qubit_pair=qp_name)
        if qp_data.success.values:
            fit_results[qp_name] = FitResults(
                control_phase_correction=qp_data.fitted_control_phase.values,
                target_phase_correction=qp_data.fitted_target_phase.values,
                success=True,
            )
        else:
            fit_results[qp_name] = FitResults(
                control_phase_correction=np.nan,
                target_phase_correction=np.nan,
                success=False,
            )

    return ds_fit, fit_results
