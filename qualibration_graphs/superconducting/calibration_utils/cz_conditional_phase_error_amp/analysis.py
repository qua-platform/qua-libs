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
    """Extract and fix the phase parameter from oscillation fit data."""
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

    def abs_amp(qp, amp):
        return amp * qp.macros["cz_unipolar"].flux_pulse_control.amplitude

    def detuning(qp, amp):
        amplitude_squared = (amp * qp.macros["cz_unipolar"].flux_pulse_control.amplitude) ** 2
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
    """Fit oscillations per number_of_operations and aggregate results.

    For each number_of_operations value:
        - Fit oscillation of the selected signal.
        - Store fitted oscillatory curve.
        - Compute and store phase difference between control_axis 0 and 1 (normalized 0..1 for 0..2π).
    Returns the original dataset with two new data variables: 'fitted' and 'phase_diff'.
    """

    data_var = "state_target" if "state_target" in da else "I_target"
    nops_vals = da.number_of_operations.values

    fitted_list = []
    phase_diff_list = []

    for nops in nops_vals:
        da_sel = da.sel(number_of_operations=nops)
        fit_data = fit_oscillation(da_sel[data_var], "frame")

        fitted_curve = (
            oscillation(
                da_sel.frame,
                fit_data.sel(fit_vals="a"),
                fit_data.sel(fit_vals="f"),
                fit_data.sel(fit_vals="phi"),
                fit_data.sel(fit_vals="offset"),
            )
            .rename("fitted")
            .expand_dims(number_of_operations=[nops])
        )
        fitted_list.append(fitted_curve)

        phase = fix_oscillation_phi_2pi(fit_data)
        phase_diff = (
            ((phase.sel(control_axis=0) - phase.sel(control_axis=1)) % 1)
            .rename("phase_diff")
            .expand_dims(number_of_operations=[nops])
        )
        phase_diff_list.append(phase_diff)

    if fitted_list:
        fitted_all = xr.concat(fitted_list, dim="number_of_operations")
    else:
        fitted_all = None
    if phase_diff_list:
        phase_diff_all = xr.concat(phase_diff_list, dim="number_of_operations")
    else:
        phase_diff_all = None

    to_assign = {}
    if fitted_all is not None:
        to_assign["fitted"] = fitted_all
    if phase_diff_all is not None:
        to_assign["phase_diff"] = phase_diff_all

    if to_assign:
        da = da.assign(to_assign)

    # # --- Post-processing adjustment: add 0.5 to all numeric vars at odd number_of_operations ---
    # dim_name = "number_of_operations"
    # if dim_name in da.dims or dim_name in da.coords:
    #     try:
    #         ops = da[dim_name]
    #         # number_of_operations are 1-based; select odd labels (1,3,5,...)
    #         odd_ops = ops.where((ops % 2) == 0, drop=True).values
    #         if len(odd_ops) > 0:
    #             for var_name, dvar in da.data_vars.items():
    #                 if dim_name in dvar.dims and np.issubdtype(dvar.dtype, np.number):
    #                     da[var_name].loc[{dim_name: odd_ops}] = -(da[var_name].loc[{dim_name: odd_ops}] - 0.5)
    #     except Exception as e:
    #         # Silent fail (could optionally log); keep dataset untouched if error
    #         pass

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
