import logging
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import fit_oscillation, oscillation
from scipy.ndimage import gaussian_filter1d
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
    """Fit oscillations and derive optimal amplitude per qubit pair.

    Returns dataset with added data variables:
      - fitted, phase_diff (per number_of_operations) from oscillation fits
      - optimal_amplitude (per amp dimension collapsed)
      - success (bool)
    Also returns dict of FitResults referenced by qubit pair name.
    """
    ds_fit = ds.groupby("qubit_pair").apply(fit_routine)

    # Derive optimal amplitude per qubit pair using robust column cost over number_of_operations.
    opt_amps = []
    successes = []
    qp_names = ds_fit.qubit_pair.values
    for qp in qp_names:
        sub = ds_fit.sel(qubit_pair=qp)
        # phase_diff dims: number_of_operations, amp (no frame after fitting)
        if "phase_diff" not in sub:
            opt_amps.append(np.nan)
            successes.append(False)
            continue
        phase = sub.phase_diff  # already reduced over frame by fit_routine
        try:
            amp_coord = sub.amp_full if "amp_full" in sub.coords else sub.amp
            X = amp_coord.values
            # Ensure (ny, nx) ordering (number_of_operations, amp)
            Z = phase.transpose("number_of_operations", "amp").values
            x_star = _fit_full_amp(X, Z)
            opt_amps.append(x_star)
            successes.append(bool(np.isfinite(x_star)))
        except Exception:
            opt_amps.append(np.nan)
            successes.append(False)

    ds_fit = ds_fit.assign_coords({"optimal_amplitude": ("qubit_pair", np.array(opt_amps))})
    ds_fit["optimal_amplitude"] = ds_fit["optimal_amplitude"].astype(float)
    ds_fit = ds_fit.assign_coords({"success": ("qubit_pair", np.array(successes, dtype=bool))})

    # Build FitResults dict
    fit_results: Dict[str, FitResults] = {}
    for qp, amp, succ in zip(qp_names, opt_amps, successes):
        fit_results[str(qp)] = FitResults(optimal_amplitude=float(amp), success=bool(succ))

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

    return da


# -------------------- Optimal amplitude helper functions -------------------- #


def _circ_dist_to_half(Z):
    """Circular distance of values Z in [0,1) to 0.5 expressed in [0,0.5]."""
    return np.abs(((Z - 0.5 + 0.5) % 1.0) - 0.5)


def _fit_full_amp(X, Z, row_mask=None, trim=0.2, smooth_rows_sigma=0.6, smooth_cols_sigma=1.0):
    """Robustly select single amplitude minimizing distance of phase to 0.5 across repetitions.

    Parameters mirror the exploratory implementation embedded previously in node file.
    X: (nx,) amplitude array
    Z: (ny, nx) phase_diff array in [0,1)
    Returns best amplitude.
    """
    Zw = Z.copy()
    if smooth_rows_sigma and smooth_rows_sigma > 0:
        Zw = gaussian_filter1d(Zw, sigma=smooth_rows_sigma, axis=0, mode="nearest")
    if row_mask is None:
        row_mask = np.ones(Zw.shape[0], dtype=bool)
    D = _circ_dist_to_half(Zw[row_mask, :])
    n = D.shape[0]
    if n == 0:
        return np.nan
    k = int(np.floor(trim * n))
    D_sorted = np.sort(D, axis=0)
    if 2 * k < n:
        C = D_sorted[k : n - k, :].mean(axis=0)
    else:
        C = D_sorted.mean(axis=0)
    if smooth_cols_sigma and smooth_cols_sigma > 0:
        C = gaussian_filter1d(C, sigma=smooth_cols_sigma, axis=0, mode="nearest")
    j0 = int(np.argmin(C))
    j_star = float(j0)
    if 0 < j0 < len(X) - 1:
        y1, y2, y3 = C[j0 - 1], C[j0], C[j0 + 1]
        denom = y1 - 2 * y2 + y3
        if denom != 0:
            delta = 0.5 * (y1 - y3) / denom
            j_star = j0 + np.clip(delta, -0.5, 0.5)
    return float(np.interp(j_star, np.arange(len(X)), X))


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
