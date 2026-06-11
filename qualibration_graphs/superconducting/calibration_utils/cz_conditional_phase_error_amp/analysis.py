import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import fit_oscillation, oscillation
from scipy.ndimage import gaussian_filter1d

from calibration_utils.cz_conditional_phase.analysis import fix_oscillation_phi_2pi

@dataclass
class FitResults:
    """Fit results for a single qubit pair from the CZ conditional phase (error amplification) calibration.

    Attributes:
    -----------
    optimal_amplitude : float
        Flux pulse amplitude (V) at which the conditional phase equals π, selected
        by minimising the trimmed-mean phase distance across all repetition counts.
    success : bool
        True if a finite optimal amplitude was found within the swept range.
    """

    optimal_amplitude: float
    success: bool


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
        return amp * qp.macros[operation].flux_pulse_qubit.amplitude

    def detuning(qp, amp):
        amplitude_squared = (amp * qp.macros[operation].flux_pulse_qubit.amplitude) ** 2
        return -amplitude_squared * node.namespace["qubit_roles_map"][qp.name].moving.freq_vs_flux_01_quad_term

    ds = ds.assign_coords({"amp_full": (["qubit_pair", "amp"], np.array([abs_amp(qp, ds.amp) for qp in qubit_pairs]))})
    ds = ds.assign_coords({"detuning": (["qubit_pair", "amp"], np.array([detuning(qp, ds.amp) for qp in qubit_pairs]))})

    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitResults]]:
    """
    Fit frame-rotation oscillations and extract the optimal CZ amplitude per qubit pair.

    Parameters:
    -----------
    ds : xr.Dataset
        Processed dataset (output of ``process_raw_dataset``) containing
        ``state_stationary`` or ``I_stationary``, ``frame``, ``amp_full``,
        ``number_of_operations``, and ``control_axis`` dimensions.
    node : QualibrationNode
        Calibration node (used by ``_extract_relevant_parameters``).

    Returns:
    --------
    Tuple[xr.Dataset, Dict[str, FitResults]]
        Dataset with ``phase_diff``, ``fitted``, ``optimal_amplitude``, and
        ``success`` added, and a dictionary of ``FitResults`` keyed by qubit pair name.
    """
    ds_fit = ds.groupby("qubit_pair").apply(fit_routine)
    # Extract the relevant fitted parameters
    ds_fit, fit_results = _extract_relevant_parameters(ds_fit, node)
    return ds_fit, fit_results


def _extract_relevant_parameters(
    ds_fit: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitResults]]:
    """
    Derive the optimal CZ amplitude from the error-amplified phase-diff and build FitResults.

    For each qubit pair the ``phase_diff`` 2D array (number_of_operations × amplitude) is
    passed to ``_fit_full_amp``, which selects the amplitude column that minimises the
    trimmed-mean circular distance to 0.5 (π) across all repetition counts.

    Parameters:
    -----------
    ds_fit : xr.Dataset
        Dataset produced by ``fit_routine`` containing ``phase_diff``
        (dims: ``number_of_operations``, ``amp``) and ``amp_full`` coordinates.
    node : QualibrationNode
        Unused; retained for API consistency.

    Returns:
    --------
    Tuple[xr.Dataset, Dict[str, FitResults]]
        Dataset with ``optimal_amplitude``, ``optimal_index``, and ``success``
        coordinates added, and a dictionary of ``FitResults`` keyed by qubit pair name.
    """
    qp_names = ds_fit.qubit_pair.values
    opt_amps = []
    opt_idxs = []
    successes = []

    for qp in qp_names:
        sub = ds_fit.sel(qubit_pair=qp)
        if "phase_diff" not in sub:
            opt_amps.append(np.nan)
            opt_idxs.append(np.nan)
            successes.append(False)
            continue
        phase = sub.phase_diff
        try:
            amp_coord = sub.amp_full if "amp_full" in sub.coords else sub.amp
            X = amp_coord.values
            Z = phase.transpose("number_of_operations", "amp").values
            x_star = _fit_full_amp(X, Z)
            idx = int(np.argmin(np.abs(X - x_star)))
            opt_amps.append(x_star)
            opt_idxs.append(idx)
            successes.append(bool(np.isfinite(x_star)))
        except Exception:
            opt_amps.append(np.nan)
            opt_idxs.append(np.nan)
            successes.append(False)

    ds_fit = ds_fit.assign_coords({"optimal_amplitude": ("qubit_pair", np.array(opt_amps))})
    ds_fit["optimal_amplitude"] = ds_fit["optimal_amplitude"].astype(float)
    ds_fit["optimal_amplitude"].attrs = {"long_name": "optimal CZ amplitude", "units": "a.u."}
    ds_fit = ds_fit.assign_coords({"optimal_index": ("qubit_pair", np.array(opt_idxs))})
    ds_fit["optimal_index"] = ds_fit["optimal_index"].astype(int)
    ds_fit = ds_fit.assign_coords({"success": ("qubit_pair", np.array(successes, dtype=bool))})
    if "phase_diff" in ds_fit.data_vars:
        ds_fit.phase_diff.attrs = {"long_name": "phase difference", "units": "2π"}

    fit_results: Dict[str, FitResults] = {}
    for qp, amp, succ in zip(qp_names, opt_amps, successes):
        fit_results[str(qp)] = FitResults(optimal_amplitude=float(amp), success=bool(succ))

    return ds_fit, fit_results


def fit_routine(da):
    """Fit frame-rotation oscillations for every repetition count and aggregate.

    For each ``number_of_operations`` value:
    - Fits a sinusoidal oscillation over the ``frame`` axis for both moving-qubit
      states (``control_axis`` 0 and 1).
    - Stores the fitted oscillatory curve.
    - Computes the conditional phase difference (normalised to [0, 1) for 0 to 2π).

    Parameters:
    -----------
    da : xr.Dataset
        Single-pair dataset with ``e_state_stationary`` or ``I_stationary``, ``frame``,
        ``number_of_operations``, ``amp_full``, and ``control_axis`` dimensions.

    Returns:
    --------
    xr.Dataset
        Input dataset extended with ``fitted`` and ``phase_diff`` data variables.
    """

    data_var = "e_state_stationary" if "e_state_stationary" in da else "I_stationary"
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
    """Robustly select the single amplitude that minimises the phase distance to 0.5 (π).

    Applies optional Gaussian smoothing along the repetition axis, trims outlier rows
    via a trimmed mean, then finds the amplitude column with the smallest mean circular
    distance to 0.5.  Sub-pixel refinement via parabolic interpolation is applied when
    the optimum is not on a boundary.

    Parameters:
    -----------
    X : np.ndarray
        1D array of amplitude values, shape (nx,).
    Z : np.ndarray
        2D phase-diff array in [0, 1), shape (ny, nx) where ny = number_of_operations.
    row_mask : np.ndarray or None
        Boolean mask selecting which rows (repetitions) to include.
        If None, all rows are used.
    trim : float
        Fraction of rows to trim from each end of the sorted cost distribution (default 0.2).
    smooth_rows_sigma : float
        Gaussian smoothing sigma along the repetition axis before trimming (default 0.6).
    smooth_cols_sigma : float
        Gaussian smoothing sigma along the amplitude axis after trimming (default 1.0).

    Returns:
    --------
    float
        Best-estimate amplitude value (may be sub-pixel interpolated).  Returns ``np.nan``
        if no rows survive masking/trimming.
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
