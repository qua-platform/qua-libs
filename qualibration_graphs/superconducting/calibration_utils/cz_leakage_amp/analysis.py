"""Analysis module for CZ leakage amplification calibration."""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from scipy.ndimage import gaussian_filter1d


@dataclass
class FitResults:
    """Stores the relevant CZ leakage amplification experiment fit parameters for a single qubit pair."""

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
        s_amp = f"\tOptimal CZ coupler amplitude: {fit_result.optimal_amplitude:.6f} a.u."

        if fit_result.success:
            s_qubit += "SUCCESS!\n"
        else:
            s_qubit += "FAIL!\n"

        log_message = s_qubit + s_amp
        log_callable(log_message)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """
    Process the raw dataset by adding amplitude coordinates and pair-combined state variables.

    If the dataset has only state_moving_stationary1, state_moving_stationary2, ... they are concatenated
    along qubit_pair to form state_moving_stationary(qubit_pair, amp, number_of_operations).

    Parameters:
    -----------
    ds : xr.Dataset
        Raw dataset from the experiment
    node : QualibrationNode
        The calibration node containing qubit pairs information

    Returns:
    --------
    xr.Dataset
        Processed dataset with additional coordinates and combined state variables.
    """
    qubit_pairs = node.namespace["qubit_pairs"]
    operation = node.parameters.operation

    def abs_amp(qp, amp):
        return amp * qp.macros[operation].coupler_flux_pulse.amplitude

    ds = ds.assign_coords({"amp_full": (["qubit_pair", "amp"], np.array([abs_amp(qp, ds.amp) for qp in qubit_pairs]))})

    for base_var in (
        "state_moving",
        "state_stationary",
        "state_moving_stationary",
    ):
        if base_var in ds.data_vars:
            continue

        pair_vars = [
            var
            for var in ds.data_vars
            if var.startswith(base_var) and (var == base_var or var.replace(base_var, "").isdigit())
        ]
        if not pair_vars:
            continue

        pair_vars = sorted(pair_vars, key=lambda name: int(name.replace(base_var, "") or 0))
        state_list = [ds[var] for var in pair_vars]
        pair_names = np.array(qubit_pairs.get_names())[: len(state_list)]
        concatenated = xr.concat(state_list, dim="qubit_pair").assign_coords(qubit_pair=pair_names)
        ds = ds.assign({base_var: concatenated})

    return ds


def _optimal_amp_from_mean(X: np.ndarray, mean_vals: np.ndarray, smooth_sigma: float = 1.0) -> Tuple[float, int]:
    """Compute optimal amplitude as argmax of mean (optionally smoothed), with parabolic refinement.

    Returns (optimal_amplitude_value, optimal_index).
    """
    if smooth_sigma and smooth_sigma > 0:
        mean_vals = gaussian_filter1d(mean_vals.astype(float), sigma=smooth_sigma, mode="nearest")
    j0 = int(np.nanargmax(mean_vals))
    j_star = float(j0)
    n = len(X)
    if 0 < j0 < n - 1:
        y1, y2, y3 = mean_vals[j0 - 1], mean_vals[j0], mean_vals[j0 + 1]
        denom = y1 - 2 * y2 + y3
        if denom and np.isfinite(denom):
            delta = 0.5 * (y1 - y3) / denom
            j_star = j0 + np.clip(delta, -0.5, 0.5)
    x_star = float(np.interp(j_star, np.arange(n), X))
    return x_star, j0


def fit_routine(da: xr.Dataset) -> xr.Dataset:
    """Compute mean P(11) over number_of_operations for each amplitude.

    Adds mean_state_moving_stationary(amp). No oscillation fit.
    """
    data_var = "state_moving_stationary" if "state_moving_stationary" in da else None
    if data_var is None:
        return da
    arr = da[data_var]
    if "number_of_operations" not in arr.dims:
        return da
    mean_vals = arr.mean(dim="number_of_operations").rename("mean_state_moving_stationary")
    return da.assign(mean_state_moving_stationary=mean_vals)


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitResults]]:
    """Compute mean over number_of_operations per amp and derive optimal amplitude (argmax mean) per qubit pair.

    Returns dataset with added mean state variables and coordinates
    optimal_amplitude, optimal_index, success per qubit_pair. Also returns dict of FitResults.
    """
    ds_fit = ds.groupby("qubit_pair").apply(fit_routine)

    opt_amps = []
    opt_idxs = []
    successes = []
    qp_names = ds_fit.qubit_pair.values

    for qp in qp_names:
        sub = ds_fit.sel(qubit_pair=qp)
        mean_var = "mean_state_moving_stationary" if "mean_state_moving_stationary" in sub else None
        if mean_var is None:
            # Fallback: compute mean from combined P(11) state data.
            if "state_moving_stationary" in sub:
                mean_vals = sub.state_moving_stationary.mean(dim="number_of_operations")
                sub = sub.assign(mean_state_moving_stationary=mean_vals)
                mean_var = "mean_state_moving_stationary"
            else:
                opt_amps.append(np.nan)
                opt_idxs.append(0)
                successes.append(False)
                continue
        try:
            amp_coord = sub.amp_full if "amp_full" in sub.coords else sub.amp
            X = np.asarray(amp_coord.values)
            mean_arr = sub[mean_var].values
            x_star, idx = _optimal_amp_from_mean(X, mean_arr, smooth_sigma=1.0)
            amp_min, amp_max = float(np.min(X)), float(np.max(X))
            if not np.isfinite(x_star) or not (amp_min <= x_star <= amp_max):
                opt_amps.append(np.nan)
                opt_idxs.append(idx)
                successes.append(False)
            else:
                opt_amps.append(x_star)
                opt_idxs.append(idx)
                successes.append(True)
        except Exception:
            opt_amps.append(np.nan)
            opt_idxs.append(0)
            successes.append(False)

    ds_fit = ds_fit.assign_coords({"optimal_amplitude": ("qubit_pair", np.array(opt_amps))})
    ds_fit["optimal_amplitude"] = ds_fit["optimal_amplitude"].astype(float)
    ds_fit = ds_fit.assign_coords({"optimal_index": ("qubit_pair", np.array(opt_idxs))})
    ds_fit["optimal_index"] = ds_fit["optimal_index"].astype(int)
    ds_fit = ds_fit.assign_coords({"success": ("qubit_pair", np.array(successes, dtype=bool))})

    fit_results: Dict[str, FitResults] = {}
    for qp, amp, succ in zip(qp_names, opt_amps, successes):
        fit_results[str(qp)] = FitResults(optimal_amplitude=float(amp), success=bool(succ))

    return ds_fit, fit_results
