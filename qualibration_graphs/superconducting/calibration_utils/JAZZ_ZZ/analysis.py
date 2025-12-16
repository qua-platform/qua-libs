import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import fit_oscillation, oscillation
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


@dataclass
class FitResults:
    """Stores the relevant JAZZ_ZZ experiment fit parameters for a single qubit pair"""

    optimal_amplitude: float
    success: bool


def damped_cosine(t, A, gamma, f, phi, C):
    """Damped cosine fitting function for JAZZ_ZZ oscillations."""
    return A * np.exp(-gamma * t) * np.cos(2 * np.pi * f * t + phi) + C


def log_fitted_results(fit_results: Dict[str, FitResults], log_callable=None):
    """
    Logs the JAZZ_ZZ fitted results for all qubit pairs.

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
        s_amp = f"\tOptimal JAZZ_ZZ amplitude for minimum coupling: {fit_result.optimal_amplitude:.6f} a.u."

        if fit_result.success:
            s_qubit += "SUCCESS!\n"
        else:
            s_qubit += "FAIL!\n"

        log_message = s_qubit + s_amp

        log_callable(log_message)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """
    Process the raw dataset for JAZZ_ZZ analysis.

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
    # Convert time from ns to µs for fitting
    time_us = ds.time.data * 1e-3
    ds = ds.assign_coords(time_us=("time", time_us))

    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitResults]]:
    """
    Fit the JAZZ_ZZ data by extracting effective coupling J_eff from oscillations.

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
    ds_fit = ds.groupby("qubit_pair").apply(lambda da: fit_jazz_zz_routine(da, node))

    # Extract the relevant fitted parameters
    ds_fit, fit_results = _extract_relevant_parameters(ds_fit, node)

    return ds_fit, fit_results


def fit_jazz_zz_routine(da, node):
    """
    Extract effective coupling J_eff from JAZZ_ZZ oscillations for each amplitude.

    Parameters:
    -----------
    da : xr.DataArray
        Data array containing the oscillation data
    node : QualibrationNode
        The calibration node containing parameters

    Returns:
    --------
    xr.DataArray
        Data array with added fit results
    """
    if hasattr(da, "state_target"):
        data = "state_target"
    else:
        data = "I_target"

    # Extract the data matrix (time vs amplitude)
    data_matrix = da[data].data[0].T  # shape = (n_time, n_amp)
    flux_bias = da.amp.data  # amplitude values
    time_us = da.time_us.data  # time in µs

    # Extract J_eff from each flux slice
    jeff_raw = []
    fit_mask = []

    for i in range(data_matrix.shape[1]):
        ydata = data_matrix[:, i] - np.mean(data_matrix[:, i])

        try:
            popt, _ = curve_fit(
                damped_cosine,
                time_us,
                ydata,
                p0=[0.3, 1.0, 5.0, 0.0, 0.0],
                bounds=([-np.inf, -np.inf, -np.inf, -np.pi, -np.inf], [np.inf, np.inf, np.inf, np.pi, np.inf]),
                maxfev=5000,
            )
            freq_mhz = popt[2]
            jeff_raw.append(freq_mhz)
            fit_mask.append(True)
        except RuntimeError:
            jeff_raw.append(0.0)
            fit_mask.append(False)

    jeff_raw = np.array(jeff_raw)
    fit_mask = np.array(fit_mask)

    # Smooth only the valid (nonzero) portion if there are enough valid points
    if np.sum(fit_mask) >= 9:  # Need at least 9 points for window_length=9
        jeff_smooth = np.zeros_like(jeff_raw)
        jeff_smooth[fit_mask] = savgol_filter(jeff_raw[fit_mask], window_length=9, polyorder=3)
    else:
        jeff_smooth = jeff_raw.copy()

    # Find optimal amplitude (closest to artificial_detuning_mhz to minimize coupling)
    valid_indices = np.where(fit_mask)[0]
    if len(valid_indices) > 0:
        coupling_deviation = np.abs(jeff_smooth[fit_mask] - node.parameters.artificial_detuning_mhz)
        min_coupling_idx = valid_indices[np.argmin(coupling_deviation)]
        optimal_amplitude = flux_bias[min_coupling_idx]
        success = True
    else:
        optimal_amplitude = np.nan
        success = False

    # Add results to data array
    da = da.assign(
        jeff_raw=("amp", jeff_raw),
        jeff_smooth=("amp", jeff_smooth),
        fit_mask=("amp", fit_mask),
        optimal_amplitude=optimal_amplitude,
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
        Dataset containing the fit results from fit_jazz_zz_routine.
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
        ds_fit.optimal_amplitude.attrs = {
            "long_name": "optimal JAZZ_ZZ amplitude for minimum coupling",
            "units": "a.u.",
        }
    if "jeff_raw" in ds_fit.data_vars:
        ds_fit.jeff_raw.attrs = {"long_name": "raw extracted effective coupling", "units": "MHz"}
    if "jeff_smooth" in ds_fit.data_vars:
        ds_fit.jeff_smooth.attrs = {"long_name": "smoothed effective coupling", "units": "MHz"}
    if "fit_mask" in ds_fit.data_vars:
        ds_fit.fit_mask.attrs = {"long_name": "successful fit mask", "units": "bool"}

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
