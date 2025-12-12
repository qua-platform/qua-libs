import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    success: bool
    interaction_max: int
    coupler_flux_pulse: float
    coupler_flux_min: float
    flux_at_target: float


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit xarray Dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the fitted results for all qubits.
    log_callable : callable, optional
        Callable for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for qubit_pair, results in fit_results.items():
        log_callable(f"Results for qubit pair: {qubit_pair}")
        log_callable(f"  - Success: {results['success']}")
        log_callable(f"  - Interaction max: {results['interaction_max']} MHz")
        log_callable(f"  - Coupler flux pulse: {results['coupler_flux_pulse']:.3f} V")
        log_callable(f"  - Coupler flux min: {results['coupler_flux_min']:.3f} V")
        log_callable(f"  - Flux at target: {results['flux_at_target']:.3f} V")


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """
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
    qubit_pairs = [node.machine.qubit_pairs[pair] for pair in node.parameters.qubit_pairs]
    fluxes_coupler = ds.flux_coupler.values

    ds = ds.assign_coords(idle_time=ds.idle_time * 4)
    if node.parameters.use_state_discrimination:
        ds = ds.assign({"res_sum": ds.state_control - ds.state_target})
    else:
        ds = ds.assign({"res_sum": ds.I_control - ds.I_target})
    flux_coupler_full = np.array([fluxes_coupler + qp.coupler.decouple_offset for qp in qubit_pairs])
    ds = ds.assign_coords({"flux_coupler_full": (["qubit_pair", "flux_coupler"], flux_coupler_full)})

    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the qubit frequency and FWHM for each qubit in the dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node_parameters : Parameters
        Parameters related to the node, including whether state discrimination is used.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    """

    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    fit_results = {}
    for i, qp_name in enumerate(fit.qubit_pair.values):
        ds_qp = fit.isel(qubit_pair=i)
        time_us = ds_qp.idle_time.values * 1e-3
        flux_bias = ds_qp.flux_coupler.values
        data_matrix = ds_qp.state_target.values.T

        jeff_raw, fit_mask = extract_jeff_from_flux_slices(data_matrix, time_us, node)
        jeff_smooth = savgol_filter(jeff_raw, window_length=9, polyorder=3)

        fit = fit.assign(
            {
                f"jeff_raw_{qp_name}": (("flux_coupler",), jeff_raw),
                f"fit_mask_{qp_name}": (("flux_coupler",), fit_mask),
                f"jeff_smooth_{qp_name}": (("flux_coupler",), jeff_smooth),
            }
        )

        interaction_max = np.max(jeff_smooth)
        coupler_flux_pulse = flux_bias[np.argmax(jeff_smooth)]
        coupler_flux_min = flux_bias[np.argmin(jeff_smooth)]

        target_value = 1e3 / (node.parameters.target_gate_duration_ns)  # MHz
        closest_idx = np.argmin(np.abs(jeff_smooth - target_value))
        flux_at_target = flux_bias[closest_idx]

        fit_results[qp_name] = FitParameters(
            success=True,
            interaction_max=interaction_max,
            coupler_flux_pulse=coupler_flux_pulse,
            coupler_flux_min=coupler_flux_min,
            flux_at_target=flux_at_target,
        )

    return fit, fit_results


def damped_cosine(t, A, gamma, f, phi, C):
    return A * np.exp(-gamma * t) * np.cos(2 * np.pi * f * t + phi) + C


def extract_jeff_from_flux_slices(data_matrix, time_us, node: QualibrationNode):
    jeff_raw = []
    fit_mask = []

    for i in range(data_matrix.shape[1]):
        ydata = data_matrix[:, i]
        # condition = np.min(np.abs(ydata)) > 0.5
        # if node.parameters.cz_or_iswap == "iswap":
        #     condition = np.max(np.abs(ydata)) < 0.5
        # if condition:
        #     jeff_raw.append(0.0)
        #     fit_mask.append(False)
        #     continue

        try:
            popt, _ = curve_fit(
                damped_cosine,
                time_us,
                ydata,
                p0=[0.3, 1.0, 5.0, 0.0, 0.0],
                bounds=([0, 0, 0, -np.pi, -1], [1, 100, 20, np.pi, 1]),
                maxfev=5000,
            )
            freq_mhz = popt[2]
            jeff_raw.append(freq_mhz)
            fit_mask.append(True)
        except RuntimeError:
            jeff_raw.append(0.0)
            fit_mask.append(False)

    return np.array(jeff_raw), np.array(fit_mask)
