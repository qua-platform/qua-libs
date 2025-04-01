import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.qua_datasets import add_amplitude_and_phase, convert_IQ_to_V
from quam_experiments.analysis.fit_utils import peaks_dips


@dataclass
class FitParameters:
    """Stores the relevant node-specific fitted parameters used to update the state at the end of the node."""

    success: bool
    resonator_frequency: float
    frequency_shift: float
    optimal_power: float


def log_fitted_results(fit_results: Dict, logger=None):
    """
    Logs the node-specific fitted results for all qubits from the fit xarray Dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the fitted results for all qubits.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.

    Returns:
    --------
    None

    Example:
    --------
        >>> log_fitted_results(fit_results)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_power = f"Optimal readout power: {fit_results[q]['optimal_power']:.2f} dBm | "
        s_freq = f"Resonator frequency: {1e-9 * fit_results[q]['resonator_frequency']:.3f} GHz | "
        s_shift = f"(shift of {1e-6 * fit_results[q]['frequency_shift']:.0f} MHz)\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        logger.info(s_qubit + s_power + s_freq + s_shift)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Processes the raw dataset by converting the 'I' and 'Q' quadratures to V, or adding the RF_frequency as a coordinate for instance."""

    # Convert the 'I' and 'Q' quadratures from demodulation units to V.
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    # Add the amplitude and phase to the raw dataset
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    # Add the RF frequency as a coordinate of the raw dataset
    full_freq = np.array([ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    # Normalize the IQ_abs with respect to the amplitude axis
    ds = ds.assign({"IQ_abs_norm": ds["IQ_abs"] / ds.IQ_abs.mean(dim=["detuning"])})
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the T1 relaxation time for each qubit according to ``a * np.exp(t * decay) + offset``.

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

    ds_fit = ds
    # Generate 1D dataset tracking the minimum IQ value, as a proxy for resonator frequency
    ds_fit["rr_min_response"] = ds.IQ_abs_norm.idxmin(dim="detuning")
    # Calculate the derivative along the power axis
    ds_fit["rr_min_response_diff"] = ds_fit.rr_min_response.differentiate(coord="power").dropna("power")
    # Calculate the moving average of the derivative
    ds_fit["rr_min_response_diff_avg"] = (
        ds.rr_min_response_diff.rolling(
            power=node.parameters.derivative_smoothing_window_num_points,
            center=True,  # window size in points
        )
        .mean()
        .dropna("power")
    )
    # Apply a filter to scale down the initial noisy values in the moving average if needed
    for j in range(node.parameters.moving_average_filter_window_num_points):
        ds_fit.rr_min_response_diff_avg.isel(power=j).data /= (
            node.parameters.moving_average_filter_window_num_points - j
        )
    # Find the first position where the moving average crosses below the threshold
    ds_fit["below_threshold"] = (
        ds_fit.rr_min_response_diff_avg < node.parameters.derivative_crossing_threshold_in_hz_per_dbm
    )
    # Get the first occurrence below the derivative threshold
    optimal_power = ds_fit.below_threshold.idxmax(dim="power")
    optimal_power -= node.parameters.buffer_from_crossing_threshold_in_dbm
    ds_fit = ds_fit.assign_coords({"optimal_power": (["qubit"], optimal_power.data)})

    # Define a function to fit the resonator line at the optimal power for each qubit
    def _select_optimal_power(ds, qubit):
        return peaks_dips(
            ds.sel(power=ds["optimal_power"].sel(qubit=qubit).data, method="nearest").sel(qubit=qubit).IQ_abs,
            "detuning",
        )

    # Get the resonance frequency shift at the optimal power
    freq_shift = []
    for q in node.namespace["qubits"]:
        freq_shift.append(float(_select_optimal_power(ds_fit, q.name).position.data))
    ds_fit = ds_fit.assign_coords({"freq_shift": (["qubit"], freq_shift)})

    # Extract the relevant fitted parameters
    fit_dataset, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_dataset, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the fit dataset and fit result dictionary."""

    # Get the fitted resonator frequency
    full_freq = np.array([q.resonator.RF_frequency for q in node.namespace["qubits"]])
    res_freq = fit.freq_shift + full_freq
    fit = fit.assign_coords(res_freq=("qubit", res_freq.data))
    fit.res_freq.attrs = {"long_name": "resonator frequency", "units": "Hz"}
    # Assess whether the fit was successful or not
    freq_success = np.abs(fit.freq_shift.data) < node.parameters.frequency_span_in_mhz * 1e6
    nan_success = np.isnan(fit.freq_shift.data) | np.isnan(fit.optimal_power.data)
    success_criteria = freq_success & ~nan_success
    fit = fit.assign_coords(success=("qubit", success_criteria))

    fit_results = {
        q: FitParameters(
            success=fit.sel(qubit=q).success.values.__bool__(),
            resonator_frequency=float(fit.res_freq.sel(qubit=q).values),
            frequency_shift=float(fit.freq_shift.sel(qubit=q).data),
            optimal_power=float(fit.optimal_power.sel(qubit=q).data),
        )
        for q in fit.qubit.values
    }

    return fit, fit_results
