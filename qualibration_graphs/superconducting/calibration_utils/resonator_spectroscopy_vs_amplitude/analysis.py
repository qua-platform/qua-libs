from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.stats
import xarray as xr
from qualibration_libs.analysis import peaks_dips
from qualibration_libs.analysis.parameters import analysis_config_manager
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from scipy.signal import find_peaks, savgol_filter
from sklearn.cluster import KMeans

from qualibrate import QualibrationNode

# Access parameter groups from the config manager
qc_params = analysis_config_manager.get("resonator_spectroscopy_vs_power_qc")
sp_params = analysis_config_manager.get("common_signal_processing")


@dataclass
class FitParameters:
    """Stores the relevant node-specific fitted parameters used to update the state at the end of the node."""
    resonator_frequency: float
    frequency_shift: float
    optimal_power: float
    outcome: str


def log_fitted_results(fit_results: Dict[str, FitParameters], log_callable=None) -> None:
    """
    Logs the node-specific fitted results for all qubits from the fit results.

    Parameters
    ----------
    fit_results : Dict[str, FitParameters]
        Dictionary containing the fitted results for all qubits.
    log_callable : callable, optional
        Logger for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    
    for qubit_name, results in fit_results.items():
        status_line = f"Results for qubit {qubit_name}: "
        
        # Handle both FitParameters objects and dictionaries
        if hasattr(results, 'outcome'):
            outcome = results.outcome
            optimal_power = results.optimal_power
            resonator_frequency = results.resonator_frequency
            frequency_shift = results.frequency_shift
        else:
            outcome = results.get('outcome', 'unknown')
            optimal_power = results.get('optimal_power', 0.0)
            resonator_frequency = results.get('resonator_frequency', 0.0)
            frequency_shift = results.get('frequency_shift', 0.0)
        
        if outcome == "successful":
            status_line += " SUCCESS!\n"
        else:
            status_line += f" FAIL! Reason: {outcome}\n"
        
        # Format parameters with appropriate units
        power_str = f"Optimal readout power: {optimal_power:.2f} dBm | "
        freq_str = f"Resonator frequency: {resonator_frequency * 1e-9:.3f} GHz | "
        shift_str = f"(shift of {frequency_shift * 1e-6:.0f} MHz)\n"
        
        log_callable(status_line + power_str + freq_str + shift_str)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """
    Process raw dataset for resonator spectroscopy vs amplitude analysis.
    
    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset containing measurement data
    node : QualibrationNode
        The qualibration node containing parameters and qubit information
        
    Returns
    -------
    xr.Dataset
        Processed dataset with additional coordinates and derived quantities
    """
    # Convert I/Q quadratures to voltage
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    
    # Add amplitude and phase information
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    
    # Add RF frequency coordinate
    full_freq = np.array([
        ds.detuning + q.resonator.RF_frequency 
        for q in node.namespace["qubits"]
    ])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    
    # Normalize IQ_abs with respect to the amplitude axis
    ds = ds.assign({"IQ_abs_norm": ds["IQ_abs"] / ds.IQ_abs.mean(dim=["detuning"])})
    
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Fit resonator spectroscopy vs amplitude data for each qubit in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the processed data
    node : QualibrationNode
        The qualibration node containing experiment parameters

    Returns
    -------
    Tuple[xr.Dataset, Dict[str, FitParameters]]
        Tuple containing:
        - Dataset with fit results and quality metrics
        - Dictionary mapping qubit names to fit parameters
    """
    ds_fit = ds.copy()
    qubits = node.namespace["qubits"]
    
    # Track resonator response and analyze quality
    tracking_results = _track_resonator_response_across_power(ds, qubits)
    
    # Assign optimal power using clustering-based approach
    optimal_power_dict = _assign_optimal_power_clustering(ds, qubits)
    
    # Calculate fit metrics and quality assessments
    fit_metrics = _calculate_comprehensive_fit_metrics(ds, qubits, optimal_power_dict, tracking_results)
    
    # Add all calculated metrics to dataset
    ds_fit = _add_metrics_to_dataset(ds_fit, tracking_results, optimal_power_dict, fit_metrics)
    
    # Calculate frequency shifts at optimal power
    freq_shifts = _calculate_frequency_shifts_at_optimal_power(ds_fit, node)
    ds_fit = ds_fit.assign_coords({"freq_shift": (["qubit"], freq_shifts)})
    
    # Extract and validate fit parameters
    fit_dataset, fit_results = _extract_and_validate_amplitude_parameters(ds_fit, node, ds)
    
    return fit_dataset, fit_results


def _track_resonator_response_across_power(ds: xr.Dataset, qubits: list) -> Dict[str, np.ndarray]:
    """
    Track resonator response across power levels for all qubits.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing measurement data
    qubits : list
        List of qubit objects
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing tracking results for each qubit
    """
    tracking_results = {
        "rr_tracked": [],
        "rr_tracked_filtered": [], 
        "fit_quality_masks": []
    }
    
    for q in qubits:
        qubit_results = _track_single_qubit_response(ds, q.name)
        tracking_results["rr_tracked"].append(qubit_results["smoothed_centers"])
        tracking_results["rr_tracked_filtered"].append(qubit_results["filtered_centers"])
        tracking_results["fit_quality_masks"].append(qubit_results["fit_mask"])
    
    return tracking_results


def _track_single_qubit_response(ds: xr.Dataset, qubit_name: str) -> Dict[str, np.ndarray]:
    """
    Track resonator response for a single qubit across power levels.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing measurement data
    qubit_name : str
        Name of the qubit to analyze
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing analysis results for the qubit
    """
    iq_abs = ds.IQ_abs.sel(qubit=qubit_name)
    detuning = ds.detuning.values
    centers = []
    fit_mask = []
    filtered_centers = []
    
    for power_val in ds.power.values:
        signal = iq_abs.sel(power=power_val).values
        
        # Analyze signal to find resonance centers
        center, snr_val, is_good = _analyze_signal_for_resonance(signal, detuning)
        
        centers.append(center)
        fit_mask.append(is_good)
        filtered_centers.append(center if is_good else np.nan)
    
    # Smooth the filtered centers if we have enough good points
    smoothed_centers = _smooth_resonance_centers(filtered_centers, fit_mask)
    
    return {
        "raw_centers": np.array(centers),
        "filtered_centers": np.array(filtered_centers),
        "smoothed_centers": smoothed_centers,
        "fit_mask": np.array(fit_mask)
    }


def _analyze_signal_for_resonance(signal: np.ndarray, detuning: np.ndarray) -> Tuple[float, float, bool]:
    """
    Analyze a single power level signal to find resonance center.
    
    Parameters
    ----------
    signal : np.ndarray
        Signal data for analysis
    detuning : np.ndarray
        Frequency detuning values
        
    Returns
    -------
    Tuple[float, float, bool]
        Resonance center, SNR value, and quality flag
    """
    # Baseline correction
    baseline_window = _get_safe_savgol_window_length(len(signal), sp_params.baseline_window_size.value, sp_params.polyorder.value)
    if baseline_window > 0:
        baseline = savgol_filter(signal, window_length=baseline_window, polyorder=sp_params.polyorder.value)
    else:
        baseline = np.mean(signal)

    signal_corrected = signal - baseline
    
    # Smoothing
    smooth_window = _get_safe_savgol_window_length(len(signal_corrected), sp_params.smooth_window_size.value, sp_params.polyorder.value)
    if smooth_window > 0:
        smoothed = savgol_filter(signal_corrected, window_length=smooth_window, polyorder=sp_params.polyorder.value)
    else:
        smoothed = signal_corrected

    # Noise estimation
    noise = np.std(signal_corrected - smoothed)
    
    # Peak detection
    peaks, properties = find_peaks(
        smoothed, 
        prominence=sp_params.noise_prominence_factor.value * noise, 
        distance=sp_params.peak_distance.value, 
        width=sp_params.peak_width.value
    )
    
    # Determine resonance center and quality
    if len(peaks) >= 2:
        prominences = properties["prominences"]
        idx1, idx2 = peaks[np.argsort(prominences)[-2:]]
        center = min(detuning[idx1], detuning[idx2])
        snr_val = min(np.abs(smoothed[idx1]) / noise, np.abs(smoothed[idx2]) / noise)
    elif len(peaks) == 1:
        idx1 = peaks[0]
        center = detuning[idx1]
        snr_val = np.abs(smoothed[idx1]) / noise
    else:
        center = np.nan
        snr_val = 0.0
    
    is_good = (snr_val > qc_params.snr_threshold.value) and (not np.isnan(center))
    
    return center, snr_val, is_good


def _smooth_resonance_centers(filtered_centers: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """
    Smooth resonance centers using interpolation and filtering.
    
    Parameters
    ----------
    filtered_centers : np.ndarray
        Array of filtered center frequencies
    fit_mask : np.ndarray
        Boolean mask indicating good fits
        
    Returns
    -------
    np.ndarray
        Smoothed center frequencies
    """
    filtered_centers = np.array(filtered_centers)
    fit_mask = np.array(fit_mask)
    
    if np.sum(~np.isnan(filtered_centers)) >= 5:
        from scipy.interpolate import interp1d
        
        valid_idx = ~np.isnan(filtered_centers)
        interp_func = interp1d(
            np.arange(len(filtered_centers))[valid_idx],
            filtered_centers[valid_idx],
            kind="linear",
            fill_value="extrapolate",
        )
        interp_centers = interp_func(np.arange(len(filtered_centers)))
        
        # Apply smoothing
        window = min(11, len(interp_centers) // 2 * 2 + 1)
        if window < 5:
            window = 5
        smoothed = savgol_filter(interp_centers, window_length=window, polyorder=2)
        
        # Mask out bad fits
        smoothed[~fit_mask] = np.nan
        return smoothed
    else:
        return filtered_centers


def _assign_optimal_power_clustering(
    ds: xr.Dataset, 
    qubits: list, 
    n_clusters: int = None, 
    power_margin_db: float = None
) -> Dict[str, float]:
    """
    Assign optimal readout power using KMeans clustering.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing measurement data
    qubits : list
        List of qubit objects
    n_clusters : int
        Number of clusters for KMeans
    power_margin_db : float
        Safety margin to subtract from detected power
        
    Returns
    -------
    Dict[str, float]
        Mapping from qubit name to optimal power
    """
    if n_clusters is None:
        n_clusters = qc_params.optimal_power_n_clusters.value
    if power_margin_db is None:
        power_margin_db = qc_params.optimal_power_margin_db.value

    power_vals = ds.power.values
    numeric_power_vals = _convert_power_strings_to_numeric(power_vals)
    min_power = np.min(numeric_power_vals)
    
    results = {}
    
    for qubit in qubits:
        qubit_name = qubit.name
        qubit_data = ds.sel(qubit=qubit_name)
        iq_abs = qubit_data.IQ_abs.values

        # Ensure correct shape (power, detuning)
        if iq_abs.shape[0] != len(power_vals):
            iq_abs = iq_abs.T

        # Perform clustering on minimum values
        min_values = np.min(iq_abs, axis=1).reshape(-1, 1)
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            labels = kmeans.fit_predict(min_values)

            # Identify ground state cluster (lowest values)
            cluster_order = np.argsort([np.mean(min_values[labels == i]) for i in range(n_clusters)])
            gs_power_indices = np.where(labels == cluster_order[0])[0]

            if len(gs_power_indices) == 0:
                results[qubit_name] = None
                continue

            # Select median power from ground state cluster
            gs_opt_idx = int(np.median(gs_power_indices))
            gs_opt_idx = np.clip(gs_opt_idx, 0, len(numeric_power_vals) - 1)
            detected_power = float(numeric_power_vals[gs_opt_idx])

            # Apply safety margin if valid
            adjusted_power = detected_power - power_margin_db
            if adjusted_power >= min_power:
                results[qubit_name] = adjusted_power
            else:
                results[qubit_name] = detected_power
                
        except Exception:
            # Fallback: use middle power
            results[qubit_name] = float(numeric_power_vals[len(numeric_power_vals) // 2])

    return results


def _calculate_comprehensive_fit_metrics(
    ds: xr.Dataset, 
    qubits: list, 
    optimal_power_dict: Dict[str, float],
    tracking_results: Dict[str, np.ndarray]
) -> Dict[str, list]:
    """
    Calculate comprehensive fit metrics for all qubits.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing measurement data
    qubits : list
        List of qubit objects
    optimal_power_dict : Dict[str, float]
        Optimal power for each qubit
    tracking_results : Dict[str, np.ndarray]
        Resonator tracking results
        
    Returns
    -------
    Dict[str, list]
        Dictionary containing all calculated metrics
    """
    metrics = {
        "snrs": [],
        "num_lines": [],
        "resonance_shift_ranges": [],
        "flatness_first_n": [],
        "noise_metrics": [],
        "low_power_center_stds": []
    }
    
    # Get numeric power values from the dataset
    power_values = ds.power.values
    numeric_power_values = _convert_power_strings_to_numeric(power_values)
    
    for i, q in enumerate(qubits):
        qubit_name = q.name
        
        # Calculate metrics at optimal power - handle None values
        optimal_power = optimal_power_dict[qubit_name]
        if optimal_power is None:
            # Use middle power as fallback
            optimal_power = float(numeric_power_values[len(numeric_power_values) // 2])
        
        # Find the closest power value in the dataset
        power_idx = np.argmin(np.abs(numeric_power_values - optimal_power))
        closest_power = ds.power.values[power_idx]
        
        iq_abs_opt = ds.IQ_abs.sel(qubit=qubit_name, power=closest_power)
        
        # Perform peak detection at optimal power
        pd_result = peaks_dips(iq_abs_opt, "detuning")
        snr_val = float(pd_result.snr.values) if "snr" in pd_result else np.nan
        num_lines_val = int(pd_result.num_peaks.values) if "num_peaks" in pd_result else 1
        
        # Calculate resonance shift range
        smoothed_centers = tracking_results["rr_tracked"][i]
        fit_mask = tracking_results["fit_quality_masks"][i]
        valid_smoothed = smoothed_centers[fit_mask]
        resonance_shift_range = (
            (np.nanmax(valid_smoothed) - np.nanmin(valid_smoothed)) / 1e6 
            if len(valid_smoothed) > 0 else np.nan
        )
        
        # Calculate flatness in first N points
        flatness_val = (
            np.nanmax(smoothed_centers[:qc_params.flatness_n_points.value]) - np.nanmin(smoothed_centers[:qc_params.flatness_n_points.value])
            if np.sum(fit_mask[:qc_params.flatness_n_points.value]) > 0 else np.nan
        )
        
        # Calculate low power center standard deviation
        valid_centers = smoothed_centers[:qc_params.noise_n_points.value][~np.isnan(smoothed_centers[:qc_params.noise_n_points.value])]
        low_power_center_std = float(np.nanstd(smoothed_centers[:qc_params.noise_n_points.value])) if len(valid_centers) >= 2 else 0.0
        
        # Calculate noise metric
        iq_low = np.concatenate([
            ds.IQ_abs.sel(qubit=qubit_name, power=p).values 
            for p in ds.power.values[:qc_params.noise_n_points.value]
        ])
        noise_metric = float(np.std(iq_low))
        
        # Store all metrics
        metrics["snrs"].append(snr_val)
        metrics["num_lines"].append(num_lines_val)
        metrics["resonance_shift_ranges"].append(resonance_shift_range)
        metrics["flatness_first_n"].append(flatness_val)
        metrics["noise_metrics"].append(noise_metric)
        metrics["low_power_center_stds"].append(low_power_center_std)
    
    return metrics


def _add_metrics_to_dataset(
    ds_fit: xr.Dataset, 
    tracking_results: Dict[str, np.ndarray],
    optimal_power_dict: Dict[str, float],
    fit_metrics: Dict[str, list]
) -> xr.Dataset:
    """
    Add all calculated metrics to the dataset.
    
    Parameters
    ----------
    ds_fit : xr.Dataset
        Dataset to add metrics to
    tracking_results : Dict[str, np.ndarray]
        Resonator tracking results
    optimal_power_dict : Dict[str, float]
        Optimal power for each qubit
    fit_metrics : Dict[str, list]
        Calculated fit metrics
        
    Returns
    -------
    xr.Dataset
        Dataset with added metrics
    """
    qubit_names = [str(q) for q in ds_fit.qubit.values]
    
    # Handle None values in optimal_power_dict
    power_values = ds_fit.power.values
    numeric_power_values = _convert_power_strings_to_numeric(power_values)
    fallback_power = float(numeric_power_values[len(numeric_power_values) // 2])
    
    optimal_powers = [
        optimal_power_dict[q] if optimal_power_dict[q] is not None else fallback_power 
        for q in qubit_names
    ]
    
    ds_fit = ds_fit.assign({
        "rr_min_response": (("qubit", "power"), np.stack(tracking_results["rr_tracked"])),
        "rr_min_response_good": (("qubit", "power"), np.stack(tracking_results["rr_tracked_filtered"])),
        "fit_quality_mask": (("qubit", "power"), np.stack(tracking_results["fit_quality_masks"])),
        "optimal_power": ("qubit", optimal_powers),
        "snr": ("qubit", fit_metrics["snrs"]),
        "num_lines": ("qubit", fit_metrics["num_lines"]),
        "resonance_shift_range": ("qubit", fit_metrics["resonance_shift_ranges"]),
        "flatness_first_n": ("qubit", fit_metrics["flatness_first_n"]),
        "noise_metric": ("qubit", fit_metrics["noise_metrics"]),
        "low_power_center_std": ("qubit", fit_metrics["low_power_center_stds"]),
    })
    
    return ds_fit


def _calculate_frequency_shifts_at_optimal_power(ds_fit: xr.Dataset, node: QualibrationNode) -> list:
    """
    Calculate frequency shifts at optimal power for all qubits.
    
    Parameters
    ----------
    ds_fit : xr.Dataset
        Dataset containing fit results
    node : QualibrationNode
        Experiment node
        
    Returns
    -------
    list
        List of frequency shifts for each qubit
    """
    freq_shifts = []
    
    # Get numeric power values from the dataset
    power_values = ds_fit.power.values
    numeric_power_values = _convert_power_strings_to_numeric(power_values)
    
    for q in node.namespace["qubits"]:
        optimal_power = ds_fit["optimal_power"].sel(qubit=q.name).item()
        
        # Handle None optimal power
        if optimal_power is None or np.isnan(optimal_power):
            # Use middle power as fallback
            optimal_power = float(numeric_power_values[len(numeric_power_values) // 2])
        
        # Find the closest power value in the dataset
        power_idx = np.argmin(np.abs(numeric_power_values - optimal_power))
        closest_power = ds_fit.power.values[power_idx]
        
        peak_result = peaks_dips(
            ds_fit.sel(power=closest_power).sel(qubit=q.name).IQ_abs,
            "detuning",
        )
        freq_shifts.append(float(peak_result.position.data))
    
    return freq_shifts


def _extract_and_validate_amplitude_parameters(
    fit: xr.Dataset, 
    node: QualibrationNode, 
    ds_raw: xr.Dataset
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Extract and validate fit parameters from amplitude dependence data.
    
    Parameters
    ----------
    fit : xr.Dataset
        Dataset containing fit results
    node : QualibrationNode
        Experiment node
    ds_raw : xr.Dataset
        Raw dataset for additional analysis
        
    Returns
    -------
    Tuple[xr.Dataset, Dict[str, FitParameters]]
        Validated fit data and results dictionary
    """
    # Calculate resonator frequencies
    full_freq = np.array([q.resonator.RF_frequency for q in node.namespace["qubits"]])
    res_freq = fit.freq_shift + full_freq
    fit = fit.assign_coords(res_freq=("qubit", res_freq.data))
    fit.res_freq.attrs = {"long_name": "resonator frequency", "units": "Hz"}

    # Evaluate outcomes and create results
    outcomes = []
    fit_results = {}
    
    for i, qubit_name in enumerate(fit.qubit.values):
        # Extract parameters for outcome determination
        outcome_params = _extract_amplitude_outcome_parameters(fit, qubit_name, node, ds_raw)
        
        # Determine outcome
        outcome = _determine_amplitude_outcome(outcome_params)
        outcomes.append(outcome)
        
        # Create fit results
        fit_results[qubit_name] = FitParameters(
            resonator_frequency=fit.sel(qubit=qubit_name).res_freq.values.item(),
            frequency_shift=fit.sel(qubit=qubit_name).freq_shift.values.item(),
            optimal_power=fit.sel(qubit=qubit_name).optimal_power.values.item(),
            outcome=outcome,
        )
    
    # Add outcomes to dataset
    fit = fit.assign_coords(outcome=("qubit", outcomes))
    
    return fit, fit_results


def _extract_amplitude_outcome_parameters(
    fit: xr.Dataset, 
    qubit_name: str, 
    node: QualibrationNode,
    ds_raw: xr.Dataset
) -> Dict[str, any]:
    """Extract all parameters needed for amplitude outcome determination."""
    freq_shift = float(fit.freq_shift.sel(qubit=qubit_name).data)
    optimal_power = float(fit.optimal_power.sel(qubit=qubit_name).data)
    snr = fit.snr.sel(qubit=qubit_name).data if "snr" in fit.data_vars else None
    num_lines = fit.num_lines.sel(qubit=qubit_name).data if "num_lines" in fit.data_vars else 1
    low_power_center_std = fit.low_power_center_std.sel(qubit=qubit_name).data if "low_power_center_std" in fit.data_vars else None
    
    # Get experiment parameters
    min_power_dbm = getattr(node.parameters, "min_power_dbm", None)
    max_power_dbm = getattr(node.parameters, "max_power_dbm", None)
    power_values = fit.power.values if "power" in fit.dims or "power" in fit.coords else None
    
    # Get IQ_abs array with correct shape
    iq_abs_arr = ds_raw.IQ_abs.sel(qubit=qubit_name).values
    if iq_abs_arr.shape[0] != len(fit.power.values):
        iq_abs_arr = iq_abs_arr.T
    
    return {
        "freq_shift": freq_shift,
        "optimal_power": optimal_power,
        "frequency_span_in_mhz": node.parameters.frequency_span_in_mhz,
        "snr": snr,
        "num_lines": num_lines,
        "min_power_dbm": min_power_dbm,
        "max_power_dbm": max_power_dbm,
        "power_values": power_values,
        "low_power_center_std": low_power_center_std,
        "iq_abs": iq_abs_arr,
        "detuning": fit.detuning.values,
    }


# Helper Functions

def _get_safe_savgol_window_length(data_length: int, window_size: int, polyorder: int) -> int:
    """Safely determine the window length for savgol_filter."""
    # Ensure window_length is odd and smaller than data_length
    window_length = min(window_size, data_length)
    if window_length % 2 == 0:
        window_length -= 1
    
    # Ensure window_length is greater than polyorder
    if window_length <= polyorder:
        if data_length > polyorder:
            window_length = polyorder + 1 if (polyorder + 1) % 2 != 0 else polyorder + 2
            if window_length > data_length:
                return -1  # Not possible to find a valid window
        else:
            return -1  # Not possible to find a valid window
    
    return window_length


def _determine_amplitude_outcome(
    params: Dict[str, any],
) -> str:
    """
    Determine the outcome for amplitude spectroscopy based on parameters.
    
    Parameters
    ----------
    params : Dict[str, any]
        Dictionary containing all outcome parameters
        
    Returns
    -------
    str
        Outcome description
    """
    freq_shift = params["freq_shift"]
    optimal_power = params["optimal_power"]
    frequency_span_in_mhz = params["frequency_span_in_mhz"]
    snr = params["snr"]
    num_lines = params["num_lines"]
    low_power_center_std = params["low_power_center_std"]
    iq_abs = params["iq_abs"]
    detuning = params["detuning"]
    
    # Check for ground state detection using dip analysis
    gs_seen = True
    if iq_abs is not None and detuning is not None:
        gs_seen = _detect_ground_state_presence(
            iq_abs, 
            qc_params.gs_detection_n_low_powers.value, 
            qc_params.gs_detection_min_count.value, 
            qc_params.gs_detection_window.value
        )

    snr_low = snr is not None and snr < qc_params.snr_min.value

    # Primary checks based on ground state detection
    if not gs_seen and snr_low:
        return "The SNR is low and the power range is too small, consider increasing it"
    if not gs_seen:
        return "The power range is too small, consider increasing it"

    # Legacy logic for compatibility
    if low_power_center_std is not None and low_power_center_std > qc_params.low_power_center_std_threshold_hz.value:
        if snr_low:
            return (
                "The SNR is low and the power is not large enough to observe the punch-out, "
                "consider increasing the maximum readout power and the number of shots"
            )
        else:
            return "The power range is not large enough to observe the punch-out, consider increasing the number of shots"
    
    # Additional quality checks
    if snr_low or np.isnan(freq_shift) or np.isnan(optimal_power):
        return "The SNR isn't large enough, consider increasing the number of shots"
    
    if np.abs(freq_shift) >= frequency_span_in_mhz * 1e6:
        return f"Frequency shift {freq_shift * 1e-6:.0f} MHz exceeds span {frequency_span_in_mhz} MHz"
    
    if num_lines > 1:
        return "Several lines were detected..."
    
    return "successful"


def _detect_ground_state_presence(
    iq_abs: np.ndarray,
    n_low_powers: int,
    min_gs_count: int,
    gs_window: int
) -> bool:
    """
    Detect ground state presence using binary mapping approach.
    
    Parameters
    ----------
    iq_abs : np.ndarray
        IQ amplitude array (power, detuning)
    n_low_powers : int
        Number of low powers to analyze
    min_gs_count : int
        Minimum count for ground state detection
    gs_window : int
        Window size for mode detection
        
    Returns
    -------
    bool
        True if ground state is detected
    """
    try:
        n_powers, n_detuning = iq_abs.shape
        
        # Create binary map of minimum positions
        binary_map = np.zeros((n_powers, n_detuning), dtype=int)
        min_indices = np.argmin(iq_abs, axis=1)
        
        for i, idx in enumerate(min_indices):
            binary_map[i, idx] = 1
        
        # Analyze low power region
        low_power_indices = np.arange(min(n_low_powers, n_powers))
        dip_indices = [
            np.argmax(binary_map[i]) if np.any(binary_map[i]) else None 
            for i in low_power_indices
        ]
        dip_indices = [idx for idx in dip_indices if idx is not None]
        
        if len(dip_indices) >= min_gs_count:
            mode_idx = scipy.stats.mode(dip_indices, keepdims=True)[0][0]
            close_to_mode = [abs(idx - mode_idx) <= gs_window for idx in dip_indices]
            return sum(close_to_mode) >= min_gs_count
        else:
            return False
            
    except Exception:
        return False


def _convert_power_strings_to_numeric(power_values):
    """Convert power string values to numeric values."""
    if len(power_values) == 0:
        return np.array([])
    
    if not isinstance(power_values[0], str):
        return np.array(power_values, dtype=float)
    
    try:
        import re
        numeric_powers = []
        for p in power_values:
            # Use regex to extract number (including negative sign and decimal point)
            match = re.search(r'-?\d+\.?\d*', str(p))
            if match:
                numeric_powers.append(float(match.group()))
            else:
                # Fallback: try to extract any numeric characters
                numeric_part = ''.join(c for c in str(p) if c.isdigit() or c in '.-')
                if numeric_part and numeric_part != '.' and numeric_part != '-':
                    numeric_powers.append(float(numeric_part))
                else:
                    # Last resort: use index
                    numeric_powers.append(float(len(numeric_powers)))
        return np.array(numeric_powers)
    except (ValueError, TypeError, ImportError):
        # Fallback: use indices
        return np.arange(len(power_values), dtype=float)
