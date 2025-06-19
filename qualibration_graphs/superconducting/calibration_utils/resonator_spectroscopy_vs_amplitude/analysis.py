from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import scipy.stats
import xarray as xr
from qualibration_libs.analysis import peaks_dips
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from scipy.signal import find_peaks, savgol_filter
from sklearn.cluster import KMeans

from qualibrate import QualibrationNode

# --- Analysis Constants ---
SNR_THRESHOLD = 2.0
FLATNESS_N_POINTS = 5
NOISE_N_POINTS = 5
SNR_MIN = 2.0
LOW_POWER_CENTER_STD_THRESHOLD_HZ = 1e6
GS_DETECTION_N_LOW_POWERS = 8
GS_DETECTION_MIN_COUNT = 5
GS_DETECTION_WINDOW = 2
OPTIMAL_POWER_N_CLUSTERS = 2
OPTIMAL_POWER_MARGIN_DB = 3.0
SAVGOL_BASELINE_WINDOW = 51
SAVGOL_SMOOTH_WINDOW = 21
SAVGOL_POLYORDER = 3
PEAK_PROMINENCE_STD_FACTOR = 3.0
PEAK_DISTANCE = 10
PEAK_WIDTH = 3


def _get_savgol_window_length(data_length: int, window_size: int, polyorder: int) -> int:
    """Safely determine the window length for savgol_filter."""
    # Ensure window_length is odd and smaller than data_length
    window_length = min(window_size, data_length)
    if window_length % 2 == 0:
        window_length -= 1
    # Ensure window_length is greater than polyorder
    if window_length <= polyorder:
        # If not possible, find the smallest valid window_length
        if data_length > polyorder:
            window_length = polyorder + 1 if (polyorder + 1) % 2 != 0 else polyorder + 2
            if window_length > data_length:
                return -1  # Not possible to find a valid window
        else:
            return -1  # Not possible to find a valid window
    return window_length


@dataclass
class FitParameters:
    """Stores the relevant node-specific fitted parameters used to update the state at the end of the node."""

    resonator_frequency: float
    frequency_shift: float
    optimal_power: float
    outcome: str


def log_fitted_results(fit_results: dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    log_callable : callable, optional
        Logger for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, results in fit_results.items():
        s_qubit = f"Results for qubit {q}: "
        s_power = f"Optimal readout power: {results['optimal_power']:.2f} dBm | "
        s_freq = f"Resonator frequency: {1e-9 * results['resonator_frequency']:.3f} GHz | "
        s_shift = f"(shift of {1e-6 * results['frequency_shift']:.0f} MHz)\n"
        if results["outcome"] == "successful":
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += f" FAIL! Reason: {results['outcome']}\n"
        log_callable(s_qubit + s_power + s_freq + s_shift)


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


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> tuple[xr.Dataset, dict[str, FitParameters]]:
    ds_fit = ds
    qubits = node.namespace["qubits"]
    rr_tracked = []
    optimal_powers = []
    snrs = []
    num_lines = []
    resonance_shift_ranges = []
    flatness_first_n = []
    noise_metrics = []
    low_power_center_stds = []
    fit_quality_masks = []
    rr_tracked_filtered = []
    flatness_n = FLATNESS_N_POINTS
    noise_power_points = NOISE_N_POINTS
    snr_threshold = SNR_THRESHOLD

    # Use clustering-based power assignment
    optimal_power_dict = _assign_optimal_power(ds, qubits)

    for q in qubits:
        iq_abs = ds.IQ_abs.sel(qubit=q.name)
        detuning = ds.detuning.values
        centers = []
        fit_mask = []
        filtered_centers = []
        for i, p in enumerate(ds.power.values):
            y = iq_abs.sel(power=p).values
            
            baseline_window = _get_savgol_window_length(len(y), SAVGOL_BASELINE_WINDOW, SAVGOL_POLYORDER)
            if baseline_window > 0:
                baseline = savgol_filter(
                    y, window_length=baseline_window, polyorder=SAVGOL_POLYORDER
                )
            else:
                baseline = np.mean(y)

            y_bc = y - baseline
            
            smooth_window = _get_savgol_window_length(len(y_bc), SAVGOL_SMOOTH_WINDOW, SAVGOL_POLYORDER)
            if smooth_window > 0:
                smoothed = savgol_filter(
                    y_bc, window_length=smooth_window, polyorder=SAVGOL_POLYORDER
                )
            else:
                smoothed = y_bc

            noise = np.std(y_bc - smoothed)
            peaks, properties = find_peaks(
                smoothed, prominence=PEAK_PROMINENCE_STD_FACTOR * noise, distance=PEAK_DISTANCE, width=PEAK_WIDTH
            )
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
            is_good = (snr_val > snr_threshold) and (not np.isnan(center))
            fit_mask.append(is_good)
            filtered_centers.append(center if is_good else np.nan)
            centers.append(center)

        centers = np.array(centers)
        fit_mask = np.array(fit_mask)
        filtered_centers = np.array(filtered_centers)

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
            window = min(11, len(interp_centers) // 2 * 2 + 1)
            if window < 5:
                window = 5
            smoothed = savgol_filter(interp_centers, window_length=window, polyorder=2)
            smoothed[~fit_mask] = np.nan
        else:
            smoothed = filtered_centers

        rr_tracked.append(smoothed)
        rr_tracked_filtered.append(filtered_centers)
        fit_quality_masks.append(fit_mask)

        valid_smoothed = smoothed[fit_mask]
        resonance_shift_range = (
            (np.nanmax(valid_smoothed) - np.nanmin(valid_smoothed)) / 1e6 if len(valid_smoothed) > 0 else np.nan
        )
        resonance_shift_ranges.append(resonance_shift_range)

        flatness_val = (
            np.nanmax(smoothed[:flatness_n]) - np.nanmin(smoothed[:flatness_n])
            if np.sum(fit_mask[:flatness_n]) > 0
            else np.nan
        )
        flatness_first_n.append(flatness_val)

        valid_centers = smoothed[:noise_power_points][~np.isnan(smoothed[:noise_power_points])]
        low_power_center_std = float(np.nanstd(smoothed[:noise_power_points])) if len(valid_centers) >= 2 else 0.0
        low_power_center_stds.append(low_power_center_std)

        optimal_power = optimal_power_dict[q.name]
        optimal_powers.append(optimal_power)

        iq_abs_opt = iq_abs.sel(power=optimal_power, method="nearest")
        pd_result = peaks_dips(iq_abs_opt, "detuning")
        snr_val = float(pd_result.snr.values) if "snr" in pd_result else float("nan")
        num_lines_val = int(pd_result.num_peaks.values) if "num_peaks" in pd_result else 1
        snrs.append(snr_val)
        num_lines.append(num_lines_val)

        iq_low = np.concatenate([iq_abs.sel(power=p).values for p in ds.power.values[:noise_power_points]])
        noise_metrics.append(float(np.std(iq_low)))

    ds_fit = ds_fit.assign(
        {
            "rr_min_response": (("qubit", "power"), np.stack(rr_tracked)),
            "rr_min_response_good": (("qubit", "power"), np.stack(rr_tracked_filtered)),
            "fit_quality_mask": (("qubit", "power"), np.stack(fit_quality_masks)),
            "optimal_power": ("qubit", np.array(optimal_powers)),
            "snr": ("qubit", np.array(snrs)),
            "num_lines": ("qubit", np.array(num_lines)),
            "resonance_shift_range": ("qubit", np.array(resonance_shift_ranges)),
            "flatness_first_n": ("qubit", np.array(flatness_first_n)),
            "noise_metric": ("qubit", np.array(noise_metrics)),
            "low_power_center_std": ("qubit", np.array(low_power_center_stds)),
        }
    )

    def _select_optimal_power(ds, qubit):
        return peaks_dips(
            ds.sel(power=ds["optimal_power"].sel(qubit=qubit).item(), method="nearest").sel(qubit=qubit).IQ_abs,
            "detuning",
        )

    freq_shift = []
    for q in node.namespace["qubits"]:
        freq_shift.append(float(_select_optimal_power(ds_fit, q.name).position.data))
    ds_fit = ds_fit.assign_coords({"freq_shift": (["qubit"], freq_shift)})

    fit_dataset, fit_results = _extract_relevant_fit_parameters(ds_fit, node, ds)
    return fit_dataset, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode, ds_raw: xr.Dataset):
    """Add metadata to the fit dataset and fit result dictionary."""

    # Get the fitted resonator frequency
    full_freq = np.array([q.resonator.RF_frequency for q in node.namespace["qubits"]])
    res_freq = fit.freq_shift + full_freq
    fit = fit.assign_coords(res_freq=("qubit", res_freq.data))
    fit.res_freq.attrs = {"long_name": "resonator frequency", "units": "Hz"}

    # Single loop for both outcomes and fit_results
    outcomes = []
    fit_results: dict[str, FitParameters] = {}
    for i, q in enumerate(fit.qubit.values):
        freq_shift = float(fit.freq_shift.sel(qubit=q).data)
        optimal_power = float(fit.optimal_power.sel(qubit=q).data)
        snr = fit.snr.sel(qubit=q).data if "snr" in fit.data_vars else None
        num_lines = fit.num_lines.sel(qubit=q).data if "num_lines" in fit.data_vars else 1
        min_power_dbm = node.parameters.min_power_dbm if hasattr(node.parameters, "min_power_dbm") else None
        max_power_dbm = node.parameters.max_power_dbm if hasattr(node.parameters, "max_power_dbm") else None
        power_values = fit.power.values if "power" in fit.dims or "power" in fit.coords else None
        low_power_center_std = (
            fit.low_power_center_std.sel(qubit=q).data if "low_power_center_std" in fit.data_vars else None
        )
        iq_abs_arr = ds_raw.IQ_abs.sel(qubit=q).values
        # Ensure shape is (power, detuning)
        if iq_abs_arr.shape[0] != len(fit.power.values):
            iq_abs_arr = iq_abs_arr.T
        outcome = _get_fit_outcome(
            freq_shift=freq_shift,
            optimal_power=optimal_power,
            frequency_span_in_mhz=node.parameters.frequency_span_in_mhz,
            snr=snr,
            num_lines=num_lines,
            min_power_dbm=min_power_dbm,
            max_power_dbm=max_power_dbm,
            power_values=power_values,
            low_power_center_std=low_power_center_std,
            iq_abs=iq_abs_arr,
            detuning=fit.detuning.values,
        )
        outcomes.append(outcome)
        fit_results[q] = FitParameters(
            resonator_frequency=fit.sel(qubit=q).res_freq.values.item(),
            frequency_shift=fit.sel(qubit=q).freq_shift.values.item(),
            optimal_power=fit.sel(qubit=q).optimal_power.values.item(),
            outcome=outcome,
        )
    fit = fit.assign_coords(outcome=("qubit", outcomes))
    return fit, fit_results


# Helper Functions
def _get_fit_outcome(
    freq_shift: float,
    optimal_power: float,
    frequency_span_in_mhz: float,
    snr: float = None,
    num_lines: int = 1,
    snr_min: float = SNR_MIN,
    min_power_dbm: float = None,
    max_power_dbm: float = None,
    power_values: np.ndarray = None,
    low_power_center_std: float = None,
    low_power_center_std_thresh: float = LOW_POWER_CENTER_STD_THRESHOLD_HZ,  # 1 MHz in Hz
    iq_abs: np.ndarray = None,
    detuning: np.ndarray = None,
    n_low_powers: int = GS_DETECTION_N_LOW_POWERS,
    min_gs_count: int = GS_DETECTION_MIN_COUNT,
    gs_window: int = GS_DETECTION_WINDOW,
) -> str:
    """
    Returns the outcome string for a given fit result, with GS detection as the primary check.
    """
    gs_seen = True
    if iq_abs is not None and detuning is not None:
        n_powers, n_detuning = iq_abs.shape
        M = n_low_powers
        K = min_gs_count
        window = gs_window
        binary_map = np.zeros((n_powers, n_detuning), dtype=int)
        min_indices = np.argmin(iq_abs, axis=1)
        for i, idx in enumerate(min_indices):
            binary_map[i, idx] = 1
        low_power_indices = np.arange(min(M, n_powers))
        dip_indices = [np.argmax(binary_map[i]) if np.any(binary_map[i]) else None for i in low_power_indices]
        dip_indices = [idx for idx in dip_indices if idx is not None]
        if len(dip_indices) >= K:
            mode_idx = scipy.stats.mode(dip_indices, keepdims=True)[0][0]
            close_to_mode = [abs(idx - mode_idx) <= window for idx in dip_indices]
            if sum(close_to_mode) >= K:
                gs_seen = True
            else:
                gs_seen = False
        else:
            gs_seen = False

    snr_low = snr is not None and snr < snr_min

    if not gs_seen and snr_low:
        return "The SNR is low and the power range is too small, consider increasing it"
    if not gs_seen:
        return "The power range is too small, consider increasing it"

    # --- legacy logic ---
    if low_power_center_std is not None and low_power_center_std > low_power_center_std_thresh:
        if snr_low:
            return (
                "The SNR is low and the power is not large enough to observe the punch-out, "
                "consider increasing the maximum readout power and the number of shots"
            )
        else:
            return "The power range is not large enough to observe the punch-out, consider increasing the number of shots"
    if snr_low or np.isnan(freq_shift) or np.isnan(optimal_power):
        return "The SNR isn't large enough, consider increasing the number of shots"
    if np.abs(freq_shift) >= frequency_span_in_mhz * 1e6:
        return f"Frequency shift {1e-6 * freq_shift:.0f} MHz exceeds span {frequency_span_in_mhz} MHz"
    if num_lines > 1:
        return "Several lines were detected..."
    return "successful"


def _assign_optimal_power(
    ds: xr.Dataset, qubits: list[AnyTransmon], n_clusters: int = OPTIMAL_POWER_N_CLUSTERS, power_margin_db: float = OPTIMAL_POWER_MARGIN_DB
) -> dict[str, float]:
    """
    Assign optimal readout power using KMeans clustering on minimum IQ_abs,
    then optionally subtract a fixed margin (default 1 dB) from the selected power
    if doing so keeps the value within the valid power range.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing 'IQ_abs', 'power', 'detuning', and 'qubit'.
    qubits : list[AnyTransmon]
        List of qubit objects with `.name` attribute.
    n_clusters : int
        Number of clusters for KMeans (default: 2).
    power_margin_db : float
        Value in dB to subtract from the detected optimal power (default: 3.0).

    Returns
    -------
    dict[str, float]
        Mapping from qubit name to optimal power (with margin applied if valid).
    """
    power_vals = np.array(ds.power.values)
    min_power = np.min(power_vals)
    max_power = np.max(power_vals)

    results = {}

    for qubit in qubits:
        qubit_name = qubit.name
        qubit_data = ds.sel(qubit=qubit_name)
        iq_abs = qubit_data.IQ_abs.values

        if iq_abs.shape[0] != len(power_vals):
            iq_abs = iq_abs.T

        min_values = np.min(iq_abs, axis=1).reshape(-1, 1)

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        labels = kmeans.fit_predict(min_values)

        cluster_order = np.argsort([np.mean(min_values[labels == i]) for i in range(n_clusters)])
        gs_power_indices = np.where(labels == cluster_order[0])[0]

        if len(gs_power_indices) == 0:
            results[qubit_name] = None
            continue

        gs_opt_idx = int(np.median(gs_power_indices))
        gs_opt_idx = np.clip(gs_opt_idx, 0, len(power_vals) - 1)
        detected_power = float(power_vals[gs_opt_idx])

        adjusted_power = detected_power - power_margin_db
        if adjusted_power >= min_power:
            results[qubit_name] = adjusted_power
        else:
            results[qubit_name] = detected_power

    return results
