from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qualibration_libs.analysis import fit_oscillation
from qualibration_libs.analysis.parameters import analysis_config_manager
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

from qualibrate import QualibrationNode

# Access parameter groups from the config manager
qc_params = analysis_config_manager.get("resonator_spectroscopy_vs_flux_qc")
sp_params = analysis_config_manager.get("common_signal_processing")


@dataclass
class FitParameters:
    """Stores the relevant node-specific fitted parameters used to update the state at the end of the node."""
    resonator_frequency: float
    frequency_shift: float
    min_offset: float
    idle_offset: float
    dv_phi0: float
    phi0_current: float
    m_pH: float
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
            idle_offset = results.idle_offset
            min_offset = results.min_offset
            resonator_frequency = results.resonator_frequency
            frequency_shift = results.frequency_shift
        else:
            outcome = results.get('outcome', 'unknown')
            idle_offset = results.get('idle_offset', 0.0)
            min_offset = results.get('min_offset', 0.0)
            resonator_frequency = results.get('resonator_frequency', 0.0)
            frequency_shift = results.get('frequency_shift', 0.0)
        
        if outcome == "successful":
            status_line += " SUCCESS!\n"
        else:
            status_line += f" FAIL! Reason: {outcome}\n"
        
        # Format parameters with appropriate units
        idle_str = f"\tidle offset: {idle_offset * 1e3:.0f} mV | "
        min_str = f"min offset: {min_offset * 1e3:.0f} mV | "
        freq_str = f"Resonator frequency: {resonator_frequency * 1e-9:.3f} GHz | "
        shift_str = f"(shift of {frequency_shift * 1e-6:.0f} MHz)\n"
        
        log_callable(status_line + idle_str + min_str + freq_str + shift_str)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """
    Process raw dataset for resonator spectroscopy vs flux analysis.
    
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
    
    # Add current axis for each qubit
    current = ds.flux_bias / node.parameters.input_line_impedance_in_ohm
    ds = ds.assign_coords({"current": (["flux_bias"], current.data)})
    ds.current.attrs["long_name"] = "Current"
    ds.current.attrs["units"] = "A"
    
    # Add attenuated current to dataset
    attenuation_factor = 10 ** (-node.parameters.line_attenuation_in_db / 20)
    attenuated_current = ds.current * attenuation_factor
    ds = ds.assign_coords({"attenuated_current": (["flux_bias"], attenuated_current.values)})
    ds.attenuated_current.attrs["long_name"] = "Attenuated Current"
    ds.attenuated_current.attrs["units"] = "A"
    
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Robustly fit the resonance for each qubit as a function of flux.
    
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
    # Find minimum frequency points for each flux bias
    peak_freq = ds.IQ_abs.idxmin(dim="detuning")
    
    # Evaluate data quality for each qubit
    qubit_outcomes = _evaluate_qubit_data_quality(ds, peak_freq)
    
    # Fit oscillation only for qubits with valid data, but keep all qubits for final processing
    peak_freq_for_fit = peak_freq.dropna(dim="flux_bias")
    
    if peak_freq_for_fit.qubit.size > 0:
        # Fit oscillation for valid qubits
        fit_results_da = fit_oscillation(peak_freq_for_fit, "flux_bias")
        fit_results_ds = xr.merge([fit_results_da.rename("fit_results"), peak_freq.rename("peak_freq")])
    else:
        # No qubits have valid data - create empty fit dataset with all qubits
        fit_results_ds = xr.Dataset({
            "peak_freq": peak_freq
        })
    
    # Extract and validate fit parameters for ALL qubits (including failed ones)
    fit_dataset, fit_results = _extract_and_validate_flux_parameters(fit_results_ds, node, qubit_outcomes)
    
    return fit_dataset, fit_results


def _evaluate_qubit_data_quality(ds: xr.Dataset, peak_freq: xr.Dataset) -> Dict[str, str]:
    """
    Evaluate data quality for each qubit to determine which can be fit.
    
    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset
    peak_freq : xr.Dataset
        Peak frequency data
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping qubit names to quality assessment outcomes
    """
    qubit_outcomes = {}
    
    for q in ds.qubit.values:
        # Check for resonator trace
        if not _has_resonator_trace(ds, q):
            qubit_outcomes[q] = "no_peaks"
            continue

        # Check for insufficient flux modulation
        if _has_insufficient_flux_modulation(ds, q):
            qubit_outcomes[q] = "no_oscillations"
            continue

        # Check peak frequency quality
        pf = peak_freq.sel(qubit=q)
        pf_vals = pf.values
        n_nan = np.sum(np.isnan(pf_vals))
        frac_nan = n_nan / pf_vals.size
        pf_valid = pf_vals[~np.isnan(pf_vals)]
        
        if pf_valid.size == 0 or frac_nan > qc_params.nan_fraction_threshold.value:
            qubit_outcomes[q] = "no_peaks"
        else:
            pf_std = np.nanstd(pf_valid)
            pf_mean = np.nanmean(np.abs(pf_valid))
            if pf_mean == 0 or pf_std < qc_params.flat_std_rel_threshold.value * pf_mean:
                qubit_outcomes[q] = "no_peaks"
            else:
                qubit_outcomes[q] = "fit"

    return qubit_outcomes


def _extract_and_validate_flux_parameters(
    fit: xr.Dataset, 
    node: QualibrationNode, 
    qubit_outcomes: Dict[str, str]
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Extract and validate fit parameters from flux dependence data.
    
    Parameters
    ----------
    fit : xr.Dataset
        Dataset containing fit results
    node : QualibrationNode
        Experiment node
    qubit_outcomes : Dict[str, str]
        Initial quality assessment for each qubit
        
    Returns
    -------
    Tuple[xr.Dataset, Dict[str, FitParameters]]
        Validated fit data and results dictionary
    """
    # Calculate flux bias parameters only for qubits that have fit_results
    if "fit_results" in fit.data_vars:
        fit = _calculate_flux_bias_parameters(fit)
        fit = _calculate_flux_frequency_parameters(fit, node)
    
    # Evaluate outcomes and create results for ALL qubits
    fit_results = {}
    outcomes = []
    
    # Process all qubits from the original dataset
    all_qubits = list(node.namespace["qubits"])
    
    for i, q_obj in enumerate(all_qubits):
        q = q_obj.name
        
        # Check if this qubit has fit results or failed initial quality checks
        if q in qubit_outcomes and qubit_outcomes[q] in ["no_peaks", "no_oscillations"]:
            # This qubit failed initial quality checks
            if qubit_outcomes[q] == "no_peaks":
                outcome = "No peaks were detected, consider changing the frequency range"
            else:
                outcome = "No oscillations were detected, consider checking that the flux line is connected or increase the flux range"
            
            # Create default FitParameters for failed qubit
            fit_results[q] = FitParameters(
                resonator_frequency=np.nan,
                frequency_shift=np.nan,
                min_offset=np.nan,
                idle_offset=np.nan,
                dv_phi0=np.nan,
                phi0_current=np.nan,
                m_pH=np.nan,
                outcome=outcome,
            )
            outcomes.append(outcome)
            
        elif q in fit.qubit.values and "fit_results" in fit.data_vars:
            # This qubit has fit results - process normally
            q_index = i
            
            # Extract parameters for outcome determination
            outcome_params = _extract_outcome_parameters(
                fit, q, q_index, node, qubit_outcomes
            )
            
            # Determine outcome
            outcome = _determine_flux_outcome(outcome_params, qubit_outcomes.get(q, "fit"))
            outcomes.append(outcome)
            
            # Apply frequency correction if needed
            corrected_frequency = _apply_frequency_correction_if_needed(
                fit, q, q_index, node, outcome_params
            )
            
            # Create fit parameters
            fit_results[q] = _create_flux_fit_parameters(
                fit, q, q_index, node, outcome, corrected_frequency
            )
        else:
            # This qubit doesn't have fit results but also wasn't marked as failed
            # This shouldn't happen, but handle gracefully
            outcome = "No peaks were detected, consider changing the frequency range"
            fit_results[q] = FitParameters(
                resonator_frequency=np.nan,
                frequency_shift=np.nan,
                min_offset=np.nan,
                idle_offset=np.nan,
                dv_phi0=np.nan,
                phi0_current=np.nan,
                m_pH=np.nan,
                outcome=outcome,
            )
            outcomes.append(outcome)
    
    # Add outcomes to dataset (only for qubits that have fit results)
    if "fit_results" in fit.data_vars and len(outcomes) > 0:
        fit = fit.assign_coords(outcome=("qubit", [outcomes[i] for i, q_obj in enumerate(all_qubits) if q_obj.name in fit.qubit.values]))
    
    return fit, fit_results


def _calculate_flux_bias_parameters(fit: xr.Dataset) -> xr.Dataset:
    """Calculate flux bias parameters from fit results."""
    # Ensure phase is between -π and π
    flux_idle = -fit.sel(fit_vals="phi").fit_results
    flux_idle = np.mod(flux_idle + np.pi, 2 * np.pi) - np.pi
    
    # Convert phase to voltage
    flux_idle = flux_idle / fit.sel(fit_vals="f").fit_results / 2 / np.pi
    fit = fit.assign_coords(idle_offset=("qubit", flux_idle.data))
    fit.idle_offset.attrs = {"long_name": "idle flux bias", "units": "V"}
    
    # Find minimum frequency flux point
    flux_min = flux_idle + ((flux_idle < 0) - 0.5) / fit.sel(fit_vals="f").fit_results
    flux_min = flux_min * (np.abs(flux_min) < 0.5) + 0.5 * (flux_min > 0.5) - 0.5 * (flux_min < -0.5)
    fit = fit.assign_coords(flux_min=("qubit", flux_min.data))
    fit.flux_min.attrs = {"long_name": "minimum frequency flux bias", "units": "V"}
    
    return fit


def _calculate_flux_frequency_parameters(fit: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Calculate frequency parameters from flux dependence."""
    # Get base frequencies
    full_freq = np.array([q.resonator.RF_frequency for q in node.namespace["qubits"]])
    
    # Calculate frequency shift at sweet spot
    flux_idle = fit.idle_offset
    freq_shift = fit.peak_freq.sel(flux_bias=flux_idle, method="nearest")
    fit = fit.assign_coords(freq_shift=("qubit", freq_shift.data))
    fit.freq_shift.attrs = {"long_name": "frequency shift", "units": "Hz"}
    
    # Calculate sweet spot frequency
    fit = fit.assign_coords(sweet_spot_frequency=("qubit", freq_shift.data + full_freq))
    fit.sweet_spot_frequency.attrs = {"long_name": "sweet spot frequency", "units": "Hz"}
    
    return fit


def _extract_outcome_parameters(
    fit: xr.Dataset, 
    qubit: str, 
    q_index: int, 
    node: QualibrationNode,
    qubit_outcomes: Dict[str, str]
) -> Dict[str, float]:
    """Extract all parameters needed for outcome determination."""
    flux_idle_val = float(fit.idle_offset.sel(qubit=qubit).data) if not np.isnan(fit.idle_offset.sel(qubit=qubit).data) else np.nan
    flux_min_val = float(fit.flux_min.sel(qubit=qubit).data) if not np.isnan(fit.flux_min.sel(qubit=qubit).data) else np.nan
    freq_shift_val = float(fit.freq_shift.sel(qubit=qubit).values) if not np.isnan(fit.freq_shift.sel(qubit=qubit).values) else np.nan
    
    # Calculate SNR
    snr = _calculate_snr(node.results["ds_raw"], qubit)
    
    # Check for oscillations
    has_oscillations = _has_oscillations_in_fit(fit, qubit, qubit_outcomes)
    
    return {
        "freq_shift": freq_shift_val,
        "flux_min": flux_min_val,
        "flux_idle": flux_idle_val,
        "frequency_span_in_mhz": node.parameters.frequency_span_in_mhz,
        "snr": snr,
        "has_oscillations": has_oscillations,
        "has_anticrossings": False,  # Not currently detected
    }


def _apply_frequency_correction_if_needed(
    fit: xr.Dataset, 
    qubit: str, 
    q_index: int, 
    node: QualibrationNode,
    outcome_params: Dict[str, float]
) -> float:
    """Apply frequency correction if confidence is low."""
    # Get base frequency
    full_freq = np.array([q.resonator.RF_frequency for q in node.namespace["qubits"]])
    base_freq = full_freq[q_index]
    
    # Calculate confidence metric
    fitted_flux_idle = outcome_params["flux_idle"]
    peak_freq_vs_flux = fit.peak_freq.sel(qubit=qubit)
    flux_biases = peak_freq_vs_flux.coords["flux_bias"].values
    freq_vals = peak_freq_vs_flux.values

    # Check confidence around idle flux
    idx_idle = np.argmin(np.abs(flux_biases - fitted_flux_idle))
    freq_at_idle = freq_vals[idx_idle]
    window = qc_params.resonance_correction_window.value
    start = max(0, idx_idle - window)
    end = min(len(freq_vals), idx_idle + window + 1)
    local_min = np.min(freq_vals[start:end])
    local_mean = np.mean(freq_vals[start:end])
    global_min = np.min(freq_vals)
    global_max = np.max(freq_vals)
    confidence = (local_mean - freq_at_idle) / (global_max - global_min + 1e-12)

    # Correct frequency if confidence is low
    if confidence < qc_params.resonance_correction_confidence_threshold.value:
        corrected_freq = _correct_resonator_frequency(
            ds_raw=node.results["ds_raw"],
            qubit=qubit,
            ds_fit=fit,
        )
        # Update the fit dataset
        fit["sweet_spot_frequency"].loc[dict(qubit=qubit)] = corrected_freq
        fit["freq_shift"].loc[dict(qubit=qubit)] = corrected_freq - base_freq
        return corrected_freq
    else:
        return outcome_params["freq_shift"] + base_freq


def _create_flux_fit_parameters(
    fit: xr.Dataset, 
    qubit: str, 
    q_index: int, 
    node: QualibrationNode, 
    outcome: str, 
    corrected_frequency: float
) -> FitParameters:
    """Create FitParameters object for a qubit."""
    # Get attenuation factor
    attenuation_factor = 10 ** (-node.parameters.line_attenuation_in_db / 20)
    
    # Calculate m_pH
    frequency_factor = fit.sel(fit_vals="f", qubit=qubit).fit_results.data
    m_pH = (1e12 * 2.068e-15 / (1 / frequency_factor) / 
            node.parameters.input_line_impedance_in_ohm * attenuation_factor)
    
    # Get base frequency
    full_freq = np.array([q.resonator.RF_frequency for q in node.namespace["qubits"]])
    base_freq = full_freq[q_index]
    
    return FitParameters(
        resonator_frequency=corrected_frequency if outcome == "successful" else np.nan,
        frequency_shift=corrected_frequency - base_freq if outcome == "successful" else float(fit.sel(qubit=qubit).freq_shift.values),
        min_offset=float(fit.sel(qubit=qubit).flux_min.data),
        idle_offset=float(fit.sel(qubit=qubit).idle_offset.data),
        dv_phi0=1 / frequency_factor if outcome == "successful" else np.nan,
        phi0_current=(1 / frequency_factor * node.parameters.input_line_impedance_in_ohm * 
                     attenuation_factor if outcome == "successful" else np.nan),
        m_pH=m_pH if outcome == "successful" else np.nan,
        outcome=outcome,
    )


# Helper Functions for Quality Assessment

def _has_oscillations_in_fit(
    fit: xr.Dataset,
    qubit: str,
    qubit_outcomes: Dict[str, str],
) -> bool:
    """Determine if oscillations were detected in the fit for a given qubit."""
    if qubit_outcomes.get(qubit) in ["no_peaks", "no_oscillations"]:
        return False

    try:
        # Extract fit amplitude
        amp = float(fit.sel(fit_vals="a", qubit=qubit).fit_results.data)
        pf_valid = fit.peak_freq.sel(qubit=qubit).values
        pf_valid = pf_valid[~np.isnan(pf_valid)]
        pf_median = np.median(np.abs(pf_valid)) if pf_valid.size > 0 else 1.0
        return np.abs(amp) >= qc_params.amp_rel_threshold.value * pf_median
    except:
        return False


def _calculate_snr(ds: xr.Dataset, qubit: str) -> float:
    """Calculate signal-to-noise ratio for a qubit's spectroscopy data."""
    try:
        data = ds.IQ_abs.sel(qubit=qubit)
        signal = np.ptp(data.values)  # peak-to-peak as signal measure
        noise = np.std(data.values)   # standard deviation as noise measure
        return signal / (noise + sp_params.epsilon.value) if noise > 0 else np.inf
    except:
        return np.nan


def _determine_flux_outcome(
    params: Dict[str, float],
    initial_outcome: str,
) -> str:
    """Determine the outcome for flux spectroscopy based on parameters."""
    freq_shift = params["freq_shift"]
    flux_min = params["flux_min"]
    flux_idle = params["flux_idle"]
    frequency_span_in_mhz = params["frequency_span_in_mhz"]
    snr = params["snr"]
    has_oscillations = params["has_oscillations"]
    has_anticrossings = params["has_anticrossings"]
    
    if not has_oscillations:
        return (
            "No oscillations were detected, "
            "consider checking that the flux line is connected or increase the flux range"
        )

    if has_anticrossings:
        return "Anti-crossings were detected, consider adjusting the flux range or checking the device setup"

    snr_low = snr is not None and snr < qc_params.snr_min.value
    if snr_low:
        return "The SNR isn't large enough, consider increasing the number of shots"

    if np.isnan(freq_shift) or np.isnan(flux_min) or np.isnan(flux_idle):
        return "No peaks were detected, consider changing the frequency range"

    if np.abs(freq_shift) >= frequency_span_in_mhz * 1e6:
        return f"Frequency shift {freq_shift * 1e-6:.0f} MHz exceeds span {frequency_span_in_mhz} MHz"

    return "successful"


def _has_resonator_trace(
    ds: xr.Dataset,
    qubit: str,
    var_name: str = "IQ_abs",
    freq_dim: str = "detuning",
    flux_dim: str = "flux_bias",
) -> bool:
    """Detect whether a resonator-like frequency trace exists."""
    try:
        # Extract and normalize data
        da = ds[var_name].sel(qubit=qubit)
        da_norm = da / (da.mean(dim=freq_dim) + sp_params.epsilon.value)

        # Get minimum trace across frequency sweep
        min_trace = da_norm.min(dim=freq_dim).values

        # Smooth to suppress noise
        min_trace_smooth = gaussian_filter1d(min_trace, sigma=qc_params.resonator_trace_smooth_sigma.value)

        # Calculate peak-to-peak dip depth and variation
        dip_depth = np.max(min_trace_smooth) - np.min(min_trace_smooth)
        gradient = np.max(np.abs(np.gradient(min_trace_smooth)))

        return (dip_depth > qc_params.resonator_trace_dip_threshold.value) and (gradient > qc_params.resonator_trace_gradient_threshold.value)
    except:
        return False


def _has_insufficient_flux_modulation(
    ds: xr.Dataset, 
    qubit: str, 
) -> bool:
    """Detect whether the flux modulation of the resonator is too small."""
    try:
        # Calculate peak frequency from IQ_abs minimum
        peak_idx = ds.IQ_abs.argmin(dim="detuning")
        peak_freq = ds.full_freq.isel(detuning=peak_idx)
        peak_freq = peak_freq.transpose("qubit", "flux_bias")

        pf = peak_freq.sel(qubit=qubit)
        if pf.isnull().all():
            return True

        freq_vals = pf.values
        freq_vals = freq_vals[~np.isnan(freq_vals)]

        if freq_vals.size < 2:
            return True

        smoothed = gaussian_filter1d(freq_vals, sigma=qc_params.flux_modulation_smooth_sigma.value)
        modulation_range = np.max(smoothed) - np.min(smoothed)

        return modulation_range < qc_params.min_flux_modulation_hz.value
    except:
        return True


def _correct_resonator_frequency(ds_raw: xr.Dataset, qubit: str, ds_fit: xr.Dataset) -> float:
    """Compute corrected resonator frequency using |IQ| minimum at fitted idle flux."""
    try:
        # Extract data
        IQ_abs = ds_raw["IQ_abs"].sel(qubit=qubit)
        flux_vals = ds_raw["flux_bias"].values
        freq_vals = ds_raw["full_freq"].sel(qubit=qubit).values / 1e9  # GHz
        
        # Get idle flux from fit
        idle_flux = float(ds_fit.sel(qubit=qubit)["idle_offset"].values)
        
        # Find nearest flux point and get minimum
        nearest_flux = flux_vals[np.argmin(np.abs(flux_vals - idle_flux))]
        trace = ds_raw["IQ_abs"].sel(qubit=qubit, flux_bias=nearest_flux)
        smoothed = gaussian_filter1d(trace.values, sigma=2)
        min_idx = np.argmin(smoothed)
        corrected_freq = freq_vals[min_idx] * 1e9  # Hz
        
        return corrected_freq
    except:
        # Fallback to original value
        return float(ds_fit.sel(qubit=qubit)["sweet_spot_frequency"].values)