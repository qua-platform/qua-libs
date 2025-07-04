from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
from qualibration_libs.analysis import peaks_dips
from qualibration_libs.analysis.models import lorentzian_dip
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V

from qualibrate import QualibrationNode

# Analysis Constants
SNR_MIN: float = 2.5
SNR_DISTORTED: float = 5.0
ASYMMETRY_MIN: float = 0.4
ASYMMETRY_MAX: float = 2.5
SKEWNESS_MAX: float = 1.1  
DISTORTED_FRACTION_LOW_SNR: float = 0.15  
DISTORTED_FRACTION_HIGH_SNR: float = 0.25  
FWHM_ABSOLUTE_THRESHOLD_HZ: float = 3e6  
NRMSE_THRESHOLD: float = 0.3  # More lenient threshold, R-squared removed as it's too aggressive

@dataclass
class FitParameters:
    """Stores the relevant resonator spectroscopy experiment fit parameters for a single qubit."""
    frequency: float
    fwhm: float
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
            frequency = results.frequency
            fwhm = results.fwhm
        else:
            outcome = results.get('outcome', 'unknown')
            frequency = results.get('frequency', 0.0)
            fwhm = results.get('fwhm', 0.0)
        
        if outcome == "successful":
            status_line += " SUCCESS!\n"
        else:
            status_line += f" FAIL! Reason: {outcome}\n"
        
        # Format frequency and FWHM with appropriate units
        freq_str = f"\tResonator frequency: {frequency * 1e-9:.3f} GHz | "
        fwhm_str = f"FWHM: {fwhm * 1e-3:.1f} kHz | "
        
        log_callable(status_line + freq_str + fwhm_str)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """
    Process raw dataset for resonator spectroscopy analysis.
    
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
    
    # Calculate full RF frequency for each qubit
    full_freq = np.array([
        ds.detuning + q.resonator.RF_frequency 
        for q in node.namespace["qubits"]
    ])
    
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Fit resonator spectroscopy data for each qubit in the dataset.

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
    # Perform peak/dip detection on IQ amplitude
    fit_dataset = peaks_dips(ds.IQ_abs, "detuning")
    
    # Extract and validate fit parameters
    fit_data, fit_results = _extract_and_validate_fit_parameters(ds, fit_dataset, node)
    
    return fit_data, fit_results


def _extract_and_validate_fit_parameters(
    ds: xr.Dataset,
    fit: xr.Dataset, 
    node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Extract fit parameters and validate their quality.
    
    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset containing IQ data
    fit : xr.Dataset
        Dataset containing fit results from peak detection
    node : QualibrationNode
        Experiment node
        
    Returns
    -------
    Tuple[xr.Dataset, Dict[str, FitParameters]]
        Validated fit data and results dictionary
    """
    # Calculate resonator frequencies and FWHM
    fit = _calculate_resonator_parameters(fit, node)
    
    # Evaluate fit quality for each qubit and determine outcomes
    fit_results = {}
    outcomes = []
    
    for qubit_name in fit.qubit.values:
        # Extract fit metrics for this qubit
        fit_metrics = _extract_qubit_fit_metrics(ds, fit, qubit_name)
        
        # Determine outcome based on quality checks
        outcome = _determine_resonator_outcome(fit_metrics)
        outcomes.append(outcome)
        
        # Store results
        fit_results[qubit_name] = FitParameters(
            frequency=fit.sel(qubit=qubit_name).res_freq.values.item(),
            fwhm=fit.sel(qubit=qubit_name).fwhm.values.item(),
            outcome=outcome,
        )
    
    # Add outcomes to the fit dataset
    fit = fit.assign_coords(outcome=("qubit", outcomes))
    fit.outcome.attrs = {"long_name": "fit outcome", "units": ""}
    
    return fit, fit_results


def _calculate_resonator_parameters(fit: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """
    Calculate resonator frequency and FWHM from fit results.
    
    Parameters
    ----------
    fit : xr.Dataset
        Dataset with peak detection results
    node : QualibrationNode
        Experiment node
        
    Returns
    -------
    xr.Dataset
        Dataset with resonator parameters added
    """
    # Add metadata to fit results
    fit.attrs = {"long_name": "frequency", "units": "Hz"}
    
    # Calculate fitted resonator frequency
    rf_frequencies = np.array([q.resonator.RF_frequency for q in node.namespace["qubits"]])
    res_freq = fit.position + rf_frequencies
    fit = fit.assign_coords(res_freq=("qubit", res_freq.data))
    fit.res_freq.attrs = {"long_name": "resonator frequency", "units": "Hz"}
    
    # Calculate FWHM
    fwhm = np.abs(fit.width)
    fit = fit.assign_coords(fwhm=("qubit", fwhm.data))
    fit.fwhm.attrs = {"long_name": "resonator fwhm", "units": "Hz"}

    return fit


def _extract_qubit_fit_metrics(ds_raw: xr.Dataset, fit: xr.Dataset, qubit_name: str) -> Dict[str, float]:
    """
    Extract all relevant fit metrics for a single qubit.
    
    Parameters
    ----------
    ds_raw : xr.Dataset
        The raw dataset containing IQ data
    fit : xr.Dataset
        Dataset containing fit results
    qubit_name : str
        Name of the qubit to extract metrics for
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing all fit metrics for the qubit
    """
    qubit_data = fit.sel(qubit=qubit_name)

    # Calculate sweep span for relative comparisons
    if "detuning" in fit.dims:
        sweep_span = float(fit.coords["detuning"].max() - fit.coords["detuning"].min())
    elif "full_freq" in fit.dims:
        sweep_span = float(fit.coords["full_freq"].max() - fit.coords["full_freq"].min())
    else:
        sweep_span = 0.0
    
    # Get raw data for quality metrics
    raw_data = ds_raw.IQ_abs.sel(qubit=qubit_name).values
    detuning_values = ds_raw.detuning.values
    
    # Generate the Lorentzian fit
    fitted_data = lorentzian_dip(
        detuning_values,
        float(qubit_data.amplitude.values),
        float(qubit_data.position.values),
        float(qubit_data.width.values) / 2,  # Convert to half-width at half-max
        float(qubit_data.base_line.mean().values)
    )
    
    # Calculate residuals and quality metrics
    residuals = raw_data - fitted_data
    rmse = np.sqrt(np.mean(residuals**2))
    
    # NRMSE (normalized by peak-to-peak range)
    data_range = np.ptp(raw_data)
    nrmse = rmse / data_range if data_range > 0 else np.inf
    
    # R-squared (coefficient of determination)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((raw_data - np.mean(raw_data))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Ensure R-squared is valid
    if r_squared < 0 or r_squared > 1:
        r_squared = 0  # Invalid fit
    
    return {
        "num_peaks": int(qubit_data.num_peaks.values),
        "snr": float(qubit_data.snr.values),
        "fwhm": float(qubit_data.fwhm.values),
        "sweep_span": sweep_span,
        "asymmetry": float(qubit_data.asymmetry.values),
        "skewness": float(qubit_data.skewness.values),
        "opx_bandwidth_artifact": not bool(qubit_data.opx_bandwidth_artifact.values),
        "rmse": rmse,
        "nrmse": nrmse,
        "r_squared": r_squared,
    }


def _determine_resonator_outcome(
    metrics: Dict[str, float],
    snr_min: float = SNR_MIN,
    snr_distorted: float = SNR_DISTORTED,
    asymmetry_min: float = ASYMMETRY_MIN,
    asymmetry_max: float = ASYMMETRY_MAX,
    skewness_max: float = SKEWNESS_MAX,
    distorted_fraction_low_snr: float = DISTORTED_FRACTION_LOW_SNR,
    distorted_fraction_high_snr: float = DISTORTED_FRACTION_HIGH_SNR,
    fwhm_absolute_threshold: float = FWHM_ABSOLUTE_THRESHOLD_HZ,
) -> str:
    """
    Determine the outcome for resonator spectroscopy based on fit metrics.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary containing fit metrics
    snr_min : float
        Minimum acceptable SNR
    snr_distorted : float
        SNR threshold for distortion detection
    asymmetry_min : float
        Minimum acceptable asymmetry
    asymmetry_max : float
        Maximum acceptable asymmetry
    skewness_max : float
        Maximum acceptable skewness
    distorted_fraction_low_snr : float
        Maximum distortion fraction for low SNR
    distorted_fraction_high_snr : float
        Maximum distortion fraction for high SNR
    fwhm_absolute_threshold : float
        Absolute FWHM threshold
        
    Returns
    -------
    str
        Outcome description
    """
    num_peaks = metrics["num_peaks"]
    snr = metrics["snr"]
    fwhm = metrics["fwhm"]
    sweep_span = metrics["sweep_span"]
    asymmetry = metrics["asymmetry"]
    skewness = metrics["skewness"]
    opx_bandwidth_artifact = metrics["opx_bandwidth_artifact"]
    nrmse = metrics.get("nrmse", np.inf)
    r_squared = metrics.get("r_squared", 0)
    
    # Log metrics for debugging
    print(f"Metrics - SNR: {snr:.2f}, R²: {r_squared:.4f}, NRMSE: {nrmse:.4f}, Asymmetry: {asymmetry:.2f}, Skewness: {skewness:.2f}")

    # Check SNR first
    if snr < snr_min:
        return "The SNR isn't large enough, consider increasing the number of shots"
    
    # Check for OPX bandwidth artifacts
    if opx_bandwidth_artifact:
        return "OPX bandwidth artifact detected, check your experiment bandwidth settings"
    
    # Check number of peaks
    if num_peaks > 1:
        return "Several peaks were detected"
    
    if num_peaks == 0:
        if snr < snr_min:
            return (
                "The SNR isn't large enough, consider increasing the number of shots "
                "and ensure you are looking at the correct frequency range"
            )
        return "No peaks were detected, consider changing the frequency range"
    
    # Check peak shape quality
    if _is_peak_shape_distorted(asymmetry, skewness, asymmetry_min, asymmetry_max, skewness_max):
        return "The peak shape is distorted"
    
    # Check for peak width issues
    if _is_peak_too_wide(fwhm, sweep_span, snr, snr_distorted, 
                        distorted_fraction_low_snr, distorted_fraction_high_snr, 
                        fwhm_absolute_threshold):
        if snr < snr_distorted:
            return "The SNR isn't large enough and the peak shape is distorted"
        else:
            return "Distorted peak detected"
    
    # Final check on NRMSE only (R-squared can be misleading for resonator fits)
    if nrmse > NRMSE_THRESHOLD:
        return f"High fitting error (NRMSE = {nrmse:.3f} > {NRMSE_THRESHOLD})"
    
    # All checks passed
    return "successful"


def _is_peak_shape_distorted(
    asymmetry: float, 
    skewness: float,
    asymmetry_min: float,
    asymmetry_max: float,
    skewness_max: float
) -> bool:
    """
    Check if peak shape indicates distortion based on asymmetry and skewness.
    
    Parameters
    ----------
    asymmetry : float
        Peak asymmetry value
    skewness : float
        Peak skewness value
    asymmetry_min : float
        Minimum acceptable asymmetry
    asymmetry_max : float
        Maximum acceptable asymmetry
    skewness_max : float
        Maximum acceptable skewness
        
    Returns
    -------
    bool
        True if peak shape is distorted
    """
    asymmetry_bad = (asymmetry is not None and 
                    (asymmetry < asymmetry_min or asymmetry > asymmetry_max))
    skewness_bad = (skewness is not None and abs(skewness) > skewness_max)
    
    return asymmetry_bad or skewness_bad


def _is_peak_too_wide(
    fwhm: float,
    sweep_span: float,
    snr: float,
    snr_distorted: float,
    distorted_fraction_low_snr: float,
    distorted_fraction_high_snr: float,
    fwhm_absolute_threshold: float
) -> bool:
    """
    Check if peak is too wide relative to sweep or absolutely.
    
    Parameters
    ----------
    fwhm : float
        Full width at half maximum
    sweep_span : float
        Total sweep span
    snr : float
        Signal-to-noise ratio
    snr_distorted : float
        SNR threshold for distortion classification
    distorted_fraction_low_snr : float
        Maximum acceptable fraction for low SNR
    distorted_fraction_high_snr : float
        Maximum acceptable fraction for high SNR
    fwhm_absolute_threshold : float
        Absolute FWHM threshold
        
    Returns
    -------
    bool
        True if peak is too wide
    """
    # Determine distortion threshold based on SNR
    distorted_fraction = distorted_fraction_low_snr if snr < snr_distorted else distorted_fraction_high_snr
    
    # Check relative width
    relative_width_bad = (sweep_span > 0 and (fwhm / sweep_span > distorted_fraction))
    
    # Check absolute width
    absolute_width_bad = (fwhm > fwhm_absolute_threshold)
    
    return relative_width_bad or absolute_width_bad
