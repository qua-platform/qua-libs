from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import xarray as xr
from qualibration_libs.analysis import peaks_dips
from qualibration_libs.analysis.feature_detection import (
    estimate_width_from_curvature, find_all_peaks_from_signal)
from qualibration_libs.analysis.fitting import calculate_quality_metrics
from qualibration_libs.analysis.models import lorentzian_dip
from qualibration_libs.analysis.parameters import analysis_config_manager
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from scipy.optimize import curve_fit

from qualibrate import QualibrationNode

# Access quality check parameters from the config object
qc_params = analysis_config_manager.get("resonator_spectroscopy_qc")


@dataclass
class FitParameters:
    """Stores the relevant resonator spectroscopy experiment fit parameters for a single qubit."""
    frequency: float
    fwhm: float
    outcome: str
    fit_metrics: Optional[Dict[str, Any]] = None


def log_fitted_results(fit_results: Dict[str, FitParameters], log_callable=None) -> None:
    """
    Logs the node-specific fitted results for all qubits from the fit results.

    Args:
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
    
    Args:
        ds : xr.Dataset
            Raw dataset containing measurement data
        node : QualibrationNode
            The qualibration node containing parameters and qubit information
        
    Returns:
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

    Args:
        ds : xr.Dataset
            Dataset containing the processed data
        node : QualibrationNode
            The qualibration node containing experiment parameters

    Returns:
        Tuple[xr.Dataset, Dict[str, FitParameters]]
            Tuple containing:
            - Dataset with fit results and quality metrics
            - Dictionary mapping qubit names to fit parameters
    """
    # Perform peak/dip detection on IQ amplitude, providing initial estimates
    fit_dataset = peaks_dips(ds.IQ_abs, "detuning")

    # Extract, validate, and potentially re-fit parameters
    fit_data, fit_results = _extract_and_validate_fit_parameters(ds, fit_dataset, node)

    return fit_data, fit_results


def _fit_multi_peak_resonator(
    ds_qubit: xr.Dataset, fit_qubit: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Apply advanced fitting for a qubit with multiple detected peaks.

    This method implements a robust fitting pipeline for signals with multiple resonance features:
    1. It inverts the signal to treat dips as peaks.
    2. Finds all significant peaks in the signal.
    3. Selects the most prominent peak as the candidate for the main resonance.
    4. Estimates its width using the signal's curvature (2nd derivative).
    5. Defines a narrow fitting window around the candidate peak.
    6. Guesses initial parameters from the data within the window.
    7. Performs a Lorentzian fit on the windowed data.
    8. Validates the fit and returns the parameters.

    Args:
        ds_qubit: The raw dataset for a single qubit.
        fit_qubit: The initial fit dataset from `peaks_dips`.
        node: The QualibrationNode.

    Returns:
        A tuple containing the final fit dataset and the fit parameters dictionary for the qubit.
    """
    signal_1d = ds_qubit.IQ_abs.values
    detuning_axis = ds_qubit.detuning.values

    # Find all dips by inverting the signal for the peak finder
    peaks, properties, _, _ = find_all_peaks_from_signal(-signal_1d)

    if len(peaks) == 0:
        return fit_qubit, None

    main_peak_idx = np.argmax(properties["prominences"])
    main_peak_pos = peaks[main_peak_idx]

    # Estimate width from scipy.signal.peak_widths and define a fit window
    width_samples = properties["widths"][main_peak_idx]
        
    fit_window_radius = int(width_samples * qc_params.fit_window_radius_multiple.value)
    fit_start = max(0, main_peak_pos - fit_window_radius)
    fit_end = min(len(signal_1d), main_peak_pos + fit_window_radius + 1)
    
    x_window = detuning_axis[fit_start:fit_end]
    y_window = signal_1d[fit_start:fit_end]

    # Guess initial fit parameters for a dip
    p0_amplitude = properties["prominences"][main_peak_idx]  # Positive amplitude for dip depth
    p0_center = detuning_axis[main_peak_pos]
    detuning_step = abs(detuning_axis[1] - detuning_axis[0])
    p0_width = width_samples * detuning_step
    # Use the mean of the window edges for a robust baseline guess
    p0_offset = (y_window[0] + y_window[-1]) / 2 if len(y_window) > 1 else np.mean(y_window)

    # Define bounds for the fit parameters to stabilize the fit
    lower_bounds = [
        0,                                          # Amplitude > 0
        x_window[0],                                # Center within window
        0,                                          # Width > 0
        np.min(signal_1d),                          # Offset within signal range
    ]
    upper_bounds = [
        (np.max(y_window) - np.min(y_window)) * 1.5, # Amplitude < 1.5x window range
        x_window[-1],                               # Center within window
        (x_window[-1] - x_window[0]),               # Width < window size
        np.max(signal_1d),                          # Offset within signal range
    ]
    bounds = (lower_bounds, upper_bounds)

    try:
        popt, _ = curve_fit(
            lorentzian_dip,
            x_window,
            y_window,
            p0=[p0_amplitude, p0_center, p0_width / 2, p0_offset], # HWHM for fit
            bounds=bounds,
            maxfev=qc_params.maxfev.value,
        )
        fit_amp, fit_center, fit_hwhm, fit_offset = popt
        fit_fwhm = abs(fit_hwhm * 2)

        # Update the fit dataset with new values
        fit_qubit["position"] = fit_center
        fit_qubit["width"] = fit_fwhm
        fit_qubit["amplitude"] = fit_amp # Amplitude is now positive
        
        # Create a new baseline array with the fitted offset
        new_baseline = xr.DataArray(
            np.full_like(fit_qubit.base_line.values, fit_offset),
            dims=fit_qubit.base_line.dims,
            coords=fit_qubit.base_line.coords,
        )
        fit_qubit["base_line"] = new_baseline
        
        return fit_qubit, None

    except RuntimeError:
        # If the advanced fit fails, we return the original `peaks_dips` result
        return fit_qubit, None


def _extract_and_validate_fit_parameters(
    ds: xr.Dataset,
    fit: xr.Dataset, 
    node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Extract fit parameters and validate their quality.
    
    Args:
        ds : xr.Dataset
            Raw dataset containing IQ data
        fit : xr.Dataset
            Dataset containing fit results from peak detection
        node : QualibrationNode
            Experiment node
        
    Returns:
        Tuple[xr.Dataset, Dict[str, FitParameters]]
            Validated fit data and results dictionary
    """
    # Calculate resonator frequencies and FWHM
    fit = _calculate_resonator_parameters(fit, node)
    
    # Evaluate fit quality for each qubit and determine outcomes
    fit_results = {}
    outcomes = []
    
    for qubit_name in fit.qubit.values:
        # Select the initial fit for the current qubit
        qubit_fit_initial = fit.sel(qubit=qubit_name)

        # If multiple peaks are detected, attempt a more robust fit.
        # The multi-peak fitter returns an updated version of the fit dataset for the qubit,
        # which we then update in the main 'fit' dataset.
        if qubit_fit_initial.raw_num_peaks.item() > 1:
            fit_qubit_updated, _ = _fit_multi_peak_resonator(
                ds.sel(qubit=qubit_name), qubit_fit_initial, node
            )
            fit.loc[dict(qubit=qubit_name)] = fit_qubit_updated

        # Now, using the final (potentially updated) fit data, extract metrics and determine the outcome.
        fit_metrics = _extract_qubit_fit_metrics(ds, fit, qubit_name)
        outcome = _determine_resonator_outcome(fit_metrics)
        outcomes.append(outcome)
        
        # Store the final results
        fit_results[qubit_name] = FitParameters(
            frequency=fit.sel(qubit=qubit_name).res_freq.values.item(),
            fwhm=fit.sel(qubit=qubit_name).fwhm.values.item(),
            outcome=outcome,
            fit_metrics=fit_metrics,
        )
    
    # Add outcomes to the fit dataset
    fit = fit.assign_coords(outcome=("qubit", outcomes))
    fit.outcome.attrs = {"long_name": "fit outcome", "units": ""}
    
    return fit, fit_results


def _calculate_resonator_parameters(fit: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """
    Calculate resonator frequency and FWHM from fit results.
    
    Args:
        fit : xr.Dataset
            Dataset with peak detection results
        node : QualibrationNode
            Experiment node
        
    Returns:
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


def _calculate_sweep_span(fit: xr.Dataset) -> float:
    """Calculate sweep span for relative comparisons."""
    if "detuning" in fit.dims:
        return float(fit.coords["detuning"].max() - fit.coords["detuning"].min())
    if "full_freq" in fit.dims:
        return float(fit.coords["full_freq"].max() - fit.coords["full_freq"].min())
    return 0.0


def _generate_lorentzian_fit(
    qubit_data: xr.Dataset, detuning_values: np.ndarray
) -> np.ndarray:
    """Generate the Lorentzian fit from qubit data."""
    return lorentzian_dip(
        detuning_values,
        float(qubit_data.amplitude.values),
        float(qubit_data.position.values),
        float(qubit_data.width.values) / 2,  # Convert to half-width at half-max
        float(qubit_data.base_line.mean().values),
    )


def _extract_qubit_fit_metrics(
    ds_raw: xr.Dataset, fit: xr.Dataset, qubit_name: str
) -> Dict[str, float]:
    """
    Extract all relevant fit metrics for a single qubit.

    Args:
        ds_raw : xr.Dataset
            The raw dataset containing IQ data
        fit : xr.Dataset
            Dataset containing fit results
        qubit_name : str
            Name of the qubit to extract metrics for

    Returns:
        Dict[str, float]
            Dictionary containing all fit metrics for the qubit
    """
    qubit_data = fit.sel(qubit=qubit_name)
    sweep_span = _calculate_sweep_span(fit)

    # Get raw data for quality metrics
    raw_data = ds_raw.IQ_abs.sel(qubit=qubit_name).values
    detuning_values = ds_raw.detuning.values

    # Generate the Lorentzian fit and calculate quality metrics
    fitted_data = _generate_lorentzian_fit(qubit_data, detuning_values)
    quality_metrics = calculate_quality_metrics(raw_data, fitted_data)

    return {
        "num_peaks": int(qubit_data.num_peaks.values),
        "raw_num_peaks": int(qubit_data.raw_num_peaks.values),
        "snr": float(qubit_data.snr.values),
        "fwhm": float(qubit_data.fwhm.values),
        "sweep_span": sweep_span,
        "asymmetry": float(qubit_data.asymmetry.values),
        "skewness": float(qubit_data.skewness.values),
        **quality_metrics,
    }


def _determine_resonator_outcome(
    metrics: Dict[str, float],
) -> str:
    """
    Determine the outcome for resonator spectroscopy based on fit metrics.
    
    Args:
        metrics : Dict[str, float]
            Dictionary containing fit metrics
        
    Returns:
        str
            Outcome description
    """
    num_peaks = metrics["num_peaks"]
    raw_num_peaks = metrics["raw_num_peaks"]
    snr = metrics["snr"]
    fwhm = metrics["fwhm"]
    sweep_span = metrics["sweep_span"]
    asymmetry = metrics["asymmetry"]
    skewness = metrics["skewness"]
    nrmse = metrics.get("nrmse", np.inf)
    
    # Check SNR first
    if snr < qc_params.min_snr.value:
        return "The SNR isn't large enough, consider increasing the number of shots"
    
    
    # Check number of peaks
    if num_peaks == 0:
        if snr < qc_params.min_snr.value:
            return (
                "The SNR isn't large enough, consider increasing the number of shots "
                "and ensure you are looking at the correct frequency range"
            )
        return "No peaks were detected, consider changing the frequency range"
    
    # Check peak shape quality and multiple resonances
    if _is_peak_shape_distorted(asymmetry, skewness, nrmse):
        if raw_num_peaks > 1:
            if nrmse <= qc_params.nrmse_threshold.value:
                return "successful"
            else:
                return "Multiple resonances detected, consider adjusting the span of the frequency range"
        return "The peak shape is distorted"
    
    # Check for peak width issues
    if _is_peak_too_wide(fwhm, sweep_span, snr):
        if snr < qc_params.snr_for_distortion.value:
            return "The SNR isn't large enough and the peak shape is distorted"
        else:
            return "Distorted peak detected"
    
    
    # All checks passed
    return "successful"


def _is_peak_shape_distorted(
    asymmetry: float, 
    skewness: float,
    nrmse: float,
) -> bool:
    """
    Check if peak shape indicates distortion based on asymmetry and skewness.
    
    Args:
        asymmetry : float
            Peak asymmetry value
        skewness : float
            Peak skewness value
        nrmse : float
            Normalized Root Mean Square Error
        
    Returns:
        bool
            True if peak shape is distorted
    """
    asymmetry_bad = (asymmetry is not None and 
                    (asymmetry < qc_params.min_asymmetry.value or asymmetry > qc_params.max_asymmetry.value))
    skewness_bad = (skewness is not None and abs(skewness) > qc_params.max_skewness.value)
    poor_geom_fit = nrmse > qc_params.nrmse_threshold.value

    return asymmetry_bad or skewness_bad or poor_geom_fit


def _is_peak_too_wide(
    fwhm: float,
    sweep_span: float,
    snr: float,
) -> bool:
    """
    Check if peak is too wide relative to sweep or absolutely.
    
    Args:
        fwhm : float
            Full width at half maximum
        sweep_span : float
            Total sweep span
        snr : float
            Signal-to-noise ratio
        
    Returns:
        bool
            True if peak is too wide
    """
    # Determine distortion threshold based on SNR
    distorted_fraction = (
        qc_params.distorted_fraction_low_snr.value
        if snr < qc_params.snr_for_distortion.value
        else qc_params.distorted_fraction_high_snr.value
    )
    
    # Check relative width
    relative_width_bad = (sweep_span > 0 and (fwhm / sweep_span > distorted_fraction))
    
    # Check absolute width
    absolute_width_bad = (fwhm > qc_params.fwhm_absolute_threshold_hz.value)
    
    return relative_width_bad or absolute_width_bad
