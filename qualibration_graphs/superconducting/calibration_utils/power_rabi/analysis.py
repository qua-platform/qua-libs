from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import xarray as xr
from qualibration_libs.analysis import fit_oscillation
from qualibration_libs.data import convert_IQ_to_V
from quam_config.instrument_limits import instrument_limits
from scipy.signal import correlate

from qualibrate import QualibrationNode

# Analysis Constants
MIN_FIT_QUALITY: float = 0.5
MIN_AMPLITUDE: float = 0.01
MAX_AMP_PREFACTOR: float = 10.0
SNR_MIN: float = 2.5
AUTOCORRELATION_R1_THRESHOLD: float = 0.8
AUTOCORRELATION_R2_THRESHOLD: float = 0.5
CHEVRON_MODULATION_THRESHOLD: float = 0.4


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit."""
    opt_amp_prefactor: float
    opt_amp: float
    operation: str
    outcome: str
    fit_quality: Optional[float] = None


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
            opt_amp = results.opt_amp
            opt_amp_prefactor = results.opt_amp_prefactor
            operation = results.operation
            fit_quality = results.fit_quality
        else:
            outcome = results.get('outcome', 'unknown')
            opt_amp = results.get('opt_amp', 0.0)
            opt_amp_prefactor = results.get('opt_amp_prefactor', 1.0)
            operation = results.get('operation', 'x180')
            fit_quality = results.get('fit_quality', None)
        
        if outcome == "successful":
            status_line += " SUCCESS!\n"
        else:
            status_line += f" FAIL! Reason: {outcome}\n"
        
        # Format amplitude with appropriate units
        amp_mV = opt_amp * 1e3
        amp_str = f"The calibrated {operation} amplitude: {amp_mV:.2f} mV "
        prefactor_str = f"(x{opt_amp_prefactor:.2f})\n"
        
        # Add fit quality information if available
        quality_str = ""
        if fit_quality is not None:
            quality_str = f"Fit quality (R²): {fit_quality:.3f}\n"
        
        log_callable(status_line + amp_str + prefactor_str + quality_str)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """
    Process raw dataset for power Rabi analysis.
    
    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset containing measurement data
    node : QualibrationNode
        The qualibration node containing parameters and qubit information
        
    Returns
    -------
    xr.Dataset
        Processed dataset with additional coordinates
    """
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    
    # Calculate full amplitude array for all qubits
    full_amp = np.array([
        ds.amp_prefactor * q.xy.operations[node.parameters.operation].amplitude 
        for q in node.namespace["qubits"]
    ])
    
    ds = ds.assign_coords(full_amp=(["qubit", "amp_prefactor"], full_amp))
    ds.full_amp.attrs = {"long_name": "pulse amplitude", "units": "V"}
    
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Fit the qubit power Rabi data for each qubit in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the raw data
    node : QualibrationNode
        The qualibration node containing experiment parameters

    Returns
    -------
    Tuple[xr.Dataset, Dict[str, FitParameters]]
        Tuple containing:
        - Dataset with fit results
        - Dictionary mapping qubit names to fit parameters
    """
    if node.parameters.max_number_pulses_per_sweep == 1:
        ds_fit = _fit_single_pulse_experiment(ds, node)
    else:
        ds_fit = _fit_multi_pulse_experiment(ds, node)

    # Extract and validate fit parameters
    fit_data, fit_results = _extract_and_validate_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _fit_single_pulse_experiment(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """
    Fit single pulse power Rabi experiment data.
    
    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset
    node : QualibrationNode
        Experiment node
        
    Returns
    -------
    xr.Dataset
        Dataset with oscillation fit results
    """
    ds_fit = ds.sel(nb_of_pulses=1)
    
    # Choose signal based on discrimination method
    signal_data = ds_fit.state if node.parameters.use_state_discrimination else ds_fit.I
    fit_vals = fit_oscillation(signal_data, "amp_prefactor")

    return xr.merge([ds, fit_vals.rename("fit")])


def _fit_multi_pulse_experiment(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """
    Fit multi-pulse power Rabi experiment data by finding optimal amplitude.
    
    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset
    node : QualibrationNode
        Experiment node
        
    Returns
    -------
    xr.Dataset
        Dataset with optimal amplitude prefactor
    """
    ds_fit = ds.copy()
    
    # Calculate mean across pulse dimension
    signal_data = ds.state if node.parameters.use_state_discrimination else ds.I
    ds_fit["data_mean"] = signal_data.mean(dim="nb_of_pulses")
    
    # Determine optimization direction based on pulse parity and operation
    first_pulse_even = (ds.nb_of_pulses.data[0] % 2 == 0)
    is_x180_operation = (node.parameters.operation == "x180")
    
    should_minimize = (first_pulse_even and is_x180_operation) or (not first_pulse_even and not is_x180_operation)
    
    if should_minimize:
        ds_fit["opt_amp_prefactor"] = ds_fit["data_mean"].idxmin(dim="amp_prefactor")
    else:
        ds_fit["opt_amp_prefactor"] = ds_fit["data_mean"].idxmax(dim="amp_prefactor")
    
    return ds_fit


def _extract_and_validate_fit_parameters(
    fit: xr.Dataset, 
    node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Extract fit parameters and validate their quality.
    
    Parameters
    ----------
    fit : xr.Dataset
        Dataset containing fit results
    node : QualibrationNode
        Experiment node
        
    Returns
    -------
    Tuple[xr.Dataset, Dict[str, FitParameters]]
        Validated fit data and results dictionary
    """
    # Get instrument limits for validation
    limits = [instrument_limits(q.xy) for q in node.namespace["qubits"]]
    
    # Process parameters based on experiment type
    if node.parameters.max_number_pulses_per_sweep == 1:
        fit = _calculate_single_pulse_parameters(fit, node)
    else:
        fit = _calculate_multi_pulse_parameters(fit, node)

    # Perform quality assessments
    fit = _assess_parameter_validity(fit)
    fit = _calculate_signal_quality_metrics(fit)
    fit = _evaluate_amplitude_constraints(fit, limits)
    fit = _determine_fit_outcomes(fit, node, limits)

    # Create results dictionary
    fit_results = _create_fit_results_dictionary(fit, node)
    
    return fit, fit_results


def _calculate_single_pulse_parameters(fit: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Calculate optimal amplitude parameters for single pulse experiments."""
    # Extract phase and correct for wrapping
    phase = fit.fit.sel(fit_vals="phi") - np.pi * (fit.fit.sel(fit_vals="phi") > np.pi / 2)
    
    # Calculate amplitude factor for π pulse
    factor = (np.pi - phase) / (2 * np.pi * fit.fit.sel(fit_vals="f"))
    fit = fit.assign({"opt_amp_prefactor": factor})
    fit.opt_amp_prefactor.attrs = {
        "long_name": "factor to get a pi pulse",
        "units": "Hz",
    }
    
    # Calculate optimal amplitude
    current_amps = xr.DataArray(
        [q.xy.operations[node.parameters.operation].amplitude for q in node.namespace["qubits"]],
        coords=dict(qubit=fit.qubit.data),
    )
    opt_amp = factor * current_amps
    fit = fit.assign({"opt_amp": opt_amp})
    fit.opt_amp.attrs = {"long_name": "x180 pulse amplitude", "units": "V"}
    
    return fit


def _calculate_multi_pulse_parameters(fit: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Calculate optimal amplitude parameters for multi-pulse experiments."""
    current_amps = xr.DataArray(
        [q.xy.operations[node.parameters.operation].amplitude for q in node.namespace["qubits"]],
        coords=dict(qubit=fit.qubit.data),
    )
    fit = fit.assign({"opt_amp": fit.opt_amp_prefactor * current_amps})
    fit.opt_amp.attrs = {
        "long_name": f"{node.parameters.operation} pulse amplitude",
        "units": "V",
    }
    return fit


def _assess_parameter_validity(fit: xr.Dataset) -> xr.Dataset:
    """Check for NaN values in fit parameters."""
    nan_success = ~(np.isnan(fit.opt_amp_prefactor) | np.isnan(fit.opt_amp))
    return fit.assign_coords(nan_success=("qubit", nan_success.data))


def _calculate_signal_quality_metrics(fit: xr.Dataset) -> xr.Dataset:
    """Calculate signal-to-noise ratio for each qubit."""
    snrs = []
    for q in fit.qubit.values:
        # Choose appropriate signal data
        if hasattr(fit, "I") and hasattr(fit.I, "sel"):
            signal = fit.I.sel(qubit=q).values
        elif hasattr(fit, "state") and hasattr(fit.state, "sel"):
            signal = fit.state.sel(qubit=q).values
        else:
            signal = np.zeros(2)
        
        snrs.append(_compute_signal_to_noise_ratio(signal))
    
    fit = fit.assign_coords(snr=("qubit", snrs))
    fit.snr.attrs = {"long_name": "signal-to-noise ratio", "units": ""}
    return fit


def _evaluate_amplitude_constraints(fit: xr.Dataset, limits: list) -> xr.Dataset:
    """Check if calculated amplitudes are within hardware limits."""
    # Use first limit as they should be consistent across qubits
    max_amp = limits[0].max_x180_wf_amplitude
    amp_success = fit.opt_amp < max_amp
    return fit.assign_coords(amp_success=("qubit", amp_success.data))


def _determine_fit_outcomes(fit: xr.Dataset, node: QualibrationNode, limits: list) -> xr.Dataset:
    """Determine analysis outcomes for all qubits based on quality checks."""
    # Check if signals are suitable for fitting
    signal_key = "state" if hasattr(fit, "state") else "I"
    fit_flags = _evaluate_signal_fit_worthiness(fit, signal_key)

    outcomes = []
    for i, q in enumerate(fit.qubit.values):
        # Extract fit quality if available
        fit_quality = _extract_fit_quality(fit)

        # Check for chevron modulation in 2D case
        has_structure = True
        if node.parameters.max_number_pulses_per_sweep > 1:
            signal = _extract_qubit_signal(fit, q)
            has_structure = _detect_chevron_modulation(signal)

        outcome = _determine_qubit_outcome(
            nan_success=bool(fit.nan_success.sel(qubit=q).values),
            amp_success=bool(fit.amp_success.sel(qubit=q).values),
            opt_amp=float(fit.sel(qubit=q).opt_amp.values),
            max_amplitude=float(limits[i].max_x180_wf_amplitude),
            fit_quality=fit_quality,
            amp_prefactor=float(fit.sel(qubit=q).opt_amp_prefactor.values),
            snr=float(fit.sel(qubit=q).snr.values),
            should_fit=fit_flags[q],
            is_1d_dataset=node.parameters.max_number_pulses_per_sweep == 1,
            has_structure=has_structure,
            fit=fit,
            qubit=q,
            node=node,
        )
        outcomes.append(outcome)
    
    fit = fit.assign_coords(outcome=("qubit", outcomes))
    fit.outcome.attrs = {"long_name": "fit outcome", "units": ""}
    return fit


def _create_fit_results_dictionary(fit: xr.Dataset, node: QualibrationNode) -> Dict[str, FitParameters]:
    """Create the final fit results dictionary."""
    fit_quality = _extract_fit_quality(fit)
    
    return {
        q: FitParameters(
            opt_amp_prefactor=fit.sel(qubit=q).opt_amp_prefactor.values.item(),
            opt_amp=fit.sel(qubit=q).opt_amp.values.item(),
            operation=node.parameters.operation,
            outcome=str(fit.sel(qubit=q).outcome.values),
            fit_quality=fit_quality,
        )
        for q in fit.qubit.values
    }


# Helper Functions for Signal Analysis

def _compute_signal_to_noise_ratio(signal: np.ndarray) -> float:
    """
    Compute signal-to-noise ratio for a 1D signal array.
    
    Parameters
    ----------
    signal : np.ndarray
        1D array of signal values
        
    Returns
    -------
    float
        Signal-to-noise ratio
    """
    if signal.size < 2:
        return 0.0
    
    try:
        from scipy.ndimage import uniform_filter1d

        # Smooth signal to estimate underlying trend
        smooth = uniform_filter1d(signal, size=max(3, signal.size // 10))
        # Estimate noise as residual
        noise = signal - smooth
        noise_std = np.std(noise)
        
        if noise_std == 0:
            return np.inf
        
        # Signal strength as peak-to-peak range
        signal_strength = np.ptp(signal)
        return signal_strength / noise_std
        
    except ImportError:
        # Fallback if scipy not available
        signal_std = np.std(signal)
        return np.ptp(signal) / signal_std if signal_std > 0 else np.inf


def _extract_fit_quality(fit: xr.Dataset) -> Optional[float]:
    """Safely extract R² fit quality from dataset."""
    try:
        if "fit" in fit and hasattr(fit.fit, "attrs") and "r_squared" in fit.fit.attrs:
            return float(fit.fit.attrs["r_squared"])
    except (AttributeError, KeyError, ValueError, TypeError):
        pass
    return None


def _extract_qubit_signal(fit: xr.Dataset, qubit: str) -> np.ndarray:
    """Extract signal data for a specific qubit."""
    if hasattr(fit, "I") and hasattr(fit.I, "sel"):
        return fit.I.sel(qubit=qubit).values
    elif hasattr(fit, "state") and hasattr(fit.state, "sel"):
        return fit.state.sel(qubit=qubit).values
    else:
        return np.zeros((2, 2))


def _evaluate_signal_fit_worthiness(ds: xr.Dataset, signal_key: str = "state") -> Dict[str, bool]:
    """
    Evaluate whether each qubit's signal is suitable for fitting based on autocorrelation.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing signal data
    signal_key : str
        Key for signal data ("state" or "I")
        
    Returns
    -------
    Dict[str, bool]
        Dictionary mapping qubit names to fit-worthiness
    """
    result = {}

    for q in ds.qubit.values:
        try:
            signal = ds[signal_key].sel(qubit=q).squeeze().values
            if signal.ndim == 2:
                signal = signal.mean(axis=0)

            # Check if signal has meaningful variation
            ptp = np.ptp(signal)
            if ptp == 0 or np.isnan(ptp):
                result[q] = False
                continue

            # Normalize signal
            signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-9)
            
            # Calculate autocorrelation
            autocorr = correlate(signal_norm, signal_norm, mode="full")
            autocorr = autocorr[autocorr.size // 2 :]  # positive lags only
            autocorr /= autocorr[0]  # normalize

            # Check autocorrelation at lag-1 and lag-2
            r1, r2 = autocorr[1], autocorr[2]
            result[q] = (r1 > AUTOCORRELATION_R1_THRESHOLD and 
                        r2 > AUTOCORRELATION_R2_THRESHOLD)

        except Exception:
            result[q] = False

    return result


def _detect_chevron_modulation(signal_2d: np.ndarray, threshold: float = CHEVRON_MODULATION_THRESHOLD) -> bool:
    """
    Detect chevron-like modulation in 2D signal data.
    
    Parameters
    ----------
    signal_2d : np.ndarray
        2D array of signal values
    threshold : float
        Minimum modulation threshold
        
    Returns
    -------
    bool
        True if chevron modulation detected
    """
    if signal_2d.ndim != 2:
        return False
    
    # Calculate peak-to-peak amplitude across frequency axis
    ptps = np.ptp(signal_2d, axis=0)
    modulation_range = np.ptp(ptps)
    avg_ptp = np.mean(ptps)
    
    # Check if modulation is significant
    return (modulation_range / (avg_ptp + 1e-12)) > threshold


def _determine_detuning_direction(fit: xr.Dataset, qubit: str, node=None) -> str:
    """
    Determine detuning direction from frequency information.
    
    Parameters
    ----------
    fit : xr.Dataset
        Fit dataset
    qubit : str
        Qubit identifier
    node : QualibrationNode, optional
        Experiment node
        
    Returns
    -------
    str
        Detuning direction ('positive' or 'negative')
    """
    try:
        # Try to get frequencies from node
        if node and hasattr(node, "namespace") and "qubits" in node.namespace:
            qubits_dict = node.namespace["qubits"]
            for qubit_obj in (qubits_dict.values() if isinstance(qubits_dict, dict) else qubits_dict):
                if hasattr(qubit_obj, "id") and getattr(qubit_obj, "id", None) == qubit:
                    drive_freq = getattr(qubit_obj.xy, "RF_frequency", None)
                    qubit_freq = getattr(qubit_obj, "f_01", None)
                    if drive_freq is not None and qubit_freq is not None:
                        detuning = drive_freq - qubit_freq
                        return "positive" if detuning > 0 else "negative"
        
        # Fallback: analyze fit parameters
        if "fit" in fit:
            fit_params = fit.fit.sel(qubit=qubit).values
            fit_vals = fit.fit.sel(qubit=qubit, fit_vals=slice(None)).coords["fit_vals"].values
            fit_dict = {k: v for k, v in zip(fit_vals, fit_params)}
            
            phase = float(fit_dict.get("phi", np.nan))
            freq = float(fit_dict.get("f", np.nan))
            
            if not (np.isnan(phase) or np.isnan(freq)):
                phase_norm = (phase + np.pi) % (2 * np.pi) - np.pi
                if abs(phase_norm) < (np.pi / 2):
                    return "positive" if freq > 0 else "negative"
                else:
                    return "positive" if phase_norm < 0 else "negative"
        
    except (AttributeError, KeyError, ValueError):
        pass
    
    return ""


# Main Outcome Determination Function

def _determine_qubit_outcome(
    nan_success: bool,
    amp_success: bool,
    opt_amp: float,
    max_amplitude: float,
    fit_quality: Optional[float] = None,
    min_fit_quality: float = MIN_FIT_QUALITY,
    min_amplitude: float = MIN_AMPLITUDE,
    max_amp_prefactor: float = MAX_AMP_PREFACTOR,
    amp_prefactor: Optional[float] = None,
    snr: Optional[float] = None,
    snr_min: float = SNR_MIN,
    should_fit: bool = True,
    is_1d_dataset: bool = True,
    has_structure: bool = True,
    fit: Optional[xr.Dataset] = None,
    qubit: Optional[str] = None,
    node: Optional[QualibrationNode] = None,
) -> str:
    """
    Determine the outcome for a single qubit based on comprehensive quality checks.
    
    Parameters
    ----------
    nan_success : bool
        Whether fit parameters are free of NaN values
    amp_success : bool
        Whether amplitude is within hardware limits
    opt_amp : float
        Optimal amplitude value
    max_amplitude : float
        Maximum allowed amplitude
    fit_quality : float, optional
        R² value of the fit
    min_fit_quality : float
        Minimum acceptable fit quality
    min_amplitude : float
        Minimum amplitude threshold
    max_amp_prefactor : float
        Maximum amplitude prefactor
    amp_prefactor : float, optional
        Amplitude prefactor value
    snr : float, optional
        Signal-to-noise ratio
    snr_min : float
        Minimum SNR threshold
    should_fit : bool
        Whether data quality is sufficient for fitting
    is_1d_dataset : bool
        Whether this is a 1D dataset
    has_structure : bool
        Whether data shows expected structure
    fit : xr.Dataset, optional
        Fit dataset for additional checks
    qubit : str, optional
        Qubit identifier
    node : QualibrationNode, optional
        Analysis node
        
    Returns
    -------
    str
        Outcome description
    """
    # Check for invalid fit parameters
    if not nan_success:
        return "Fit parameters are invalid (NaN values detected)"

    # Check for low amplitude with poor fit quality (1D only)
    if is_1d_dataset and opt_amp < min_amplitude:
        if fit_quality is not None and fit_quality < min_fit_quality:
            return (
                f"Poor fit quality (R² = {fit_quality:.3f} < {min_fit_quality}). "
                "There is too much noise in the data, consider increasing averaging or shot count"
            )
        else:
            return "There is too much noise in the data, consider increasing averaging or shot count"

    # Check for missing structure in 2D datasets
    if not has_structure and not is_1d_dataset:
        return "No chevron modulation detected. Please check the drive frequency and amplitude sweep range"

    # Check signal-to-noise ratio
    if snr is not None and snr < snr_min:
        return f"SNR too low (SNR = {snr:.2f} < {snr_min})"

    # Check amplitude hardware limits
    if not amp_success:
        detuning_direction = ""
        if fit is not None and qubit is not None:
            detuning_direction = _determine_detuning_direction(fit, qubit, node)
        
        direction_str = f"{detuning_direction} detuning" if detuning_direction else ""
        return f"The drive frequency is off-resonant with high {direction_str}, please adjust the drive frequency"

    # Check amplitude prefactor bounds
    if amp_prefactor is not None and abs(amp_prefactor) > max_amp_prefactor:
        return f"Amplitude prefactor too large (|{amp_prefactor:.2f}| > {max_amp_prefactor})"

    # Check if signal should be fit (1D only)
    if not should_fit and is_1d_dataset:
        return "There is too much noise in the data, consider increasing averaging or shot count"

    # Check fit quality
    if fit_quality is not None and fit_quality < min_fit_quality:
        # Special check for 1D datasets with too many oscillations
        if (is_1d_dataset and fit is not None and qubit is not None and 
            "fit" in fit and hasattr(fit.fit, "sel")):
            try:
                freq = float(fit.fit.sel(qubit=qubit, fit_vals="f").values)
                if freq > 1.0:
                    return "Fit quality poor due to large pulse amplitude range, consider a smaller range"
            except (ValueError, KeyError, AttributeError):
                pass
        
        return f"Poor fit quality (R² = {fit_quality:.3f} < {min_fit_quality})"

    return "successful"