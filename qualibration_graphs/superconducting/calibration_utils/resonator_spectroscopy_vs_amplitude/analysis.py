from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
from qualibration_libs.analysis import peaks_dips
from qualibration_libs.analysis.helpers.resonator_spectroscopy_vs_amplitude import (
    add_metrics_to_dataset, assign_optimal_power_clustering,
    calculate_comprehensive_fit_metrics,
    calculate_frequency_shifts_at_optimal_power, determine_amplitude_outcome,
    extract_amplitude_outcome_parameters,
    track_resonator_response_across_power)
from qualibration_libs.analysis.parameters import analysis_config_manager
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V

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
    ds_fit = ds.copy()
    qubits = node.namespace["qubits"]
    
    # Track resonator response and analyze quality
    tracking_results = track_resonator_response_across_power(ds, qubits)
    
    # Assign optimal power using clustering-based approach
    optimal_power_dict = assign_optimal_power_clustering(ds, qubits)
    
    # Calculate fit metrics and quality assessments
    fit_metrics = calculate_comprehensive_fit_metrics(ds, qubits, optimal_power_dict, tracking_results)
    
    # Add all calculated metrics to dataset
    ds_fit = add_metrics_to_dataset(ds_fit, tracking_results, optimal_power_dict, fit_metrics)
    
    # Calculate frequency shifts at optimal power
    freq_shifts = calculate_frequency_shifts_at_optimal_power(ds_fit, node)
    ds_fit = ds_fit.assign_coords({"freq_shift": (["qubit"], freq_shifts)})
    
    # Extract and validate fit parameters
    fit_dataset, fit_results = _extract_and_validate_amplitude_parameters(ds_fit, node, ds)
    
    return fit_dataset, fit_results


def _extract_and_validate_amplitude_parameters(
    fit: xr.Dataset, 
    node: QualibrationNode, 
    ds_raw: xr.Dataset
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Extract and validate fit parameters from amplitude dependence data.
    
    Args:
        fit : xr.Dataset
            Dataset containing fit results
        node : QualibrationNode
            Experiment node
        ds_raw : xr.Dataset
            Raw dataset for additional analysis
        
    Returns:
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
        outcome_params = extract_amplitude_outcome_parameters(fit, qubit_name, node, ds_raw)
        
        # Determine outcome
        outcome = determine_amplitude_outcome(outcome_params)
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