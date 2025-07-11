from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibration_libs.analysis import fit_oscillation
from qualibration_libs.analysis.helpers.resonator_spectroscopy_vs_flux import (
    apply_frequency_correction_if_needed, calculate_flux_bias_parameters,
    calculate_flux_frequency_parameters, create_flux_fit_parameters,
    determine_flux_outcome, evaluate_qubit_data_quality,
    extract_outcome_parameters, refine_min_offset)
from qualibration_libs.analysis.parameters import analysis_config_manager
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V

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
    # Find minimum frequency points for each flux bias
    peak_freq = ds.IQ_abs.idxmin(dim="detuning")
    
    # Evaluate data quality for each qubit
    qubit_outcomes = evaluate_qubit_data_quality(ds, peak_freq)
    
    # Fit oscillation only for qubits with valid data, but keep all qubits for final processing
    peak_freq_for_fit = peak_freq.dropna(dim="flux_bias")
    
    if peak_freq_for_fit.qubit.size > 0:
        # Fit oscillation for valid qubits
        fit_results_da = fit_oscillation(peak_freq_for_fit, "flux_bias")
        fit_results_ds = xr.merge(
            [fit_results_da.rename("fit_results"), peak_freq.rename("peak_freq")]
        )
    else:
        # No qubits have valid data - create empty fit dataset with all qubits
        fit_results_ds = xr.Dataset({"peak_freq": peak_freq})
    
    # Extract and validate fit parameters for ALL qubits (including failed ones)
    fit_dataset, fit_results = _extract_and_validate_flux_parameters(
        fit_results_ds, node, qubit_outcomes
    )
    
    return fit_dataset, fit_results


def _extract_and_validate_flux_parameters(
    fit: xr.Dataset, node: QualibrationNode, qubit_outcomes: Dict[str, str]
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Extract and validate fit parameters from flux dependence data.

    Args:
        fit : xr.Dataset
            Dataset containing fit results
        node : QualibrationNode
            Experiment node
        qubit_outcomes : Dict[str, str]
            Initial quality assessment for each qubit

    Returns:
        Tuple[xr.Dataset, Dict[str, FitParameters]]
            Validated fit data and results dictionary
    """
    # Calculate flux bias parameters only for qubits that have fit_results
    if "fit_results" in fit.data_vars:
        fit = calculate_flux_bias_parameters(fit)
        fit = refine_min_offset(fit)
        rf_frequencies = np.array(
            [q.resonator.RF_frequency for q in node.namespace["qubits"]]
        )
        fit = calculate_flux_frequency_parameters(fit, rf_frequencies)

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
            # Extract parameters for outcome determination
            outcome_params = extract_outcome_parameters(
                fit,
                q,
                node.results["ds_raw"],
                node.parameters.frequency_span_in_mhz,
                qubit_outcomes,
            )

            # Determine outcome
            outcome = determine_flux_outcome(outcome_params, qubit_outcomes.get(q, "fit"))
            outcomes.append(outcome)

            # Apply frequency correction if needed
            base_freq = all_qubits[i].resonator.RF_frequency
            corrected_frequency = apply_frequency_correction_if_needed(
                fit, q, base_freq, node.results["ds_raw"], outcome_params
            )

            # Create fit parameters
            fit_results[q] = create_flux_fit_parameters(
                fit,
                q,
                node.parameters,
                all_qubits,
                outcome,
                corrected_frequency,
                FitParameters,
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
        fit = fit.assign_coords(
            outcome=(
                "qubit",
                [
                    outcomes[i]
                    for i, q_obj in enumerate(all_qubits)
                    if q_obj.name in fit.qubit.values
                ],
            )
        )

    return fit, fit_results