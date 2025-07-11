from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import xarray as xr
from qualibration_libs.analysis import peaks_dips
from qualibration_libs.analysis.helpers.resonator_spectroscopy import (
    calculate_resonator_parameters, determine_resonator_outcome,
    extract_qubit_fit_metrics, fit_multi_peak_resonator)
from qualibration_libs.analysis.parameters import analysis_config_manager
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V

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


def _extract_and_validate_fit_parameters(
    ds: xr.Dataset,
    fit: xr.Dataset,
    node: QualibrationNode,
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
    rf_frequencies = np.array(
        [q.resonator.RF_frequency for q in node.namespace["qubits"]]
    )
    fit = calculate_resonator_parameters(fit, rf_frequencies)

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
            fit_qubit_updated, _ = fit_multi_peak_resonator(
                ds.sel(qubit=qubit_name), qubit_fit_initial
            )
            fit.loc[dict(qubit=qubit_name)] = fit_qubit_updated

        # Now, using the final (potentially updated) fit data, extract metrics and determine the outcome.
        fit_metrics = extract_qubit_fit_metrics(ds, fit, qubit_name)
        outcome = determine_resonator_outcome(fit_metrics)
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
