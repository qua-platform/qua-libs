from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import xarray as xr
from qualibration_libs.analysis.helpers.power_rabi import (
    assess_parameter_validity, calculate_multi_pulse_parameters,
    calculate_signal_quality_metrics, calculate_single_pulse_parameters,
    create_fit_results_dictionary, determine_fit_outcomes,
    evaluate_amplitude_constraints, fit_multi_pulse_experiment,
    fit_single_pulse_experiment)
from qualibration_libs.analysis.parameters import analysis_config_manager
from qualibration_libs.data import convert_IQ_to_V
from quam_config.instrument_limits import instrument_limits

from qualibrate import QualibrationNode

# Access parameter groups from the config manager
qc_params = analysis_config_manager.get("power_rabi_qc")


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit."""
    opt_amp_prefactor: float
    opt_amp: float
    operation: str
    outcome: str
    fit_quality: Optional[float] = None


def log_fitted_results(fit_results: Dict[str, Union[FitParameters, dict]], log_callable=None) -> None:
    """
    Logs the node-specific fitted results for all qubits from the fit results.
    It handles both FitParameters objects and dictionaries to support loaded data from older versions.

    Args:
        fit_results : Dict[str, Union[FitParameters, dict]]
            Dictionary containing the fitted results for all qubits. Can be objects or dicts.
        log_callable : callable, optional
            Logger for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qubit_name, results in fit_results.items():
        if isinstance(results, dict):
            # Convert dict to FitParameters object for consistent handling
            results = FitParameters(
                outcome=results.get("outcome", "unknown"),
                opt_amp=results.get("opt_amp", 0.0),
                opt_amp_prefactor=results.get("opt_amp_prefactor", 1.0),
                operation=results.get("operation", "x180"),
                fit_quality=results.get("fit_quality", None),
            )

        status_line = f"Results for qubit {qubit_name}: "

        if results.outcome == "successful":
            status_line += " SUCCESS!\n"
        else:
            status_line += f" FAIL! Reason: {results.outcome}\n"

        # Format amplitude with appropriate units
        amp_mV = results.opt_amp * 1e3
        amp_str = f"The calibrated {results.operation} amplitude: {amp_mV:.2f} mV "
        prefactor_str = f"(x{results.opt_amp_prefactor:.2f})\n"

        log_callable(status_line + amp_str + prefactor_str)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """
    Process raw dataset for power Rabi analysis.
    
    Args:
        ds : xr.Dataset
            Raw dataset containing measurement data
        node : QualibrationNode
            The qualibration node containing parameters and qubit information
        
    Returns:
        xr.Dataset
            Processed dataset with additional coordinates
    """
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])

    # Calculate full amplitude array for all qubits
    full_amp = np.array(
        [
            ds.amp_prefactor * q.xy.operations[node.parameters.operation].amplitude
            for q in node.namespace["qubits"]
        ]
    )

    ds = ds.assign_coords(full_amp=(["qubit", "amp_prefactor"], full_amp))
    ds.full_amp.attrs = {"long_name": "pulse amplitude", "units": "V"}

    return ds


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Fit the qubit power Rabi data for each qubit in the dataset.

    Args:
        ds : xr.Dataset
            Dataset containing the raw data
        node : QualibrationNode
            The qualibration node containing experiment parameters

    Returns:
        Tuple[xr.Dataset, Dict[str, FitParameters]]
            Tuple containing:
            - Dataset with fit results
            - Dictionary mapping qubit names to fit parameters
    """
    if node.parameters.max_number_pulses_per_sweep == 1:
        ds_fit = fit_single_pulse_experiment(ds, node.parameters.use_state_discrimination)
    else:
        ds_fit = fit_multi_pulse_experiment(
            ds, node.parameters.use_state_discrimination, node.parameters.operation
        )

    # Extract and validate fit parameters
    fit_data, fit_results = _extract_and_validate_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _extract_and_validate_fit_parameters(
    fit: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Extract fit parameters and validate their quality.

    Args:
        fit : xr.Dataset
            Dataset containing fit results
        node : QualibrationNode
            Experiment node

    Returns:
        Tuple[xr.Dataset, Dict[str, FitParameters]]
            Validated fit data and results dictionary
    """
    # Get instrument limits for validation
    limits = [instrument_limits(q.xy) for q in node.namespace["qubits"]]
    qubits = node.namespace["qubits"]

    # Process parameters based on experiment type
    if node.parameters.max_number_pulses_per_sweep == 1:
        fit = calculate_single_pulse_parameters(fit, node.parameters.operation, qubits)
    else:
        fit = calculate_multi_pulse_parameters(fit, node.parameters.operation, qubits)

    # Perform quality assessments
    fit = assess_parameter_validity(fit)
    fit = calculate_signal_quality_metrics(fit)
    fit = evaluate_amplitude_constraints(fit, limits)
    fit = determine_fit_outcomes(
        fit, node.parameters.max_number_pulses_per_sweep, limits, node
    )

    # Create results dictionary
    fit_results = create_fit_results_dictionary(
        fit, node.parameters.operation, FitParameters
    )

    return fit, fit_results