import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit xarray Dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the fitted results for all qubits.
    log_callable : callable, optional
        Callable for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    pass


def _promote_numbered_stream_variables(ds: xr.Dataset, qubits) -> xr.Dataset:
    """Stack single-qubit ``I1``/``Q1``/``state1`` streams when fetcher left them ungrouped."""

    qubit_names = [q.name for q in qubits]
    promotions = {
        "I": "I1",
        "Q": "Q1",
        "state": "state1",
    }
    for target, source in promotions.items():
        if target in ds.data_vars or source not in ds.data_vars:
            continue
        promoted = ds[source]
        if "qubit" not in promoted.dims:
            promoted = promoted.expand_dims(qubit=qubit_names)
        ds = ds.assign({target: promoted})
    return ds


def _prepare_plottable_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Ensure the dataset exposes a variable that downstream analysis and plotting can use."""

    ds = _promote_numbered_stream_variables(ds, node.namespace["qubits"])
    has_state = "state" in ds.data_vars
    has_iq = "I" in ds.data_vars and "Q" in ds.data_vars

    if node.parameters.use_state_discrimination and has_state:
        return ds
    if has_iq:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
        return add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=False)
    if has_state:
        return ds

    raise ValueError(
        "Rabi chevron dataset must contain on-the-fly 'state' or demodulated 'I'/'Q' streams. "
        f"Found data variables: {list(ds.data_vars)}. "
        "Set use_state_discrimination=False for IQ-based readout during early bring-up."
    )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    ds = _prepare_plottable_dataset(ds, node)
    full_freq = np.array([ds.detuning + q.xy.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the qubit frequency and FWHM for each qubit in the dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node_parameters : Parameters
        Parameters related to the node, including whether state discrimination is used.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    """

    ds_fit = ds
    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""

    # Populate the FitParameters class with fitted values
    fit_results = {
        q: FitParameters(
            success=False,
        )
        for q in fit.qubit.values
    }
    return fit, fit_results
