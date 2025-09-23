import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from scipy.fft import fft

from qualibrate import QualibrationNode

@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    success: bool
    interaction_max: int
    coupler_flux_pulse: float
    coupler_flux_min:float

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


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """
    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the processed data.
    node : QualibrationNode
        The calibration node containing parameters and qubit pairs.

    Returns:
    --------
    Tuple[xr.Dataset, Dict[str, FitResults]]
        Dataset with fit results and dictionary of fit results for each qubit pair.
    """
    qubit_pairs = [node.machine.qubit_pairs[pair] for pair in node.parameters.qubit_pairs]
    fluxes_coupler = ds.flux_coupler.values

    ds = ds.assign_coords(idle_time = ds.idle_time * 4)
    if node.parameters.use_state_discrimination:
        ds = ds.assign({"res_sum" : ds.state_control - ds.state_target})
    else:
        ds = ds.assign({"res_sum" : ds.I_control - ds.I_target})
    flux_coupler_full = np.array([fluxes_coupler + qp.coupler.decouple_offset for qp in qubit_pairs])
    ds = ds.assign_coords({"flux_coupler_full": (["qubit_pair", "flux_coupler"], flux_coupler_full)})

    # Add the dominant frequencies to the dataset
    ds['dominant_frequency'] = extract_dominant_frequencies(ds.res_sum)
    ds.dominant_frequency.attrs['units'] = 'GHz'

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

    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""

    # Populate the FitParameters class with fitted values
    fit_results = {}
    for qp in node.machine.qubit_pairs.values():
        target_gate_time = 50  # ns (fallback)
        if node.parameters.cz_or_iswap  == "cz" and "Cz" in qp.macros:
            target_gate_time = qp.macros["Cz"].coupler_flux_pulse.length
        max_frequency = 1 / (2 * target_gate_time)  # in GHz
        interaction_max = (fit.dominant_frequency * (fit.dominant_frequency < max_frequency)).max(dim='flux_coupler')
        coupler_flux_pulse = fit.flux_coupler.isel(flux_coupler=(fit.dominant_frequency * (fit.dominant_frequency < max_frequency)).argmax(dim='flux_coupler'))
        coupler_flux_min = fit.flux_coupler_full.isel(flux_coupler=fit.dominant_frequency.argmin(dim='flux_coupler'))

        fit_results[qp.name] = FitParameters(
            success=True,
            interaction_max= int(interaction_max.values),
            coupler_flux_pulse = float(coupler_flux_pulse.values),
            coupler_flux_min=float(coupler_flux_min.values)
        )
    return fit, fit_results


def extract_dominant_frequencies(da, dim="idle_time"):
    def extract_dominant_frequency(signal, sample_rate):
        fft_result = fft(signal)
        frequencies = np.fft.fftfreq(len(signal), 1 / sample_rate)
        positive_freq_idx = np.where(frequencies > 0)
        dominant_idx = np.argmax(np.abs(fft_result[positive_freq_idx]))
        return frequencies[positive_freq_idx][dominant_idx]

    def extract_dominant_frequency_wrapper(signal):
        sample_rate = 1 / (da.coords[dim][1].values - da.coords[dim][0].values)  # Assuming uniform sampling
        return extract_dominant_frequency(signal, sample_rate)

    dominant_frequencies = xr.apply_ufunc(
        extract_dominant_frequency_wrapper, da, input_core_dims=[[dim]], output_core_dims=[[]], vectorize=True
    )

    return dominant_frequencies