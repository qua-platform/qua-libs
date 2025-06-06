import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibration_libs.analysis import peaks_dips
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V

from qualibrate import QualibrationNode


@dataclass
class FitParameters:
    """Stores the relevant resonator spectroscopy experiment fit parameters for a single qubit."""
    frequency: float
    fwhm: float
    outcome: str


def log_fitted_results(fit_results: Dict, log_callable=None) -> None:
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_freq = f"\tResonator frequency: {1e-9 * fit_results[q]['frequency']:.3f} GHz | "
        s_fwhm = f"FWHM: {1e-3 * fit_results[q]['fwhm']:.1f} kHz | "
        if fit_results[q]["outcome"] == "successful":
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += f" FAIL! Reason: {fit_results[q]['outcome']}\n"
        log_callable(s_qubit + s_freq + s_fwhm)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    full_freq = np.array([ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Fit the T1 relaxation time for each qubit according to ``a * np.exp(t * decay) + offset``.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node : QualibrationNode
        The QUAlibrate node.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    """
    # Fit the resonator line
    fit_results = peaks_dips(ds.IQ_abs, "detuning")
    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(fit_results, node)
    return fit_data, fit_results


def get_fit_outcome(
    num_peaks: int,
    snr: float,
    fwhm: float,
    sweep_span: float,
    asymmetry: float,
    skewness: float,
    opx_bandwidth_artifact: bool,
    snr_min: float = 2.5,
    snr_distorted: float = 5.0,
    asymmetry_min: float = 0.4,
    asymmetry_max: float = 2.5,
    skewness_max: float = 1.5,
    distorted_fraction_low_snr: float = 0.2,
    distorted_fraction_high_snr: float = 0.3,
    fwhm_absolute_threshold: float = 1e6,
) -> str:
    """
    Returns the outcome string for a given fit result.

    Parameters:
    -----------
    num_peaks : int
        The number of peaks detected in the fit.
    snr : float
        The signal-to-noise ratio of the fit.
    fwhm : float
        The full width at half maximum of the fit.
    sweep_span : float
        The span of the sweep.
    asymmetry : float
        The asymmetry of the fit.
    skewness : float
        The skewness of the fit.
    opx_bandwidth_artifact : bool
        Whether the OPX bandwidth artifact is detected.
    snr_min : float
        The minimum signal-to-noise ratio for a successful fit.
    snr_distorted : float
        The signal-to-noise ratio for a distorted fit.
    asymmetry_min : float
        The minimum asymmetry for a successful fit.
    asymmetry_max : float
        The maximum asymmetry for a successful fit.
    skewness_max : float
        The maximum skewness for a successful fit.
    distorted_fraction_low_snr : float
        The distorted fraction for a low SNR.
    distorted_fraction_high_snr : float
        The distorted fraction for a high SNR.
    fwhm_absolute_threshold : float
        The absolute threshold for the FWHM.

    Returns:
    --------
    str
        The outcome string for the fit.
    """
    if snr < snr_min:
        return "The SNR isn't large enough, consider increasing the number of shots"
    if opx_bandwidth_artifact:
        return "OPX bandwidth artifact detected, check your experiment bandwidth settings"
    if num_peaks > 1:
        return "Several peaks were detected"
    if num_peaks == 0:
        if snr < snr_min:
            return "The SNR isn't large enough, consider increasing the number of shots and ensure you are looking at the correct frequency range"
        return "No peaks were detected, consider changing the frequency range"
    if (asymmetry is not None and (asymmetry < asymmetry_min or asymmetry > asymmetry_max)) or (skewness is not None and abs(skewness) > skewness_max):
        return "The peak shape is distorted"
    distorted_fraction = distorted_fraction_low_snr if snr < snr_distorted else distorted_fraction_high_snr
    if (sweep_span > 0 and (fwhm / sweep_span > distorted_fraction)) or (fwhm > fwhm_absolute_threshold):
        if snr < snr_distorted:
            return "The SNR isn't large enough and the peak shape is distorted"
        else:
            return "Distorted peak detected"
    return "successful"


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Add metadata to the dataset and fit results."""
    # Add metadata to fit results
    fit.attrs = {"long_name": "frequency", "units": "Hz"}
    # Get the fitted resonator frequency
    full_freq = np.array([q.resonator.RF_frequency for q in node.namespace["qubits"]])
    res_freq = fit.position + full_freq
    fit = fit.assign_coords(res_freq=("qubit", res_freq.data))
    fit.res_freq.attrs = {"long_name": "resonator frequency", "units": "Hz"}
    # Get the fitted FWHM
    fwhm = np.abs(fit.width)
    fit = fit.assign_coords(fwhm=("qubit", fwhm.data))
    fit.fwhm.attrs = {"long_name": "resonator fwhm", "units": "Hz"}

    fit_results: Dict[str, FitParameters] = {}
    outcomes = []
    for i, q in enumerate(fit.qubit.values):
        num_peaks = int(fit.sel(qubit=q).num_peaks.values)
        snr = float(fit.sel(qubit=q).snr.values)
        fwhm_val = float(fit.sel(qubit=q).fwhm.values)
        if 'detuning' in fit.dims:
            sweep_span = float(fit.coords['detuning'].max() - fit.coords['detuning'].min())
        elif 'full_freq' in fit.dims:
            sweep_span = float(fit.coords['full_freq'].max() - fit.coords['full_freq'].min())
        else:
            sweep_span = 0.0
        asymmetry = float(fit.sel(qubit=q).asymmetry.values)
        skewness = float(fit.sel(qubit=q).skewness.values)
        opx_bandwidth_artifact = not bool(fit.sel(qubit=q).opx_bandwidth_artifact.values)
        # Get the outcome string for this qubit
        outcome = get_fit_outcome(
            num_peaks,
            snr,
            fwhm_val,
            sweep_span,
            asymmetry,
            skewness,
            opx_bandwidth_artifact,
        )
        outcomes.append(outcome)
        fit_results[q] = FitParameters(
            frequency=fit.sel(qubit=q).res_freq.values.__float__(),
            fwhm=fwhm_val,
            outcome=outcome,
        )
    
    # Add outcomes to the fit dataset
    fit = fit.assign_coords(outcome=("qubit", outcomes))
    fit.outcome.attrs = {"long_name": "fit outcome", "units": ""}
    
    return fit, fit_results