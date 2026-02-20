import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase
from qualibration_libs.analysis import peaks_dips


@dataclass
class FitParameters:
    """Stores the relevant resonator spectroscopy experiment fit parameters for a single sensor"""

    frequency: float
    fwhm: float
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all sensors from the fit results
    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all sensors.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_sensor = f"Results for sensor {q}: "
        s_freq = f"\tResonator frequency: {1e-9 * fit_results[q]['frequency']:.3f} GHz | "
        s_fwhm = f"FWHM: {1e-3 * fit_results[q]['fwhm']:.1f} kHz | "
        if fit_results[q]["success"]:
            s_sensor += " SUCCESS!\n"
        else:
            s_sensor += " FAIL!\n"
        log_callable(s_sensor + s_freq + s_fwhm)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Process raw dataset to add amplitude and phase information."""
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    full_freq = np.array([ds.detuning + q.readout_resonator.intermediate_frequency for q in node.namespace["sensors"]])
    ds = ds.assign_coords(full_freq=(["sensors", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the resonator dip for each sensor and return the resonance frequency and FWHM.

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


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    full_freq = np.array([q.readout_resonator.intermediate_frequency for q in node.namespace["sensors"]])
    res_freq = fit.position + full_freq
    fit = fit.assign_coords(res_freq=("sensors", res_freq.data))
    fit.res_freq.attrs = {"long_name": "resonator frequency", "units": "Hz"}
    # Get the fitted FWHM
    fwhm = np.abs(fit.width)
    fit = fit.assign_coords(fwhm=("sensors", fwhm.data))
    fit.fwhm.attrs = {"long_name": "resonator fwhm", "units": "Hz"}
    # Assess whether the fit was successful or not
    freq_success = np.abs(res_freq.data) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    fwhm_success = np.abs(fwhm.data) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    success_criteria = freq_success & fwhm_success
    fit = fit.assign_coords(success=("sensors", success_criteria))

    fit_results = {
        q: FitParameters(
            frequency=fit.sel(sensors=q).res_freq.values.__float__(),
            fwhm=fit.sel(sensors=q).fwhm.values.__float__(),
            success=fit.sel(sensors=q).success.values.__bool__(),
        )
        for q in fit.sensors.values
    }
    return fit, fit_results
