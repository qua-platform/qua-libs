import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate.core import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase
from qualibration_libs.analysis import peaks_dips


@dataclass
class FitParameters:
    """Fitted resonator spectroscopy results for a single sensor."""

    frequency: float
    """Absolute readout frequency at the resonance dip, in Hz."""

    fwhm: float
    """Lorentzian linewidth (FWHM of the |I + iQ| dip), in Hz."""

    success: bool
    """True if the fit is within the sweep span and safe to use for the state update."""


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log fitted results for all sensors.

    Parameters
    ----------
    fit_results : dict
        ``fit_results[sensor_name]`` mapping to the fitted values (Hz in storage).
    log_callable : callable, optional
        Logging function (typically ``node.log``). Defaults to the module logger.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for sensor_name, result in fit_results.items():
        if result["success"]:
            msg = (
                f"[{sensor_name}] SUCCESS | "
                f"frequency = {1e-9 * result['frequency']:.6f} GHz | "
                f"fwhm = {1e-3 * result['fwhm']:.1f} kHz"
            )
        else:
            msg = f"[{sensor_name}] FAIL | fit did not pass sanity checks"
        log_callable(msg)


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
