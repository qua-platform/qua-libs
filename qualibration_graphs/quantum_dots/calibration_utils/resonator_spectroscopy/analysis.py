import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate.core import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase
from qualibration_libs.analysis import peaks_dips

from calibration_utils.common_utils.helpers import fmt_hz


@dataclass
class FitParameters:
    """Fitted resonator spectroscopy results for a single sensor (02a)."""

    success: bool
    """True if the fit passed sanity checks and is safe for the state update."""

    resonator_frequency: float
    """Absolute readout frequency at the resonance dip, in Hz."""

    frequency_shift: float
    """Fitted readout frequency offset from IF, in Hz."""

    fwhm: float
    """Lorentzian linewidth (FWHM of the |I + iQ| dip), in Hz."""


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log fitted results for all sensors.

    Parameters
    ----------
    fit_results : dict
        ``fit_results[sensor_name]`` mapping to the fitted values.
    log_callable : callable, optional
        Logging function (typically ``node.log``). Defaults to the module logger.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for sensor_name, result in fit_results.items():
        if result["success"]:
            msg = (
                f"[{sensor_name}] SUCCESS | "
                f"resonator_frequency = {fmt_hz(result['resonator_frequency'])} | "
                f"frequency_shift = {fmt_hz(result['frequency_shift'])} | "
                f"fwhm = {fmt_hz(result['fwhm'])}"
            )
        else:
            msg = f"[{sensor_name}] FAIL | fit did not pass sanity checks"
        log_callable(msg)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Process raw dataset to add amplitude and phase information."""
    ds = add_amplitude_and_phase(ds, "frequency_detuning", subtract_slope_flag=True)
    full_freq = np.array(
        [ds.frequency_detuning + q.readout_resonator.intermediate_frequency for q in node.namespace["sensors"]]
    )
    ds = ds.assign_coords(full_freq=(["sensor", "frequency_detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the resonator dip for each sensor and return the processed dataset with fit outputs.

    Parameters:
    -----------
    ds : xr.Dataset
        Processed dataset containing amplitude and phase information.
    node : QualibrationNode
        The QUAlibrate node.

    Returns:
    --------
    xr.Dataset
        Processed dataset with Lorentzian fit variables and summary coordinates added.
    """
    fit_vars = peaks_dips(ds.IQ_abs, "frequency_detuning")
    ds_fit = xr.merge([ds, fit_vars])
    ds_fit, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return ds_fit, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    full_freq = np.array([q.readout_resonator.intermediate_frequency for q in node.namespace["sensors"]])
    fitted_frequency_shift = fit.position.data
    res_freq = fitted_frequency_shift + full_freq
    fit = fit.assign_coords(res_freq=("sensor", res_freq))
    fit.res_freq.attrs = {"long_name": "resonator frequency", "units": "Hz"}
    fit = fit.assign_coords(frequency_shift=("sensor", fitted_frequency_shift))
    fit.frequency_shift.attrs = {"long_name": "readout frequency offset from IF", "units": "Hz"}
    # Get the fitted FWHM
    fwhm = np.abs(fit.width)
    fit = fit.assign_coords(fwhm=("sensor", fwhm.data))
    fit.fwhm.attrs = {"long_name": "resonator fwhm", "units": "Hz"}
    # Assess whether the fit was successful or not
    freq_success = np.abs(res_freq.data) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    fwhm_success = np.abs(fwhm.data) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    success_criteria = freq_success & fwhm_success
    fit = fit.assign_coords(success=("sensor", success_criteria))

    fit_results = {
        s: FitParameters(
            success=fit.sel(sensor=s).success.values.__bool__(),
            resonator_frequency=fit.sel(sensor=s).res_freq.values.__float__(),
            frequency_shift=fit.sel(sensor=s).frequency_shift.values.__float__(),
            fwhm=fit.sel(sensor=s).fwhm.values.__float__(),
        )
        for s in fit.sensor.values
    }
    return fit, fit_results
