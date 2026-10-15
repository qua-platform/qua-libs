import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr

from qualibrate.core import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase
from qualibration_libs.analysis import peaks_dips

from calibration_utils.common_utils.helpers import fmt_hz


@dataclass
class FitParameters:
    """Fitted resonator spectroscopy vs power results for a single sensor (02b)."""

    success: bool
    """True if the fit passed sanity checks and is safe for the state update."""

    resonator_frequency: float
    """Absolute readout frequency at the optimal power, in Hz."""

    frequency_shift: float
    """Fitted readout frequency offset at the optimal power, in Hz."""

    optimal_power: float
    """Readout power just below the onset of frequency splitting, in dBm."""

    failure_reason: Optional[str] = None
    """Human-readable explanation when ``success`` is False."""


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
                f"optimal_power = {result['optimal_power']:.2f} dBm"
            )
        else:
            reason = result.get("failure_reason") or "fit did not pass sanity checks"
            msg = f"[{sensor_name}] FAIL | {reason}"
        log_callable(msg)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Processes the raw dataset by converting the 'I' and 'Q' quadratures to V, or adding the intermediate_frequency as a coordinate for instance."""

    # Add the amplitude and phase to the raw dataset
    ds = add_amplitude_and_phase(ds, "frequency_detuning", subtract_slope_flag=True)
    # Add the RF frequency as a coordinate of the raw dataset
    full_freq = np.array(
        [ds.frequency_detuning + s.readout_resonator.intermediate_frequency for s in node.namespace["sensors"]]
    )
    ds = ds.assign_coords(full_freq=(["sensor", "frequency_detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    # Normalize the IQ_abs with respect to the amplitude axis
    ds = ds.assign({"IQ_abs_norm": ds["IQ_abs"] / ds.IQ_abs.mean(dim=["frequency_detuning"])})
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Find the optimal readout power and fit the resonator line at that power for each sensor.

    Parameters:
    -----------
    ds : xr.Dataset
        Processed dataset containing amplitude and phase information.
    node : QualibrationNode
        The QUAlibrate node.

    Returns:
    --------
    xr.Dataset
        Processed dataset with optimal-power summary coordinates added.
    """
    optimal_power = np.full(len(ds.sensor), np.nan, dtype=float)
    frequency_shift = np.full(len(ds.sensor), np.nan, dtype=float)
    failure_reasons: list[Optional[str]] = [None] * len(ds.sensor)

    for i, sensor in enumerate(node.namespace["sensors"]):
        sensor_data = ds.sel(sensor=sensor.name)
        opt_power, find_reason = _find_optimal_power(sensor_data.IQ_abs_norm, node)
        if find_reason is not None:
            failure_reasons[i] = find_reason
            continue

        optimal_power[i] = opt_power
        shift, fit_reason = _fit_frequency_shift_at_power(sensor_data, opt_power)
        frequency_shift[i] = shift
        if fit_reason is not None:
            failure_reasons[i] = fit_reason
            optimal_power[i] = np.nan

    ds_fit = ds.assign_coords(
        {
            "optimal_power": ("sensor", optimal_power),
            "frequency_shift": ("sensor", frequency_shift),
        }
    )
    ds_fit.frequency_shift.attrs = {"long_name": "readout frequency offset from IF", "units": "Hz"}

    fit_dataset, fit_results = _extract_relevant_fit_parameters(ds_fit, node, failure_reasons)
    return fit_dataset, fit_results


def _find_optimal_power(iq_abs_norm: xr.DataArray, node: QualibrationNode) -> Tuple[float, Optional[str]]:
    """Track the resonance dip vs power and locate the optimal-power crossing."""
    min_power_points = node.parameters.moving_average_filter_window_num_points
    if iq_abs_norm.sizes.get("power", 0) < min_power_points:
        return np.nan, "insufficient power points for analysis"

    rr_min_response = iq_abs_norm.idxmin(dim="frequency_detuning")
    rr_min_response_diff = rr_min_response.differentiate(coord="power").dropna("power")
    if rr_min_response_diff.sizes.get("power", 0) == 0:
        return np.nan, "no resonator dip track vs power (flat noise or missing resonance)"

    rr_min_response_filtered = rr_min_response.where(np.abs(rr_min_response_diff) < 1e6)
    rr_min_response_avg = (
        rr_min_response_filtered.rolling(
            power=node.parameters.derivative_smoothing_window_num_points,
            center=True,
        )
        .mean()
        .dropna("power")
    )
    if rr_min_response_avg.sizes.get("power", 0) < min_power_points:
        return np.nan, "no resonator dip track vs power after smoothing (noise-only data?)"

    rr_min_response_avg = rr_min_response_avg.copy(deep=True)
    n_power = rr_min_response_avg.sizes["power"]
    for j in range(min(min_power_points, n_power)):
        rr_min_response_avg.data[j] /= min_power_points - j

    below_threshold = rr_min_response_avg < node.parameters.derivative_crossing_threshold_in_hz_per_dbm
    if not bool(below_threshold.any()):
        return np.nan, "no power-splitting crossing found (resonator dip not resolved vs power)"

    crossing_power = float(below_threshold.idxmax(dim="power").values)
    optimal_power = crossing_power - node.parameters.buffer_from_crossing_threshold_in_dbm
    if not np.isfinite(optimal_power):
        return np.nan, "optimal readout power could not be determined"

    return optimal_power, None


def _fit_frequency_shift_at_power(
    sensor_data: xr.Dataset,
    optimal_power: float,
) -> Tuple[float, Optional[str]]:
    """Fit the resonator dip at the chosen readout power."""
    try:
        iq_at_power = sensor_data.sel(power=optimal_power, method="nearest").IQ_abs
        if iq_at_power.sizes.get("frequency_detuning", 0) < 3:
            return np.nan, "insufficient frequency points at optimal power"
        fit_at_power = peaks_dips(iq_at_power, "frequency_detuning")
        shift = float(fit_at_power.position.data)
        if not np.isfinite(shift):
            return np.nan, "no resonator dip found at optimal power"
        return shift, None
    except (ValueError, KeyError, IndexError):
        return np.nan, "no resonator dip found at optimal power"


def _extract_relevant_fit_parameters(
    fit: xr.Dataset,
    node: QualibrationNode,
    failure_reasons: list[Optional[str]],
):
    """Add metadata to the fit dataset and fit result dictionary."""
    full_freq = np.array([s.readout_resonator.intermediate_frequency for s in node.namespace["sensors"]])
    res_freq = fit.frequency_shift + full_freq
    fit = fit.assign_coords(res_freq=("sensor", res_freq.data))
    fit.res_freq.attrs = {"long_name": "resonator frequency", "units": "Hz"}

    span_hz = node.parameters.frequency_span_in_mhz * 1e6
    freq_success = np.abs(fit.frequency_shift.data) < span_hz
    finite_success = np.isfinite(fit.frequency_shift.data) & np.isfinite(fit.optimal_power.data)
    success_criteria = freq_success & finite_success
    fit = fit.assign_coords(success=("sensor", success_criteria))

    fit_results = {}
    for i, sensor_name in enumerate(fit.sensor.values):
        sensor_success = bool(fit.sel(sensor=sensor_name).success.values)
        reason = failure_reasons[i]
        if sensor_success:
            resolved_reason = None
        elif reason is not None:
            resolved_reason = reason
        elif not finite_success[i]:
            resolved_reason = "optimal power or frequency shift is NaN"
        elif not freq_success[i]:
            resolved_reason = "frequency shift outside sweep span"
        else:
            resolved_reason = "fit did not pass sanity checks"

        fit_results[sensor_name] = FitParameters(
            success=sensor_success,
            resonator_frequency=float(fit.res_freq.sel(sensor=sensor_name).values),
            frequency_shift=float(fit.frequency_shift.sel(sensor=sensor_name).data),
            optimal_power=float(fit.optimal_power.sel(sensor=sensor_name).data),
            failure_reason=resolved_reason,
        )

    return fit, fit_results
