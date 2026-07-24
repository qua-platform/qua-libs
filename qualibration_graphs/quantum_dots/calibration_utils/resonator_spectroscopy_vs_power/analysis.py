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
    """Fitted resonator spectroscopy vs power results for a single sensor (02b)."""

    success: bool
    """True if the fit passed sanity checks and is safe for the state update."""

    resonator_frequency: float
    """Absolute readout frequency at the optimal power, in Hz."""

    frequency_shift: float
    """Fitted readout frequency offset at the optimal power, in Hz."""

    optimal_power: float
    """Readout power just below the onset of frequency splitting, in dBm."""


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
                f"resonator_frequency = {1e-9 * result['resonator_frequency']:.6f} GHz | "
                f"frequency_shift = {1e-6 * result['frequency_shift']:.2f} MHz | "
                f"optimal_power = {result['optimal_power']:.2f} dBm"
            )
        else:
            msg = f"[{sensor_name}] FAIL | fit did not pass sanity checks"
        log_callable(msg)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Processes the raw dataset by converting the 'I' and 'Q' quadratures to V, or adding the intermediate_frequency as a coordinate for instance."""

    # Add the amplitude and phase to the raw dataset
    ds = add_amplitude_and_phase(ds, "frequency_detuning", subtract_slope_flag=True)
    # Add the RF frequency as a coordinate of the raw dataset
    full_freq = np.array(
        [
            ds.frequency_detuning + s.readout_resonator.intermediate_frequency
            for s in node.namespace["sensors"]
        ]
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
        Processed dataset with power-scan fit variables and summary coordinates added.
    """

    ds_fit = ds
    # Generate 1D dataset tracking the minimum IQ value, as a proxy for resonator frequency
    ds_fit["rr_min_response"] = ds.IQ_abs_norm.idxmin(dim="frequency_detuning")
    # Calculate the derivative along the power axis
    ds_fit["rr_min_response_diff"] = ds_fit.rr_min_response.differentiate(coord="power").dropna("power")
    ds_fit["rr_min_response_filtered"] = ds_fit.rr_min_response.where(np.abs(ds_fit["rr_min_response_diff"]) < 1e6)
    # Calculate the moving average of the derivative
    ds_fit["rr_min_response_avg"] = (
        ds_fit.rr_min_response_filtered.rolling(
            power=node.parameters.derivative_smoothing_window_num_points,
            center=True,  # window size in points
        )
        .mean()
        .dropna("power")
    )
    # ensure rr_min_response_avg buffer is writeable
    ds_fit["rr_min_response_avg"].data = ds_fit["rr_min_response_avg"].data.copy()
    # Apply a filter to scale down the initial noisy values in the moving average if needed
    for j in range(node.parameters.moving_average_filter_window_num_points):
        ds_fit.rr_min_response_avg.isel(power=j).data /= node.parameters.moving_average_filter_window_num_points - j
    # Find the first position where the moving average crosses below the threshold
    ds_fit["below_threshold"] = ds_fit.rr_min_response_avg < node.parameters.derivative_crossing_threshold_in_hz_per_dbm
    # Get the first occurrence below the derivative threshold
    optimal_power = ds_fit.below_threshold.idxmax(dim="power")
    optimal_power = optimal_power - node.parameters.buffer_from_crossing_threshold_in_dbm
    ds_fit = ds_fit.assign_coords({"optimal_power": (["sensor"], optimal_power.data)})

    # Define a function to fit the resonator line at the optimal power for each qubit
    def _select_optimal_power(ds, sensor):
        return peaks_dips(
            ds.sel(power=ds["optimal_power"].sel(sensor=sensor).data, method="nearest").sel(sensor=sensor).IQ_abs,
            "frequency_detuning",
        )

    # Get the resonance frequency shift at the optimal power
    fit_position = []
    for q in node.namespace["sensors"]:
        fit_at_power = _select_optimal_power(ds_fit, q.name)
        fit_position.append(float(fit_at_power.position.data))
    ds_fit = ds_fit.assign_coords(
        {
            "position": (["sensor"], fit_position),
            "frequency_shift": (["sensor"], fit_position),
        }
    )
    ds_fit.frequency_shift.attrs = {"long_name": "readout frequency offset from IF", "units": "Hz"}

    # Extract the relevant fitted parameters
    fit_dataset, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_dataset, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the fit dataset and fit result dictionary."""

    # Get the fitted resonator frequency
    full_freq = np.array([s.readout_resonator.intermediate_frequency for s in node.namespace["sensors"]])
    res_freq = fit.frequency_shift + full_freq
    fit = fit.assign_coords(res_freq=("sensor", res_freq.data))
    fit.res_freq.attrs = {"long_name": "resonator frequency", "units": "Hz"}
    # Assess whether the fit was successful or not
    freq_success = np.abs(fit.frequency_shift.data) < node.parameters.frequency_span_in_mhz * 1e6
    nan_success = np.isnan(fit.frequency_shift.data) | np.isnan(fit.optimal_power.data)
    success_criteria = freq_success & ~nan_success
    fit = fit.assign_coords(success=("sensor", success_criteria))

    fit_results = {
        s: FitParameters(
            success=fit.sel(sensor=s).success.values.__bool__(),
            resonator_frequency=float(fit.res_freq.sel(sensor=s).values),
            frequency_shift=float(fit.frequency_shift.sel(sensor=s).data),
            optimal_power=float(fit.optimal_power.sel(sensor=s).data),
        )
        for s in fit.sensor.values
    }

    return fit, fit_results
