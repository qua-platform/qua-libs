from typing import Dict, List

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from calibration_utils.common_utils.plot_style import apply_sensor_outcome_style, sensor_success
from quam_builder.architecture.quantum_dots.components import SensorDot


def plot_all(ds_fit: xr.Dataset, sensors: List[SensorDot]) -> Dict[str, Figure]:
    """Standard node plotting API.

    ``ds_fit`` already contains the volts-converted ADC traces alongside the
    fitted delay/offset fields, so it doubles as both the "raw" and "fit" source
    for the two figures below.

    Parameters
    ----------
    ds_fit:
        Dataset produced by ``fit_raw_data``, containing ``adc``/``adc_single_run``
        plus the fitted ``delay``/``offset_mean`` fields.
    sensors:
        SensorDot list used for naming/ordering.

    Returns
    -------
    dict[str, Figure]
        ``"single_run"`` and ``"averaged_run"`` figures.
    """
    figures: Dict[str, Figure] = {}
    figures["single_run"] = plot_single_run_with_fit(ds_fit, sensors, ds_fit)
    figures["averaged_run"] = plot_averaged_run_with_fit(ds_fit, sensors, ds_fit)
    return figures


def plot_single_run_with_fit(ds: xr.Dataset, sensors: List[SensorDot], fits: xr.Dataset):
    """Plot the single-run ADC trace with fitted TOF and offset for each sensor."""
    num_sensors = len(sensors)
    fig, axes = plt.subplots(1, num_sensors, figsize=(5 * num_sensors, 4), squeeze=False)
    axes = axes.flatten()

    for ax, sensor in zip(axes, sensors):
        sensor_data = ds.sel(sensor=sensor.name)
        fit_data = fits.sel(sensor=sensor.name)
        plot_individual_single_run_with_fit(
            ax,
            sensor_data,
            sensor.name,
            fit_data,
            sensor_success(ds_fit=fits, sensor_name=sensor.name),
        )

    fig.suptitle("Single run")
    fig.tight_layout()
    return fig


def plot_averaged_run_with_fit(ds: xr.Dataset, sensors: List[SensorDot], fits: xr.Dataset):
    """Plot the averaged ADC trace with fitted TOF and offset for each sensor."""
    num_sensors = len(sensors)
    fig, axes = plt.subplots(1, num_sensors, figsize=(5 * num_sensors, 4), squeeze=False)
    axes = axes.flatten()

    for ax, sensor in zip(axes, sensors):
        sensor_data = ds.sel(sensor=sensor.name)
        fit_data = fits.sel(sensor=sensor.name)
        plot_individual_averaged_run_with_fit(
            ax,
            sensor_data,
            sensor.name,
            fit_data,
            sensor_success(ds_fit=fits, sensor_name=sensor.name),
        )

    fig.suptitle("Averaged run")
    fig.tight_layout()
    return fig


def plot_individual_single_run_with_fit(
    ax: Axes,
    sensor_data: xr.Dataset,
    sensor_name: str,
    fit: xr.Dataset = None,
    success: bool | None = None,
):
    """Plot one sensor's single-run ADC trace with optional fit overlays."""
    sensor_data.adc_single_run.plot(ax=ax, x="readout_time", label="ADC", color="b")

    if fit is not None:
        ax.axvline(fit.delay, color="k", linestyle="--", label="TOF")
        ax.axhline(fit.offset_mean, color="g", linestyle="--", label="Offset")

    ax.fill_between(
        range(sensor_data.sizes["readout_time"]),
        -0.5,
        0.5,
        color="grey",
        alpha=0.2,
        label="ADC Range",
    )
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Readout amplitude [V]")
    apply_sensor_outcome_style(ax, sensor_name, success)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_individual_averaged_run_with_fit(
    ax: Axes,
    sensor_data: xr.Dataset,
    sensor_name: str,
    fit: xr.Dataset = None,
    success: bool | None = None,
):
    """Plot one sensor's averaged ADC trace with optional fit overlays."""
    sensor_data.adc.plot(ax=ax, x="readout_time", label="ADC", color="b")

    if fit is not None:
        ax.axvline(fit.delay, color="k", linestyle="--", label="TOF")
        ax.axhline(fit.offset_mean, color="g", linestyle="--", label="Offset")

    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Readout amplitude [V]")
    apply_sensor_outcome_style(ax, sensor_name, success)
    ax.legend()
    ax.grid(True, alpha=0.3)
