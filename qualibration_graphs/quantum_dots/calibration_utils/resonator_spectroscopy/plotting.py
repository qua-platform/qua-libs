from typing import Dict, List

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.analysis import lorentzian_dip

from calibration_utils.common_utils.plot_style import apply_sensor_outcome_style, sensor_success

u = unit(coerce_to_integer=True)


def plot_all(ds_fit: xr.Dataset, sensors: List) -> Dict[str, Figure]:
    """Return all standard figures for 1D resonator spectroscopy."""
    return {
        "phase": plot_raw_phase(ds_fit, sensors),
        "amplitude": plot_raw_amplitude_with_fit(ds_fit, sensors),
    }


def plot_raw_phase(ds_fit: xr.Dataset, sensors: List) -> Figure:
    """Plot phase vs readout frequency for each sensor."""
    num_sensors = len(sensors)
    fig, axes = plt.subplots(1, num_sensors, figsize=(max(5 * num_sensors, 8), 4), squeeze=False)
    axes = axes.flatten()

    for ax, sensor in zip(axes, sensors):
        sensor_data = ds_fit.sel(sensor=sensor.name)
        success = sensor_success(ds_fit, sensor.name)

        ax.plot(sensor_data.full_freq / u.MHz, sensor_data.phase, "o-", markersize=2)
        ax.set_xlabel("RF frequency [MHz]")
        ax.set_ylabel("Phase [rad]")

        ax2 = ax.twiny()
        ax2.plot(sensor_data.frequency_detuning / u.MHz, sensor_data.phase, "o-", markersize=2, alpha=0)
        ax2.set_xlabel("Frequency detuning [MHz]")
        apply_sensor_outcome_style(ax, sensor.name, success)

    fig.suptitle("Resonator spectroscopy (phase)")
    fig.tight_layout()
    return fig


def plot_raw_amplitude_with_fit(ds_fit: xr.Dataset, sensors: List) -> Figure:
    """Plot |I + iQ| with Lorentzian fit overlay for each sensor."""
    num_sensors = len(sensors)
    fig, axes = plt.subplots(1, num_sensors, figsize=(max(5 * num_sensors, 8), 4), squeeze=False)
    axes = axes.flatten()

    for ax, sensor in zip(axes, sensors):
        plot_individual_amplitude_with_fit(
            ax,
            ds_fit.sel(sensor=sensor.name),
            sensor.name,
            ds_fit.sel(sensor=sensor.name),
            sensor_success(ds_fit, sensor.name),
        )

    fig.suptitle("Resonator spectroscopy (amplitude + fit)")
    fig.tight_layout()
    return fig


def plot_individual_amplitude_with_fit(
    ax: Axes,
    sensor_data: xr.Dataset,
    sensor_id: str,
    fit: xr.Dataset,
    success: bool | None,
):
    """Plot one sensor amplitude trace with optional Lorentzian fit and markers."""
    ax.plot(
        sensor_data.full_freq / u.MHz,
        sensor_data.IQ_abs / u.mV,
        "o-",
        markersize=2,
        label="Data",
    )
    ax.set_xlabel("RF frequency [MHz]")
    ax.set_ylabel(r"$R=\sqrt{I^2 + Q^2}$ [mV]")

    ax2 = ax.twiny()
    ax2.plot(
        sensor_data.frequency_detuning / u.MHz,
        sensor_data.IQ_abs / u.mV,
        "o-",
        markersize=2,
        alpha=0,
    )
    ax2.set_xlabel("Frequency detuning [MHz]")

    has_fit_vars = fit is not None and all(k in fit for k in ["amplitude", "position", "width", "base_line"])
    if has_fit_vars and success is not False:
        fitted_data = lorentzian_dip(
            sensor_data.frequency_detuning,
            float(fit.amplitude.values),
            float(fit.position.values),
            float(fit.width.values) / 2,
            float(fit.base_line.mean().values),
        )
        ax2.plot(
            sensor_data.frequency_detuning / u.MHz,
            fitted_data / u.mV,
            "r--",
            label="Fit",
        )

    if success and "frequency_shift" in fit.coords:
        shift_mhz = float(fit.frequency_shift.values) * 1e-6
        ax2.axvline(shift_mhz, color="blue", linestyle="--", label="frequency_shift")

    apply_sensor_outcome_style(ax, sensor_id, success)
    if success is not False and has_fit_vars:
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(handles + handles2, labels + labels2, loc="upper right", fontsize=8)

    ax.grid(True, alpha=0.3)
