from typing import List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from calibration_utils.common_utils.plot_style import apply_sensor_outcome_style, sensor_success
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)


def plot_all(ds_fit: xr.Dataset, sensors: List) -> dict[str, Figure]:
    """Standard node plotting API."""
    figures: dict[str, Figure] = {}
    figures["phase"] = plot_raw_phase(ds_fit, sensors)
    figures["amplitude_gradient"] = plot_amplitude_with_fit(ds_fit, sensors, ds_fit)
    return figures


def plot_raw_amplitude(ds: xr.Dataset, sensors: List) -> Figure:
    """Plot IQ amplitude vs sensor bias offset for each sensor."""
    num_sensors = len(sensors)
    fig, axes = plt.subplots(1, num_sensors, figsize=(5 * num_sensors, 4), squeeze=False)
    axes = axes.flatten()

    for ax, sensor in zip(axes, sensors):
        sensor_data = ds.sel(sensor=sensor.name)
        plot_individual_raw_amplitude(
            ax,
            sensor_data,
            sensor.name,
            sensor_success(ds, sensor.name),
        )

    fig.suptitle("Sensor Gate Sweep - Amplitude")
    fig.tight_layout()
    return fig


def plot_raw_phase(ds: xr.Dataset, sensors: List) -> Figure:
    """Plot phase vs sensor bias offset for each sensor."""
    num_sensors = len(sensors)
    fig, axes = plt.subplots(1, num_sensors, figsize=(5 * num_sensors, 4), squeeze=False)
    axes = axes.flatten()

    for ax, sensor in zip(axes, sensors):
        sensor_data = ds.sel(sensor=sensor.name)
        ax.plot(sensor_data.bias_offsets, sensor_data.phase, "o-", markersize=2)
        ax.set_xlabel("Sensor bias offset [V]")
        ax.set_ylabel("Phase [rad]")
        apply_sensor_outcome_style(ax, sensor.name, sensor_success(ds, sensor.name))
        ax.grid(True, alpha=0.3)

    fig.suptitle("Sensor Gate Sweep - Phase")
    fig.tight_layout()
    return fig


def plot_amplitude_with_fit(ds: xr.Dataset, sensors: List, fits: xr.Dataset = None) -> Figure:
    """Plot the sensor gate sweep amplitude with Lorentzian fit and max-gradient point."""
    num_sensors = len(sensors)
    fig, axes = plt.subplots(1, num_sensors, figsize=(5 * num_sensors, 4), squeeze=False)
    axes = axes.flatten()

    for ax, sensor in zip(axes, sensors):
        sensor_data = ds.sel(sensor=sensor.name)
        fit_data = fits.sel(sensor=sensor.name) if fits is not None else None
        plot_individual_amplitude_with_fit(
            ax,
            sensor_data,
            sensor.name,
            fit_data,
            sensor_success(ds, sensor.name),
        )

    fig.suptitle("Sensor Gate Sweep - Amplitude + Lorentzian Fit")
    fig.tight_layout()
    return fig


def plot_individual_raw_amplitude(
    ax: Axes,
    sensor_data: xr.Dataset,
    sensor_id: str,
    success: bool | None = None,
):
    """Plot one sensor's raw IQ amplitude trace."""
    ax.plot(
        sensor_data.bias_offsets,
        sensor_data.amplitude / u.mV,
        "o-",
        markersize=2,
        label="Data",
    )
    ax.set_xlabel("Sensor bias offset [V]")
    ax.set_ylabel(r"$R=\sqrt{I^2 + Q^2}$ [mV]")
    apply_sensor_outcome_style(ax, sensor_id, success)
    ax.grid(True, alpha=0.3)


def plot_individual_amplitude_with_fit(
    ax: Axes,
    sensor_data: xr.Dataset,
    sensor_id: str,
    fit: xr.Dataset = None,
    success: bool | None = None,
):
    """Plot one sensor's IQ amplitude trace with Lorentzian fit overlays."""
    ax.plot(
        sensor_data.bias_offsets,
        sensor_data.amplitude / u.mV,
        "o-",
        markersize=2,
        label="Data",
    )
    ax.set_xlabel("Sensor bias offset [V]")
    ax.set_ylabel(r"$R=\sqrt{I^2 + Q^2}$ [mV]")
    apply_sensor_outcome_style(ax, sensor_id, success)

    if fit is not None:
        if "fitted_curve" in fit and np.any(np.isfinite(fit.fitted_curve.values)):
            ax.plot(
                sensor_data.bias_offsets,
                fit.fitted_curve.values / u.mV,
                "r-",
                lw=1.5,
                label="Lorentzian fit",
            )

        if "max_gradient_bias" in fit.coords and "fitted_curve" in fit:
            grad_bias = float(fit.max_gradient_bias.values)
            if np.isfinite(grad_bias):
                bias = sensor_data.bias_offsets.values
                idx = int(np.argmin(np.abs(bias - grad_bias)))
                grad_amp = fit.fitted_curve.values[idx]
                ax.plot(
                    grad_bias,
                    grad_amp / u.mV,
                    "o",
                    color="g",
                    markersize=10,
                    markeredgecolor="k",
                    zorder=5,
                    label=f"Max gradient @ {grad_bias:.4f} V",
                )

        ax.legend(fontsize=8)

    ax.grid(True, alpha=0.3)
