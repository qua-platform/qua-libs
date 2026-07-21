from typing import List
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from qualang_tools.units import unit

u = unit(coerce_to_integer=True)


def plot_raw_amplitude(ds: xr.Dataset, sensors: List) -> Figure:
    """
    Plots the raw amplitude data for the sensor gate sweep.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the I and Q quadrature data.
    sensors : list
        A list of sensors to plot.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each sensor.
    - Each subplot shows IQ amplitude vs sensor bias offset.
    """
    num_sensors = len(sensors)

    fig, axes = plt.subplots(
        1, num_sensors, figsize=(5 * num_sensors, 4), squeeze=False
    )
    axes = axes.flatten()

    for ax, sensor in zip(axes, sensors):
        sensor_data = ds.sel(sensors=sensor.name)
        plot_individual_raw_amplitude(ax, sensor_data, sensor.name)

    fig.suptitle("Sensor Gate Sweep - Amplitude")
    fig.tight_layout()
    return fig


def plot_raw_phase(ds: xr.Dataset, sensors: List) -> Figure:
    """
    Plots the raw phase data for the sensor gate sweep.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the I and Q quadrature data.
    sensors : list
        A list of sensors to plot.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    num_sensors = len(sensors)

    fig, axes = plt.subplots(
        1, num_sensors, figsize=(5 * num_sensors, 4), squeeze=False
    )
    axes = axes.flatten()

    for ax, sensor in zip(axes, sensors):
        sensor_data = ds.sel(sensors=sensor.name)

        ax.plot(sensor_data.bias_offsets, sensor_data.phase, "o-", markersize=2)
        ax.set_xlabel("Sensor bias offset [V]")
        ax.set_ylabel("Phase [rad]")
        ax.set_title(f"Sensor: {sensor.name}")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Sensor Gate Sweep - Phase")
    fig.tight_layout()
    return fig


def plot_amplitude_with_fit(
    ds: xr.Dataset, sensors: List, fits: xr.Dataset = None
) -> Figure:
    """
    Plots the sensor gate sweep amplitude with Lorentzian fit and max-gradient point.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    sensors : list
        A list of sensor objects to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters from peaks_dips and Lorentzian fit.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    num_sensors = len(sensors)
    fig, axes = plt.subplots(
        1, num_sensors, figsize=(5 * num_sensors, 4), squeeze=False
    )
    axes = axes.flatten()

    for ax, sensor in zip(axes, sensors):
        sensor_data = ds.sel(sensors=sensor.name)
        fit_data = fits.sel(sensors=sensor.name) if fits is not None else None

        plot_individual_amplitude_with_fit(ax, sensor_data, sensor.name, fit_data)

    fig.suptitle("Sensor Gate Sweep - Amplitude + Lorentzian Fit")
    fig.tight_layout()
    return fig


def plot_individual_raw_amplitude(ax: Axes, sensor_data: xr.Dataset, sensor_id: str):
    """
    Plots individual sensor raw amplitude data on a given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    sensor_data : xr.Dataset
        The dataset containing the sensor's quadrature data.
    sensor_id : str
        The sensor ID for the title.
    """
    ax.plot(
        sensor_data.bias_offsets,
        sensor_data.amplitude / u.mV,
        "o-",
        markersize=2,
        label="Data",
    )
    ax.set_xlabel("Sensor bias offset [V]")
    ax.set_ylabel(r"$R=\sqrt{I^2 + Q^2}$ [mV]")
    ax.set_title(f"Sensor: {sensor_id}")
    ax.grid(True, alpha=0.3)


def plot_individual_amplitude_with_fit(
    ax: Axes, sensor_data: xr.Dataset, sensor_id: str, fit: xr.Dataset = None
):
    """
    Plots individual sensor amplitude data with peak/dip markers on a given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    sensor_data : xr.Dataset
        The dataset containing the sensor's quadrature data.
    sensor_id : str
        The sensor ID for the title.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).
    """
    ax.plot(
        sensor_data.bias_offsets,
        sensor_data.amplitude / u.mV,
        "o-",
        markersize=2,
        label="Data",
    )
    ax.set_xlabel("Sensor bias offset [V]")
    ax.set_ylabel(r"$R=\sqrt{I^2 + Q^2}$ [mV]")
    ax.set_title(f"Sensor: {sensor_id}")

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
