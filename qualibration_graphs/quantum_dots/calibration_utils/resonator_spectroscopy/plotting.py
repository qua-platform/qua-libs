from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from qualang_tools.units import unit
from qualibration_libs.analysis import lorentzian_dip

u = unit(coerce_to_integer=True)


def plot_raw_phase(ds: xr.Dataset, sensors: List) -> Figure:
    """
    Plots the raw phase data for the given sensors.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    sensors : list
        A list of sensors to plot.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each sensor.
    - Each subplot contains two x-axes: one for the full frequency in GHz and one for the detuning in MHz.
    """
    num_sensors = len(sensors)

    fig, axes = plt.subplots(1, num_sensors, figsize=(5 * num_sensors, 4), squeeze=False)
    axes = axes.flatten()

    for ax, sensor in zip(axes, sensors):
        sensor_data = ds.sel(sensors=sensor.name)

        # Create a first x-axis for full_freq_GHz
        ax.plot(sensor_data.full_freq / u.GHz, sensor_data.phase, "o-", markersize=2)
        ax.set_xlabel("RF frequency [GHz]")
        ax.set_ylabel("Phase [rad]")
        ax.set_title(f"Sensor: {sensor.name}")

        # Create a second x-axis for detuning_MHz
        ax2 = ax.twiny()
        ax2.plot(sensor_data.detuning / u.MHz, sensor_data.phase, "o-", markersize=2, alpha=0)
        ax2.set_xlabel("Detuning [MHz]")

    fig.suptitle("Resonator spectroscopy (phase)")
    fig.tight_layout()
    return fig


def plot_raw_amplitude_with_fit(ds: xr.Dataset, sensors: List, fits: xr.Dataset = None) -> Figure:
    """
    Plots the resonator spectroscopy amplitude IQ_abs with fitted curves for the given sensors.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    sensors : list
        A list of sensor objects to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    num_sensors = len(sensors)
    fig, axes = plt.subplots(1, num_sensors, figsize=(5 * num_sensors, 4), squeeze=False)
    axes = axes.flatten()

    for ax, sensor in zip(axes, sensors):
        sensor_data = ds.sel(sensors=sensor.name)
        fit_data = fits.sel(sensors=sensor.name) if fits is not None else None

        plot_individual_amplitude_with_fit(ax, sensor_data, sensor.name, fit_data)

    fig.suptitle("Resonator spectroscopy (amplitude + fit)")
    fig.tight_layout()
    return fig


def plot_individual_amplitude_with_fit(ax: Axes, sensor_data: xr.Dataset, sensor_id: str, fit: xr.Dataset = None):
    """
    Plots individual sensor data on a given axis with optional fit.

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
    # Plot the amplitude data
    ax.plot(sensor_data.full_freq / u.GHz, sensor_data.IQ_abs / u.mV, "o-", markersize=2, label="Data")
    ax.set_xlabel("RF frequency [GHz]")
    ax.set_ylabel(r"$R=\sqrt{I^2 + Q^2}$ [mV]")
    ax.set_title(f"Sensor: {sensor_id}")

    # Create a second x-axis for detuning_MHz
    ax2 = ax.twiny()
    ax2.plot(sensor_data.detuning / u.MHz, sensor_data.IQ_abs / u.mV, "o-", markersize=2, alpha=0)
    ax2.set_xlabel("Detuning [MHz]")

    # Plot the fitted data if available
    if fit is not None and all(k in fit for k in ["amplitude", "position", "width", "base_line"]):
        fitted_data = lorentzian_dip(
            sensor_data.detuning,
            float(fit.amplitude.values),
            float(fit.position.values),
            float(fit.width.values) / 2,
            float(fit.base_line.mean().values),
        )
        ax2.plot(sensor_data.detuning / u.MHz, fitted_data / u.mV, "r--", label="Fit")
        ax.legend()

    ax.grid(True, alpha=0.3)
