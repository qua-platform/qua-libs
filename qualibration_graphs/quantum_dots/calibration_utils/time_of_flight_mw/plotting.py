from typing import List
import xarray as xr
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from qualang_tools.units import unit
from quam_builder.architecture.quantum_dots.components import SensorDot

u = unit(coerce_to_integer=True)


def plot_single_run_with_fit(ds: xr.Dataset, sensors: List[SensorDot], fits: xr.Dataset):
    """
    Plots the resonator spectroscopy amplitude IQ_abs with fitted curves for the given sensors.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    sensors : list of SensorDot
        A list of sensors to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each sensor.
    - Each subplot contains the raw data and the fitted curve.
    """
    num_sensors = len(sensors)
    fig, axes = plt.subplots(1, num_sensors, figsize=(5 * num_sensors, 4), squeeze=False)
    axes = axes.flatten()
    
    for ax, sensor in zip(axes, sensors):
        sensor_data = ds.sel(sensor=sensor.name)
        fit_data = fits.sel(sensor=sensor.name)
        plot_individual_single_run_with_fit(ax, sensor_data, sensor.name, fit_data)
    
    fig.suptitle("Single run")
    fig.tight_layout()
    return fig


def plot_averaged_run_with_fit(ds: xr.Dataset, sensors: List[SensorDot], fits: xr.Dataset):
    """
    Plots the resonator spectroscopy amplitude IQ_abs with fitted curves for the given sensors.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    sensors : list of SensorDot
        A list of sensors to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each sensor.
    - Each subplot contains the raw data and the fitted curve.
    """
    num_sensors = len(sensors)
    fig, axes = plt.subplots(1, num_sensors, figsize=(5 * num_sensors, 4), squeeze=False)
    axes = axes.flatten()
    
    for ax, sensor in zip(axes, sensors):
        sensor_data = ds.sel(sensor=sensor.name)
        fit_data = fits.sel(sensor=sensor.name)
        plot_individual_averaged_run_with_fit(ax, sensor_data, sensor.name, fit_data)
    
    fig.suptitle("Averaged run")
    fig.tight_layout()
    return fig

def plot_individual_single_run_with_fit(ax: Axes, sensor_data: xr.Dataset, sensor_name: str, fit: xr.Dataset = None):
    """
    Plots individual sensor data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    sensor_data : xr.Dataset
        The dataset containing the sensor's quadrature data.
    sensor_name : str
        The sensor name for the title.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """
    sensor_data.loc[sensor_name].adc_single_runI.plot(ax=ax, x="readout_time", label="I", color="b")
    sensor_data.loc[sensor_name].adc_single_runQ.plot(ax=ax, x="readout_time", label="Q", color="r")
    ax.axvline(fit.delay, color="k", linestyle="--", label="TOF")
    # ax.axhline(ds.loc[sensor].offsets_I, color="b", linestyle="--")
    # ax.axhline(ds.loc[sensor].offsets_Q, color="r", linestyle="--")
    ax.fill_between(
        range(sensor_data.sizes["readout_time"]),
        -0.5,
        0.5,
        color="grey",
        alpha=0.2,
        label="ADC Range",
    )
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Readout amplitude [mV]")
    ax.set_title(sensor_name["sensor"])


def plot_individual_averaged_run_with_fit(ax: Axes, ds: xr.Dataset, sensor: dict[str, str], fit: xr.Dataset = None):
    """
    Plots individual sensor data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    sensor : dict[str, str]
        mapping to the sensor to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """
    ds.loc[sensor].adcI.plot(ax=ax, x="readout_time", label="I", color="b")
    ds.loc[sensor].adcQ.plot(ax=ax, x="readout_time", label="Q", color="r")
    ax.axvline(fit.delay, color="k", linestyle="--", label="TOF")
    # ax.axhline(ds.loc[sensor].offsets_I, color="b", linestyle="--")
    # ax.axhline(ds.loc[sensor].offsets_Q, color="r", linestyle="--")
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Readout amplitude [mV]")
    ax.set_title(sensor["sensor"])
