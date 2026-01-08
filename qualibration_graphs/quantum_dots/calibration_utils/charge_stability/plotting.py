# calibration_utils/charge_stability/plotting_utils.py
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from typing import List
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from quam_builder.architecture.quantum_dots.components import SensorDot


def plot_raw_amplitude(ds: xr.Dataset, sensors: List[SensorDot]) -> Figure:
    """
    Plots the raw amplitude for charge stability measurements.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the I and Q quadrature data.
    sensors : List[SensorDot]
        A list of sensors to plot.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each sensor.
    - Each subplot contains the amplitude heatmap.
    """
    num_sensors = len(sensors)
    fig, axes = plt.subplots(1, num_sensors, figsize=(6 * num_sensors, 5), squeeze=False)
    axes = axes.flatten()

    for ax, sensor in zip(axes, sensors):
        sensor_data = ds.sel(sensors=sensor.id)
        plot_individual_raw_amplitude(ax, sensor_data, sensor.id)

    fig.suptitle("Charge Stability Map - Amplitude")
    fig.tight_layout()

    return fig


def plot_raw_phase(ds: xr.Dataset, sensors: List[SensorDot]) -> Figure:
    """
    Plots the raw phase for charge stability measurements.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the I and Q quadrature data.
    sensors : List[SensorDot]
        A list of sensors to plot.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each sensor.
    - Each subplot contains the phase heatmap.
    """
    num_sensors = len(sensors)
    fig, axes = plt.subplots(1, num_sensors, figsize=(6 * num_sensors, 5), squeeze=False)
    axes = axes.flatten()

    for ax, sensor in zip(axes, sensors):
        sensor_data = ds.sel(sensors=sensor.id)
        plot_individual_raw_phase(ax, sensor_data, sensor.id)

    fig.suptitle("Charge Stability Map - Phase")
    fig.tight_layout()

    return fig


def plot_individual_raw_amplitude(ax: Axes, sensor_data: xr.Dataset, sensor_id: str):
    """
    Plots individual sensor amplitude data on a given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    sensor_data : xr.Dataset
        The dataset containing the sensor's quadrature data.
    sensor_id : str
        The sensor ID for the title.
    """
    # Compute amplitude from I and Q
    amplitude = np.sqrt(sensor_data.I**2 + sensor_data.Q**2)

    # Plot using xarray's plot method
    amplitude.plot(
        ax=ax, x="x_volts", y="y_volts", cmap="viridis", add_colorbar=True, cbar_kwargs={"label": "Amplitude (a.u.)"}
    )

    ax.set_xlabel("X Voltage (V)")
    ax.set_ylabel("Y Voltage (V)")
    ax.set_title(f"{sensor_id}")


def plot_individual_raw_phase(ax: Axes, sensor_data: xr.Dataset, sensor_id: str):
    """
    Plots individual sensor phase data on a given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    sensor_data : xr.Dataset
        The dataset containing the sensor's quadrature data.
    sensor_id : str
        The sensor ID for the title.
    """
    # Compute phase from I and Q
    phase = np.arctan2(sensor_data.Q, sensor_data.I)

    # Plot using xarray's plot method
    phase.plot(
        ax=ax, x="x_volts", y="y_volts", cmap="twilight", add_colorbar=True, cbar_kwargs={"label": "Phase (rad)"}
    )

    ax.set_xlabel("X Voltage (V)")
    ax.set_ylabel("Y Voltage (V)")
    ax.set_title(f"{sensor_id}")
