from typing import List
import xarray as xr
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from qualang_tools.units import unit
from quam_builder.architecture.quantum_dots.components import SensorDot

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, sensors: List[SensorDot], fits: xr.Dataset):
    """
    Plots the raw data with fitted curves for the given sensors.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    sensors : list of SensorDot
        A list of qubits to plot.
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
        fit_data = fits.sel(sensor=sensor.name) if fits is not None else None

        plot_individual_raw_data_with_fit(ax, sensor_data, sensor.name, fit_data)

    fig.suptitle("Resonator spectroscopy vs power")
    fig.set_size_inches(15, 9)
    fig.tight_layout()
    return fig


def plot_individual_raw_data_with_fit(ax: Axes, sensor_data: xr.Dataset, sensor_id: str, fit: xr.Dataset = None):
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

    sensor_data.assign_coords(freq_GHz=sensor_data.full_freq / 1e9).IQ_abs.plot(
        ax=ax, add_colorbar=False, x="freq_GHz", y="power", linewidth=0.5
    )
    ax.set_ylabel("Power (dBm)")

    ax2 = ax.twiny()
    sensor_data.assign_coords(detuning_MHz=sensor_data.detuning / u.MHz).IQ_abs_norm.plot(
        ax=ax2, add_colorbar=False, x="detuning_MHz", y="power", robust=True
    )
    ax2.set_xlabel("Detuning [MHz]")
    ax2.set_title(sensor_id)

    if fit is not None:
        ax2.plot((fit.rr_min_response) * 1e-6, fit.power, color="orange", linewidth=0.5)
        try:
            if bool(fit.success):
                ax2.axhline(y=float(fit.optimal_power), color="g", linestyle="-")
                ax2.axvline(x=float(fit.freq_shift) * 1e-6, color="blue", linestyle="--")
        except Exception:
            pass
