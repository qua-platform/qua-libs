from typing import Dict, List

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from calibration_utils.common_utils.plot_style import apply_sensor_outcome_style, sensor_success

PCA_PEAK_LEGEND = [
    Line2D(
        [0],
        [0],
        marker="x",
        color="cyan",
        markeredgecolor="black",
        linestyle="None",
        markersize=8,
        label="PCA peak",
    ),
]


def plot_all(ds_fit: xr.Dataset, sensors: List) -> Dict[str, Figure]:
    """Return all standard figures for resonator spectroscopy vs detuning."""
    return {"amplitude": plot_raw_data_with_fit(ds_fit, sensors)}


def plot_raw_data_with_fit(ds_fit: xr.Dataset, sensors: List) -> Figure:
    """Plot PCA signal maps and extracted optimal points for all sensors."""
    num_sensors = len(sensors)
    fig, axes = plt.subplots(1, num_sensors, figsize=(max(5 * num_sensors, 8), 5), squeeze=False)
    axes = axes.flatten()

    for ax, sensor in zip(axes, sensors):
        sensor_data = ds_fit.sel(sensor=sensor.name)
        fit_data = ds_fit.sel(sensor=sensor.name)
        plot_individual_raw_data_with_fit(
            ax,
            sensor_data,
            sensor.name,
            fit_data,
            sensor_success(ds_fit, sensor.name),
        )

    fig.suptitle("Resonator spectroscopy vs detuning (PCA signal)")
    fig.tight_layout()
    return fig


def plot_individual_raw_data_with_fit(
    ax: Axes,
    sensor_data: xr.Dataset,
    sensor_id: str,
    fit: xr.Dataset,
    success: bool | None,
):
    """Plot IQ background with PCA signal overlay and optional peak marker."""
    sensor_data.assign_coords(freq_GHz=sensor_data.full_freq / 1e9).IQ_abs.plot(
        ax=ax,
        add_colorbar=False,
        x="freq_GHz",
        y="detuning",
        linewidth=0.5,
    )
    ax.set_xlabel("Readout frequency [GHz]")
    ax.set_ylabel("Detuning [V]")

    if "pca_signal_abs" in fit:
        sensor_data.assign_coords(freq_GHz=sensor_data.full_freq / 1e9).assign(
            {"pca_signal": fit.pca_signal_abs}
        ).pca_signal.plot(
            ax=ax,
            add_colorbar=True,
            x="freq_GHz",
            y="detuning",
            cmap="magma",
            alpha=0.7,
        )

    if success and "res_freq" in fit.coords and "optimal_detuning" in fit.coords:
        ax.scatter(
            float(fit.res_freq) / 1e9,
            float(fit.optimal_detuning),
            color="cyan",
            s=80,
            marker="x",
            zorder=10,
        )
        ax.legend(handles=PCA_PEAK_LEGEND, loc="upper right", fontsize=8)

    apply_sensor_outcome_style(ax, sensor_id, success)
