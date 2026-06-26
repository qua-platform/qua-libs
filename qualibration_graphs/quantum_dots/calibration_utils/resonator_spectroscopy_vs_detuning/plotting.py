from typing import List
import xarray as xr
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from quam_builder.architecture.quantum_dots.components import SensorDot


def plot_raw_data_with_fit(ds: xr.Dataset, sensors: List[SensorDot], fits: xr.Dataset):
    """Plot PCA signal maps and extracted optimal points for all sensors."""
    num_sensors = len(sensors)
    fig, axes = plt.subplots(
        1, num_sensors, figsize=(5 * num_sensors, 4), squeeze=False
    )
    axes = axes.flatten()

    for ax, sensor in zip(axes, sensors):
        sensor_data = ds.sel(sensor=sensor.name)
        fit_data = fits.sel(sensor=sensor.name) if fits is not None else None

        plot_individual_raw_data_with_fit(ax, sensor_data, sensor.name, fit_data)

    fig.suptitle("Resonator spectroscopy vs detuning (PCA signal)")
    fig.set_size_inches(15, 9)
    fig.tight_layout()
    return fig


def plot_individual_raw_data_with_fit(
    ax: Axes, sensor_data: xr.Dataset, sensor_id: str, fit: xr.Dataset = None
):
    """Plot a single sensor PCA signal map with optional optimal-point marker."""
    sensor_data.assign_coords(freq_GHz=sensor_data.full_freq / 1e9).IQ_abs.plot(
        ax=ax, add_colorbar=False, x="freq_GHz", y="detuning", linewidth=0.5
    )
    ax.set_xlabel("Readout frequency [GHz]")
    ax.set_ylabel("Detuning [V]")
    ax.set_title(sensor_id)

    if fit is not None:
        sensor_data.assign_coords(
            freq_GHz=sensor_data.full_freq / 1e9
        ).assign({"pca_signal": fit.pca_signal_abs}).pca_signal.plot(
            ax=ax,
            add_colorbar=True,
            x="freq_GHz",
            y="detuning",
            cmap="magma",
            alpha=0.7,
        )
        try:
            if bool(fit.success):
                ax.scatter(
                    float(fit.res_freq) / 1e9,
                    float(fit.optimal_detuning),
                    color="cyan",
                    edgecolors="black",
                    s=80,
                    marker="x",
                    zorder=10,
                )
        except Exception:
            pass
