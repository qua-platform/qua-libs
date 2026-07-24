from typing import Dict, List

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from qualang_tools.units import unit

from calibration_utils.common_utils.plot_style import apply_sensor_outcome_style, sensor_success

u = unit(coerce_to_integer=True)

MARKER_LEGEND = [
    Line2D([0], [0], color="orange", linewidth=1.5, label="Resonance vs power"),
    Line2D([0], [0], color="green", linewidth=1.5, label="optimal_power"),
    Line2D([0], [0], color="blue", linewidth=1.5, linestyle="--", label="frequency_shift"),
]


def _add_rf_frequency_top_axis(ax: Axes, sensor_data: xr.Dataset) -> None:
    """Add a GHz RF-frequency axis linked linearly to the MHz detuning axis below."""
    if_hz = float(
        sensor_data.full_freq.isel(frequency_detuning=0) - sensor_data.frequency_detuning.isel(frequency_detuning=0)
    )
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    detuning_ticks_mhz = ax.get_xticks()
    ax2.set_xticks(detuning_ticks_mhz)
    ax2.set_xticklabels([f"{if_hz / 1e9 + tick / 1e3:.6f}" for tick in detuning_ticks_mhz])
    ax2.set_xlabel("RF frequency [GHz]")


def plot_all(ds_fit: xr.Dataset, sensors: List) -> Dict[str, Figure]:
    """Return all standard figures for resonator spectroscopy vs power."""
    return {"amplitude": plot_raw_data_with_fit(ds_fit, sensors)}


def plot_raw_data_with_fit(ds_fit: xr.Dataset, sensors: List) -> Figure:
    """Plot normalized 2D power sweeps with dual frequency axes and fit markers."""
    num_sensors = len(sensors)
    fig, axes = plt.subplots(1, num_sensors, figsize=(max(5 * num_sensors, 8), 5), squeeze=False)
    axes = axes.flatten()

    for ax, sensor in zip(axes, sensors):
        plot_individual_raw_data_with_fit(
            ax,
            ds_fit.sel(sensor=sensor.name),
            sensor.name,
            ds_fit.sel(sensor=sensor.name),
            sensor_success(ds_fit, sensor.name),
        )

    fig.suptitle("Resonator spectroscopy vs power")
    fig.tight_layout()
    return fig


def plot_individual_raw_data_with_fit(
    ax: Axes,
    sensor_data: xr.Dataset,
    sensor_id: str,
    fit: xr.Dataset,
    success: bool | None,
):
    """Plot IQ_abs_norm vs detuning × power with MHz (bottom) and GHz (top) x-axes."""
    sensor_data.assign_coords(
        frequency_detuning_MHz=sensor_data.frequency_detuning / u.MHz
    ).IQ_abs_norm.plot(
        ax=ax,
        add_colorbar=True,
        x="frequency_detuning_MHz",
        y="power",
        robust=True,
    )
    ax.set_xlabel("Frequency detuning [MHz]")
    ax.set_ylabel("Power [dBm]")
    _add_rf_frequency_top_axis(ax, sensor_data)

    if success:
        resonance_vs_power = sensor_data.IQ_abs_norm.idxmin(dim="frequency_detuning")
        ax.plot(
            resonance_vs_power * 1e-6,
            sensor_data.power,
            color="orange",
            linewidth=1.5,
        )
        if "optimal_power" in fit.coords:
            ax.axhline(y=float(fit.optimal_power), color="green", linewidth=1.5)
        if "frequency_shift" in fit.coords:
            ax.axvline(x=float(fit.frequency_shift) * 1e-6, color="blue", linestyle="--", linewidth=1.5)
        ax.legend(handles=MARKER_LEGEND, loc="upper right", fontsize=8)

    apply_sensor_outcome_style(ax, sensor_id, success)
