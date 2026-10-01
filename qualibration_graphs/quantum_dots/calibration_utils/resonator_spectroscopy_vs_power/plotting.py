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
    Line2D([0], [0], color="green", linewidth=1.5, label="Optimal readout power"),
    Line2D([0], [0], color="blue", linewidth=1.5, linestyle="--", label="Fitted frequency shift"),
]


def _add_detuning_top_axis(ax: Axes, if_hz: float) -> Axes:
    """Add a frequency-detuning axis on top, linked to the readout-frequency axis below."""
    if_mhz = if_hz / 1e6
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    readout_ticks_mhz = ax.get_xticks()
    ax2.set_xticks(readout_ticks_mhz)
    ax2.set_xticklabels([f"{tick - if_mhz:.2f}" for tick in readout_ticks_mhz])
    ax2.set_xlabel("Frequency detuning [MHz]")
    return ax2


def _readout_if_hz(sensor_data: xr.Dataset) -> float:
    return float(
        sensor_data.full_freq.isel(frequency_detuning=0) - sensor_data.frequency_detuning.isel(frequency_detuning=0)
    )


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
    """Plot IQ_abs_norm vs readout frequency × power with MHz readout (bottom) and detuning (top) x-axes."""
    if_hz = _readout_if_hz(sensor_data)
    sensor_data.assign_coords(readout_frequency_MHz=sensor_data.full_freq / u.MHz).IQ_abs_norm.plot(
        ax=ax,
        add_colorbar=True,
        x="readout_frequency_MHz",
        y="power",
        robust=True,
    )
    ax.set_xlabel("Readout frequency [MHz]")
    ax.set_ylabel("Power [dBm]")
    _add_detuning_top_axis(ax, if_hz)

    if success:
        resonance_vs_power = sensor_data.IQ_abs_norm.idxmin(dim="frequency_detuning")
        ax.plot(
            (if_hz + resonance_vs_power) / u.MHz,
            sensor_data.power,
            color="orange",
            linewidth=1.5,
        )
        if "optimal_power" in fit.coords:
            ax.axhline(y=float(fit.optimal_power), color="green", linewidth=1.5)
        if "frequency_shift" in fit.coords:
            ax.axvline(
                x=(if_hz + float(fit.frequency_shift)) / u.MHz,
                color="blue",
                linestyle="--",
                linewidth=1.5,
            )
        ax.legend(handles=MARKER_LEGEND, loc="upper right", fontsize=8)

    apply_sensor_outcome_style(ax, sensor_id, success)
