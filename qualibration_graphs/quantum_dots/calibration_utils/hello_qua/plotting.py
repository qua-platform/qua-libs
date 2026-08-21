from typing import Dict, List

import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

__all__ = ["plot_all", "plot_dot_iq", "plot_individual_iq"]


def plot_all(ds: xr.Dataset, quantum_dots: List, sensors: List) -> Dict[str, Figure]:
    """Standard node plotting API (same shape as ``calibration_utils.sensor_dot.plot_all``).

    Parameters
    ----------
    ds:
        Raw or processed dataset, containing flat ``I_{qd.name}_{i}`` /
        ``Q_{qd.name}_{i}`` variables (dimension: ``voltage``) for each
        quantum dot ``qd`` and sensor index ``i``.
    quantum_dots:
        The swept quantum dots (one figure per dot).
    sensors:
        The sensors read out during the sweep (one subplot per sensor,
        within each dot's figure). Sensor index ``i`` is its position in
        this list, matching how the node named the streams.

    Returns
    -------
    dict[str, Figure]
        One figure per quantum dot, keyed by ``quantum_dot.name``.
    """
    figures: Dict[str, Figure] = {}
    for qd in quantum_dots:
        figures[qd.name] = plot_dot_iq(ds, qd, sensors)
    return figures


def plot_dot_iq(ds: xr.Dataset, quantum_dot, sensors: List) -> Figure:
    """Plot I and Q vs voltage for a single quantum dot, one subplot per sensor.

    Parameters
    ----------
    ds:
        Dataset containing the flat ``I_{qd.name}_{i}`` / ``Q_{qd.name}_{i}``
        variables.
    quantum_dot:
        The quantum dot whose sweep is being plotted.
    sensors:
        The sensors to show, one subplot each.

    Returns
    -------
    Figure
        The matplotlib figure for this quantum dot.
    """
    num_sensors = len(sensors)
    fig, axes = plt.subplots(1, num_sensors, figsize=(5 * num_sensors, 4), squeeze=False)
    axes = axes.flatten()

    for ax, (i, sensor) in zip(axes, enumerate(sensors)):
        plot_individual_iq(ax, ds, quantum_dot.name, i, sensor.name)

    fig.suptitle(quantum_dot.name)
    fig.tight_layout()
    return fig


def plot_individual_iq(ax: Axes, ds: xr.Dataset, quantum_dot_name: str, sensor_index: int, sensor_id: str) -> None:
    """Plot I and Q vs voltage for one (quantum dot, sensor) pair on a given axis.

    Parameters
    ----------
    ax:
        The axis to plot on.
    ds:
        Dataset containing ``I_{quantum_dot_name}_{sensor_index}`` /
        ``Q_{quantum_dot_name}_{sensor_index}``.
    quantum_dot_name:
        Name of the quantum dot, as used in the stream/variable names.
    sensor_index:
        The sensor's position in the sweep's sensor list, as used in the
        stream/variable names.
    sensor_id:
        Sensor name, used for the subplot title.
    """
    I = ds[f"I_{quantum_dot_name}_{sensor_index}"]
    Q = ds[f"Q_{quantum_dot_name}_{sensor_index}"]

    ax.plot(ds.voltage, I, "o-", markersize=2, label="I")
    ax.plot(ds.voltage, Q, "o-", markersize=2, label="Q")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Signal (a.u.)")
    ax.set_title(sensor_id)
    ax.legend()
    ax.grid(True, alpha=0.3)
