# calibration_utils/charge_stability/plotting_utils.py
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from typing import List
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from quam_builder.architecture.quantum_dots.components import SensorDot


def _align_base_to_overlay(base_array: np.ndarray, overlay_array: np.ndarray) -> np.ndarray:
    """Align base array to overlay shape when change-point maps are offset by one pixel."""
    if base_array.shape == overlay_array.shape:
        return base_array
    br, bc = base_array.shape[:2]
    or_, oc = overlay_array.shape[:2]
    if br == or_ + 1 and bc == oc + 1:
        return base_array[1:, 1:]
    if br == or_ and bc == oc + 1:
        return base_array[:, 1:]
    if br == or_ + 1 and bc == oc:
        return base_array[1:, :]
    return base_array


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


def overlay_binary_mask(
    ax: Axes,
    base_array: np.ndarray,
    overlay_array: np.ndarray,
    threshold: float,
    base_cmap: str = "viridis",
    overlay_color: str = "red",
    alpha: float = 0.9,
):
    """
    Overlay a binary threshold mask on top of a base image.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the overlay.
    base_array : np.ndarray
        The underlying 2D array (e.g., background color blocks).
    overlay_array : np.ndarray
        The array to threshold for the overlay (e.g., line detection result).
    threshold : float
        Only pixels > threshold will be highlighted.
    base_cmap : str
        Colormap for the base array.
    overlay_color : str
        Color for highlighted pixels (e.g. 'red', 'white', '#ff00ff').
    alpha : float
        Transparency for the overlay color (0 = invisible, 1 = opaque).
    """
    from matplotlib.colors import ListedColormap

    base_array = _align_base_to_overlay(base_array, overlay_array)

    # Base layer
    im0 = ax.imshow(base_array, cmap=base_cmap, origin="lower")

    # Binary mask overlay
    mask = np.where(overlay_array > threshold, 1, np.nan)

    # Make a single-color colormap
    overlay_cmap = ListedColormap([overlay_color])

    # Plot only where mask==1
    ax.imshow(mask, cmap=overlay_cmap, origin="lower", alpha=alpha)

    ax.set_title(f"Threshold overlay (>{threshold})")


def plot_change_point_overlays(
    sensor_data: xr.Dataset, fit_params: dict, sensor_id: str, threshold: float = 0.25
) -> Figure:
    """
    Plot change point detection overlays for a sensor.

    Parameters
    ----------
    sensor_data : xr.Dataset
        Dataset containing the sensor's I and Q quadrature data.
    fit_params : dict
        Dictionary containing fit parameters with 'cp', 'cp2', and 'mean_cp' arrays.
    sensor_id : str
        The sensor ID for the title.
    threshold : float
        Threshold for highlighting change points.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    amplitude = np.sqrt(sensor_data.I**2 + sensor_data.Q**2).values

    # Convert lists to numpy arrays if needed
    mean_cp = np.array(fit_params["mean_cp"])
    cp = np.array(fit_params["cp"])
    cp2 = np.array(fit_params["cp2"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Mean change point
    overlay_binary_mask(
        axes[0],
        base_array=amplitude,
        overlay_array=mean_cp,
        threshold=threshold,
    )
    axes[0].set_title(f"{sensor_id} - Mean CP")

    # Horizontal change points
    overlay_binary_mask(
        axes[1],
        base_array=amplitude,
        overlay_array=cp,
        threshold=threshold,
    )
    axes[1].set_title(f"{sensor_id} - Horizontal CP")

    # Vertical change points
    overlay_binary_mask(
        axes[2],
        base_array=amplitude,
        overlay_array=cp2.T,
        threshold=threshold,
    )
    axes[2].set_title(f"{sensor_id} - Vertical CP")

    fig.tight_layout()
    return fig


def plot_line_fit_overlays(
    sensor_data: xr.Dataset,
    fit_params: dict,
    sensor_id: str,
) -> Figure:
    """
    Plot line-segment fits and intersections extracted from the edge map.

    Parameters
    ----------
    sensor_data : xr.Dataset
        Dataset containing the sensor's I and Q quadrature data.
    fit_params : dict
        Dictionary containing edge_binary, skeleton, segments, and intersections.
    sensor_id : str
        The sensor ID for the title.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    amplitude = np.sqrt(sensor_data.I**2 + sensor_data.Q**2).values
    edge_binary = np.array(fit_params.get("edge_binary", []))
    skeleton = np.array(fit_params.get("skeleton", []))
    segments = fit_params.get("segments", [])
    intersections = np.array(fit_params.get("intersections", []))

    base = amplitude
    if edge_binary.size > 0:
        base = _align_base_to_overlay(amplitude, edge_binary)
    elif "mean_cp" in fit_params and len(fit_params["mean_cp"]) > 0:
        base = _align_base_to_overlay(amplitude, np.array(fit_params["mean_cp"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # binary + skeleton overlay
    ax0 = axes[0]
    im0 = ax0.imshow(base, origin="lower", cmap="magma")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label="sensor")
    if edge_binary.size > 0:
        ax0.imshow(np.ma.masked_where(edge_binary == 0, edge_binary), cmap="Reds", alpha=0.4, origin="lower")
    if skeleton.size > 0:
        ax0.imshow(np.ma.masked_where(skeleton == 0, skeleton), cmap="Blues", alpha=0.5, origin="lower")
    ax0.set_title(f"{sensor_id} - Edge mask & skeleton")
    ax0.set_xlabel("col (V2)")
    ax0.set_ylabel("row (V1)")

    # fitted segments and intersections
    ax1 = axes[1]
    ax1.imshow(base, origin="lower", cmap="gray", alpha=0.35)
    if skeleton.size > 0:
        ax1.imshow(np.ma.masked_where(skeleton == 0, skeleton), cmap="Blues", alpha=0.4, origin="lower")

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(segments), 1)))
    for idx, seg in enumerate(segments):
        start = np.array(seg.get("start", []))
        end = np.array(seg.get("end", []))
        if start.size == 2 and end.size == 2:
            ax1.plot([start[1], end[1]], [start[0], end[0]], "-", color=colors[idx % len(colors)], lw=2)

    if intersections.size > 0:
        intersections = np.array(intersections)
        if intersections.ndim == 1 and intersections.size == 0:
            pass
        else:
            ax1.scatter(
                intersections[:, 1],
                intersections[:, 0],
                marker="*",
                s=120,
                c="gold",
                edgecolor="k",
                zorder=5,
                label="triple points",
            )

    ax1.set_title(f"{sensor_id} - Fitted segments & intersections")
    ax1.set_xlabel("col (V2)")
    ax1.set_ylabel("row (V1)")
    if len(segments) > 0 or intersections.size > 0:
        ax1.legend(loc="upper right")

    # constrained_layout is already enabled in subplots; avoid tight_layout clashes with colorbars
    return fig
