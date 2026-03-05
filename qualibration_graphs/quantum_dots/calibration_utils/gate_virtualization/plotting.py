"""Plotting functions for gate virtualization calibration nodes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from calibration_utils.gate_virtualization.sensor_compensation_analysis import (
    shifted_lorentzian_2d,
)


def plot_sensor_compensation_diagnostic(
    ds_processed: xr.Dataset,
    fit_result: Optional[Dict[str, Any]],
    pair_key: str,
) -> Figure:
    """3-panel diagnostic figure: raw data / Lorentzian fit / residual."""
    amplitude = ds_processed["amplitude"].isel(sensors=0).values
    v_s = ds_processed["amplitude"].coords["x_volts"].values
    v_d = ds_processed["amplitude"].coords["y_volts"].values
    extent = [v_s[0], v_s[-1], v_d[0], v_d[-1]]

    fp = fit_result["fit_params"] if fit_result else None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(amplitude, extent=extent, origin="lower", aspect="auto", cmap="hot")
    axes[0].set_title(f"Sensor scan: {pair_key}")
    axes[0].set_xlabel("Sensor gate (V)")
    axes[0].set_ylabel("Device gate (V)")

    if fp is not None:
        model_signal = shifted_lorentzian_2d(
            v_s, v_d, fp["A"], fp["v0"], fp["alpha"], fp["gamma"], fp["offset"]
        )
        axes[1].imshow(model_signal, extent=extent, origin="lower", aspect="auto", cmap="hot")
        axes[1].set_title(f"Lorentzian fit (α={fp['alpha']:.4f})")
        axes[1].set_xlabel("Sensor gate (V)")

        residual = amplitude - model_signal
        axes[2].imshow(residual, extent=extent, origin="lower", aspect="auto", cmap="RdBu_r")
        axes[2].set_title("Residual")
        axes[2].set_xlabel("Sensor gate (V)")

    plt.tight_layout()
    return fig


def _get_signal_2d(
    ds: xr.Dataset,
    *,
    signal_var: str = "amplitude",
    sensor_name: Optional[str] = None,
) -> xr.DataArray:
    """Return a 2D signal map from a dataset with optional sensor selection."""
    if signal_var in ds:
        signal = ds[signal_var]
    elif "I" in ds and "Q" in ds:
        signal = np.hypot(ds["I"], ds["Q"])
    else:
        raise ValueError(f"Dataset must contain '{signal_var}' or both 'I' and 'Q'.")

    if "sensors" in signal.dims:
        if sensor_name is not None and sensor_name in signal.coords["sensors"].values:
            signal = signal.sel(sensors=sensor_name)
        else:
            signal = signal.isel(sensors=0)

    return signal


def _segment_to_voltage_coords(
    seg: Any,
    x_values: np.ndarray,
    y_values: np.ndarray,
    ny: int,
    nx: int,
) -> tuple[float, float, float, float]:
    """Convert a segment's pixel start/end points to voltage-space coordinates."""
    r_start, c_start = seg.start
    r_end, c_end = seg.end

    x_s = x_values[0] + (c_start / max(nx, 1)) * (x_values[-1] - x_values[0])
    x_e = x_values[0] + (c_end / max(nx, 1)) * (x_values[-1] - x_values[0])
    y_s = y_values[0] + (r_start / max(ny, 1)) * (y_values[-1] - y_values[0])
    y_e = y_values[0] + (r_end / max(ny, 1)) * (y_values[-1] - y_values[0])
    return x_s, x_e, y_s, y_e


def plot_2d_scan(
    ds: xr.Dataset,
    x_axis: str = "x_volts",
    y_axis: str = "y_volts",
    sensor_name: Optional[str] = None,
    title: Optional[str] = None,
) -> Figure:
    """Plot a 2D signal scan as a colour map."""
    signal = _get_signal_2d(ds, sensor_name=sensor_name)
    x_values = signal.coords[x_axis].values
    y_values = signal.coords[y_axis].values
    extent = [x_values[0], x_values[-1], y_values[0], y_values[-1]]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        signal.values,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="hot",
    )
    ax.set_title(title or "2D Scan")
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    fig.colorbar(im, ax=ax, label="Signal")
    fig.tight_layout()
    return fig


def plot_compensation_fit(
    ds: xr.Dataset,
    fit_results: Dict[str, Any],
    gate_x_name: str,
    gate_y_name: str,
    title: Optional[str] = None,
) -> Figure:
    """Overlay fitted line segments from virtual-plunger analysis on a 2D scan."""
    signal = _get_signal_2d(ds)
    x_values = signal.coords["x_volts"].values
    y_values = signal.coords["y_volts"].values
    ny, nx = signal.values.shape
    extent = [x_values[0], x_values[-1], y_values[0], y_values[-1]]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        signal.values,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="hot",
    )

    for seg in fit_results.get("segments", []):
        x_s, x_e, y_s, y_e = _segment_to_voltage_coords(seg, x_values, y_values, ny, nx)
        ax.plot([x_s, x_e], [y_s, y_e], "c-", linewidth=1.5, alpha=0.8)

    theta1 = fit_results.get("theta1")
    theta2 = fit_results.get("theta2")
    T = fit_results.get("T_matrix")
    subtitle = ""
    if theta1 is not None and theta2 is not None:
        subtitle += f"θ1={np.degrees(theta1):.1f}°, θ2={np.degrees(theta2):.1f}°"
    if T is not None:
        if subtitle:
            subtitle += "\n"
        subtitle += f"T=[[{T[0,0]:.3f}, {T[0,1]:.3f}], [{T[1,0]:.3f}, {T[1,1]:.3f}]]"

    ax.set_title((title or "Compensation Fit") + (f"\n{subtitle}" if subtitle else ""))
    ax.set_xlabel(gate_x_name)
    ax.set_ylabel(gate_y_name)
    fig.colorbar(im, ax=ax, label="Signal")
    fig.tight_layout()
    return fig


def plot_virtual_plunger_diagnostic(
    ds: xr.Dataset,
    fit_results: Optional[Dict[str, Any]],
    pair_key: str,
) -> Figure:
    """Create a 3-panel diagnostic figure for virtual-plunger calibration."""
    signal = _get_signal_2d(ds)
    amplitude = signal.values
    x_values = signal.coords["x_volts"].values
    y_values = signal.coords["y_volts"].values
    ny, nx = amplitude.shape
    extent = [x_values[0], x_values[-1], y_values[0], y_values[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    im0 = axes[0].imshow(amplitude, extent=extent, origin="lower", aspect="auto", cmap="hot")
    axes[0].set_title(f"Amplitude\n{pair_key}")
    axes[0].set_xlabel("x_volts (V)")
    axes[0].set_ylabel("y_volts (V)")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)

    mean_cp = None if fit_results is None else fit_results.get("mean_cp")
    if mean_cp is not None:
        im1 = axes[1].imshow(mean_cp, origin="lower", aspect="auto", cmap="magma")
        fig.colorbar(im1, ax=axes[1], shrink=0.85)
        axes[1].set_title("Edge Map (BayesianCP)")
    else:
        axes[1].text(0.5, 0.5, "No edge map", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("Edge Map")
    axes[1].set_xlabel("x index")
    axes[1].set_ylabel("y index")

    im2 = axes[2].imshow(amplitude, extent=extent, origin="lower", aspect="auto", cmap="hot")
    fig.colorbar(im2, ax=axes[2], shrink=0.85)
    axes[2].set_xlabel("x_volts (V)")
    axes[2].set_ylabel("y_volts (V)")

    if fit_results is not None:
        for seg in fit_results.get("segments", []):
            x_s, x_e, y_s, y_e = _segment_to_voltage_coords(seg, x_values, y_values, ny, nx)
            axes[2].plot([x_s, x_e], [y_s, y_e], "c-", linewidth=1.5, alpha=0.8)

        theta1 = fit_results.get("theta1")
        theta2 = fit_results.get("theta2")
        T = fit_results.get("T_matrix")
        title_lines = ["Segments + transform"]
        if theta1 is not None and theta2 is not None:
            title_lines.append(f"θ1={np.degrees(theta1):.1f}°, θ2={np.degrees(theta2):.1f}°")
        if T is not None:
            title_lines.append(f"T=[[{T[0,0]:.3f}, {T[0,1]:.3f}], [{T[1,0]:.3f}, {T[1,1]:.3f}]]")
        axes[2].set_title("\n".join(title_lines))
    else:
        axes[2].set_title("Segments + transform")

    fig.tight_layout()
    return fig


def plot_virtual_gate_matrix(
    matrix: np.ndarray,
    gate_names: List[str],
    title: Optional[str] = None,
) -> Figure:
    """Visualise the current virtual gate compensation matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="equal")
    ax.set_title(title or "Virtual Gate Matrix")
    ax.set_xticks(np.arange(len(gate_names)))
    ax.set_yticks(np.arange(len(gate_names)))
    ax.set_xticklabels(gate_names, rotation=45, ha="right")
    ax.set_yticklabels(gate_names)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Coefficient")
    fig.tight_layout()
    return fig
