"""Plotting functions for gate virtualization calibration nodes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure


def _select_signal(ds: xr.Dataset) -> xr.DataArray:
    for candidate in ("amplitude", "IQ_abs", "I"):
        if candidate in ds.data_vars:
            return ds[candidate]
    # Fallback to first numeric variable.
    for _, data_var in ds.data_vars.items():
        if np.issubdtype(data_var.dtype, np.number):
            return data_var
    raise ValueError("No numeric data variable available for plotting.")


def plot_2d_scan(
    ds: xr.Dataset,
    x_axis: str = "x_volts",
    y_axis: str = "y_volts",
    sensor_name: Optional[str] = None,
    title: Optional[str] = None,
) -> Figure:
    """Plot a 2D voltage scan as a colour map."""
    signal = _select_signal(ds)
    if "sensors" in signal.dims:
        if sensor_name is not None and sensor_name in ds.coords.get("sensors", []):
            signal = signal.sel(sensors=sensor_name)
        else:
            signal = signal.isel(sensors=0)

    if x_axis not in signal.dims or y_axis not in signal.dims:
        raise ValueError(f"Requested axes '{x_axis}', '{y_axis}' not in signal dims {signal.dims}.")

    x = np.asarray(ds.coords[x_axis].values, dtype=float)
    y = np.asarray(ds.coords[y_axis].values, dtype=float)
    z = np.asarray(signal.transpose(y_axis, x_axis).values, dtype=float)

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    pcm = ax.pcolormesh(x, y, z, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=ax, label=str(signal.name or "signal"))
    ax.set_title(title or "2D Scan")
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    fig.tight_layout()
    return fig


def plot_tunnel_slope_fit(
    drive_values: Sequence[float],
    tunnel_couplings: Sequence[float],
    fit_results: Dict[str, Any],
    title: Optional[str] = None,
    x_label: str = "drive",
) -> Figure:
    """Plot extracted tunnel couplings vs drive values with a linear fit."""
    x = np.asarray(drive_values, dtype=float)
    y = np.asarray(tunnel_couplings, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(x[valid], y[valid], "o", label="extracted")

    slope = fit_results.get("coefficient", fit_results.get("slope_fit_slope", np.nan))
    intercept = fit_results.get("slope_fit_intercept", np.nan)
    if np.isfinite(slope) and np.isfinite(intercept) and np.any(valid):
        x_line = np.linspace(np.nanmin(x[valid]), np.nanmax(x[valid]), 200)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, "r--", label="linear fit")

    r2 = fit_results.get("fit_quality", fit_results.get("slope_fit_fit_quality", np.nan))
    ax.set_title((title or "Tunnel Coupling Slope Fit") + (f" | R²={r2:.3f}" if np.isfinite(r2) else ""))
    ax.set_xlabel(x_label)
    ax.set_ylabel("tunnel coupling (arb.)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_compensation_fit(
    ds: xr.Dataset,
    fit_results: Dict[str, Any],
    gate_x_name: str,
    gate_y_name: str,
    title: Optional[str] = None,
) -> Figure:
    """Show the raw 2D map and the extracted tunnel-slope fit."""
    fig = plt.figure(figsize=(7.0, 7.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 2.0])

    # Top: raw map
    ax0 = fig.add_subplot(gs[0, 0])
    signal = _select_signal(ds)
    if "sensors" in signal.dims:
        signal = signal.isel(sensors=0)

    x_axis = fit_results.get("barrier_axis", "x_volts")
    y_axis = fit_results.get("compensation_axis", "y_volts")
    if x_axis in signal.dims and y_axis in signal.dims:
        x = np.asarray(ds.coords[x_axis].values, dtype=float)
        y = np.asarray(ds.coords[y_axis].values, dtype=float)
        z = np.asarray(signal.transpose(y_axis, x_axis).values, dtype=float)
        pcm = ax0.pcolormesh(x, y, z, shading="auto", cmap="viridis")
        fig.colorbar(pcm, ax=ax0, label=str(signal.name or "signal"))
    ax0.set_title(title or f"Compensation Fit: {gate_x_name} vs {gate_y_name}")
    ax0.set_xlabel(gate_x_name)
    ax0.set_ylabel(gate_y_name)

    # Bottom: tunnel-coupling slope fit
    ax1 = fig.add_subplot(gs[1, 0])
    x_vals = np.asarray(fit_results.get("drive_values", []), dtype=float)
    y_vals = np.asarray(fit_results.get("tunnel_couplings", []), dtype=float)
    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    if np.any(valid):
        ax1.plot(x_vals[valid], y_vals[valid], "o", label="extracted")
        slope = fit_results.get("coefficient", np.nan)
        intercept = fit_results.get("slope_fit_intercept", np.nan)
        if np.isfinite(slope) and np.isfinite(intercept):
            x_line = np.linspace(np.nanmin(x_vals[valid]), np.nanmax(x_vals[valid]), 200)
            ax1.plot(x_line, slope * x_line + intercept, "r--", label="linear fit")
    ax1.set_xlabel(gate_x_name)
    ax1.set_ylabel("tunnel coupling (arb.)")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best")

    fig.tight_layout()
    return fig


def plot_virtual_gate_matrix(
    matrix: np.ndarray,
    gate_names: List[str],
    title: Optional[str] = None,
) -> Figure:
    """Visualise the current virtual gate compensation matrix as a heatmap."""
    mat = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(mat, cmap="coolwarm", aspect="auto")
    fig.colorbar(im, ax=ax, label="coefficient")
    ax.set_xticks(np.arange(len(gate_names)))
    ax.set_yticks(np.arange(len(gate_names)))
    ax.set_xticklabels(gate_names, rotation=45, ha="right")
    ax.set_yticklabels(gate_names)
    ax.set_title(title or "Virtual Gate Matrix")

    for row in range(mat.shape[0]):
        for col in range(mat.shape[1]):
            value = mat[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    return fig


def plot_barrier_transform_history(
    history: Sequence[Dict[str, Any]],
    barrier_names: Sequence[str],
    title: Optional[str] = None,
) -> Figure:
    """Plot evolution of barrier-transform matrices across B* stages."""
    if not history:
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        ax.set_title(title or "Barrier Transform History")
        ax.text(0.5, 0.5, "No history available", ha="center", va="center")
        ax.axis("off")
        return fig

    n = len(history)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)
    for idx, entry in enumerate(history):
        ax = axes[0, idx]
        mat = np.asarray(entry.get("matrix"), dtype=float)
        im = ax.imshow(mat, cmap="coolwarm", aspect="auto")
        ax.set_title(entry.get("label", f"step {idx + 1}"))
        ax.set_xticks(np.arange(len(barrier_names)))
        ax.set_yticks(np.arange(len(barrier_names)))
        ax.set_xticklabels(barrier_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(barrier_names, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title or "Barrier Transform Evolution")
    fig.tight_layout()
    return fig
