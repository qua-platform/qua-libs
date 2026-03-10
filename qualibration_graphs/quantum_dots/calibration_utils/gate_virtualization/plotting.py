"""Plotting functions for gate virtualization calibration nodes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

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
        model_signal = shifted_lorentzian_2d(v_s, v_d, fp["A"], fp["v0"], fp["alpha"], fp["gamma"], fp["offset"])
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


def plot_tunnel_slope_fit(
    drive_values: Sequence[float],
    tunnel_couplings: Sequence[float],
    fit_results: Dict[str, Any],
    title: Optional[str] = None,
    x_label: str = "drive",
) -> Figure:
    """Plot extracted tunnel couplings versus drive with a linear fit."""
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
    ax.set_title((title or "Tunnel Coupling Slope Fit") + (f" | R^2={r2:.3f}" if np.isfinite(r2) else ""))
    ax.set_xlabel(x_label)
    ax.set_ylabel("tunnel coupling (arb.)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _sample_successful_point_fits(point_fits: Sequence[Dict[str, Any]], max_traces: int = 5) -> List[Dict[str, Any]]:
    """Pick up to ``max_traces`` successful point fits uniformly across drive values."""
    successful = [fit for fit in point_fits if bool(fit.get("success", False))]
    if len(successful) <= max_traces:
        return successful
    indices = np.linspace(0, len(successful) - 1, num=max_traces, dtype=int)
    return [successful[idx] for idx in indices]


def plot_detuning_fit_family(
    fit_results: Dict[str, Any],
    title: Optional[str] = None,
    max_traces: int = 5,
) -> Figure:
    """Plot measured detuning traces and model fits for selected drive points."""
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    sampled_fits = _sample_successful_point_fits(fit_results.get("point_fits", []), max_traces=max_traces)
    if len(sampled_fits) == 0:
        ax.set_title(title or "Detuning Traces (No Valid Fits)")
        ax.text(0.5, 0.5, "No successful detuning fits", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    for fit in sampled_fits:
        detuning = np.asarray(fit.get("detuning", []), dtype=float)
        signal = np.asarray(fit.get("signal", []), dtype=float)
        signal_fit = np.asarray(fit.get("signal_fit", []), dtype=float)
        drive_value = float(fit.get("drive_value", np.nan))
        t_val = float(fit.get("tunnel_coupling", np.nan))
        label = (
            f"dB={1e3 * drive_value:.2f} mV, t={t_val:.3g}"
            if np.isfinite(drive_value) and np.isfinite(t_val)
            else "trace"
        )
        ax.plot(detuning, signal, "o", ms=2.6, alpha=0.70, label=label)
        if detuning.size == signal_fit.size and detuning.size > 0:
            ax.plot(detuning, signal_fit, "--", lw=1.5, alpha=0.95)

    ax.set_title(title or "Detuning Trace Fits")
    ax.set_xlabel("Detuning (scaled units)")
    ax.set_ylabel("Sensor signal (arb.)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_barrier_pair_diagnostics(
    fit_results: Dict[str, Any],
    drive_label: str,
    target_label: str,
    title: Optional[str] = None,
) -> Figure:
    """Plot pair diagnostics: detuning-trace fits and extracted t-vs-drive slope."""
    fig = plt.figure(figsize=(10.8, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.26)

    ax_left = fig.add_subplot(gs[0, 0])
    sampled_fits = _sample_successful_point_fits(fit_results.get("point_fits", []), max_traces=5)
    for fit in sampled_fits:
        detuning = np.asarray(fit.get("detuning", []), dtype=float)
        signal = np.asarray(fit.get("signal", []), dtype=float)
        signal_fit = np.asarray(fit.get("signal_fit", []), dtype=float)
        drive_value = float(fit.get("drive_value", np.nan))
        t_val = float(fit.get("tunnel_coupling", np.nan))
        label = f"{1e3 * drive_value:.2f} mV, t={t_val:.3g}" if np.isfinite(drive_value) and np.isfinite(t_val) else "trace"
        ax_left.plot(detuning, signal, "o", ms=2.5, alpha=0.72, label=label)
        if detuning.size == signal_fit.size and detuning.size > 0:
            ax_left.plot(detuning, signal_fit, "--", lw=1.4, alpha=0.95)

    ax_left.set_xlabel("Detuning (scaled units)")
    ax_left.set_ylabel("Sensor signal (arb.)")
    ax_left.set_title(f"{target_label}: detuning fits")
    ax_left.grid(alpha=0.25)
    if sampled_fits:
        ax_left.legend(loc="best", fontsize=8)

    ax_right = fig.add_subplot(gs[0, 1])
    x_vals = np.asarray(fit_results.get("drive_values", []), dtype=float)
    y_vals = np.asarray(fit_results.get("tunnel_couplings", []), dtype=float)
    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    ax_right.plot(1e3 * x_vals[valid], y_vals[valid], "o", ms=5, label="extracted")

    slope = float(fit_results.get("coefficient", np.nan))
    intercept = float(fit_results.get("slope_fit_intercept", np.nan))
    if np.isfinite(slope) and np.isfinite(intercept) and np.any(valid):
        x_line = np.linspace(np.nanmin(x_vals[valid]), np.nanmax(x_vals[valid]), 200)
        y_line = slope * x_line + intercept
        ax_right.plot(1e3 * x_line, y_line, "--", lw=1.8, label="linear fit")

    r2 = float(fit_results.get("fit_quality", np.nan))
    ax_right.set_xlabel(f"d{drive_label} (mV)")
    ax_right.set_ylabel(f"t ({target_label}, arb.)")
    ax_right.set_title(f"slope fit (R^2={r2:.3f})" if np.isfinite(r2) else "slope fit")
    ax_right.grid(alpha=0.25)
    ax_right.legend(loc="best", fontsize=8)

    fig.suptitle(title or f"Barrier Diagnostics: {target_label} vs {drive_label}")
    fig.tight_layout()
    return fig


def plot_target_barrier_coupling_summary(
    target_barrier: str,
    fit_results_by_pair: Dict[str, Dict[str, Any]],
    title: Optional[str] = None,
) -> Figure:
    """Overlay extracted tunnel-coupling-vs-drive curves for one target barrier."""
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    plotted_any = False
    for _, fit in sorted(fit_results_by_pair.items(), key=lambda item: str(item[1].get("drive_barrier", ""))):
        drive_name = str(fit.get("drive_barrier", "drive"))
        x_vals = np.asarray(fit.get("drive_values", []), dtype=float)
        y_vals = np.asarray(fit.get("tunnel_couplings", []), dtype=float)
        valid = np.isfinite(x_vals) & np.isfinite(y_vals)
        if not np.any(valid):
            continue

        plotted_any = True
        ax.plot(1e3 * x_vals[valid], y_vals[valid], "o", ms=4, alpha=0.9, label=f"{drive_name} (data)")

        slope = float(fit.get("coefficient", np.nan))
        intercept = float(fit.get("slope_fit_intercept", np.nan))
        if np.isfinite(slope) and np.isfinite(intercept):
            x_line = np.linspace(np.nanmin(x_vals[valid]), np.nanmax(x_vals[valid]), 200)
            y_line = slope * x_line + intercept
            ax.plot(1e3 * x_line, y_line, "--", lw=1.5, alpha=0.95, label=f"{drive_name} (fit)")

    ax.set_title(title or f"{target_barrier}: extracted tunnel coupling vs drive")
    ax.set_xlabel("Drive barrier voltage (mV)")
    ax.set_ylabel("Tunnel coupling (arb.)")
    ax.grid(alpha=0.25)
    if plotted_any:
        ax.legend(loc="best", fontsize=8, ncol=1)
    else:
        ax.text(0.5, 0.5, "No valid fit data", ha="center", va="center", transform=ax.transAxes)
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
