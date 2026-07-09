"""Plotting utilities for two-qubit randomized benchmarking experiments."""

from __future__ import annotations

from typing import Dict, Literal

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibration_libs.plotting import grid_iter

from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names
from calibration_utils.two_qubit_rb.analysis import format_error_rate, rb_decay_curve

RbPlotStyle = Literal["error_bars", "per_sequence"]


def _plot_per_sequence_scatter(
    ax: Axes,
    depths: np.ndarray,
    per_sequence: np.ndarray,
    *,
    color: str = "0.75",
    zorder: int = 1,
) -> None:
    """Scatter shot-averaged P(|00>) for each random sequence at every depth."""
    x = np.broadcast_to(depths[:, None], per_sequence.shape).ravel()
    y = np.asarray(per_sequence).ravel()
    finite = np.isfinite(y)
    if not finite.any():
        return
    ax.scatter(
        x[finite],
        y[finite],
        c=color,
        s=8,
        alpha=0.8,
        linewidths=0,
        zorder=zorder,
    )


def _plot_mean_with_sem(
    ax: Axes,
    depths: np.ndarray,
    survival: np.ndarray,
    stderr: np.ndarray,
    *,
    color: str,
    marker: str,
    label: str,
    zorder: int,
) -> None:
    ax.errorbar(
        depths,
        survival,
        yerr=stderr,
        fmt=marker,
        color=color,
        markersize=5,
        capsize=2,
        elinewidth=0.8,
        zorder=zorder,
        label=label,
    )


def _plot_mean_marker(
    ax: Axes,
    depths: np.ndarray,
    survival: np.ndarray,
    *,
    color: str,
    marker: str,
    label: str,
    zorder: int,
) -> None:
    ax.plot(
        depths,
        survival,
        marker,
        color=color,
        markersize=5,
        linestyle="none",
        zorder=zorder,
        label=label,
    )


def plot_raw_data_with_fit(
    ds_fit: xr.Dataset,
    qubit_pairs: list,
    *,
    interleaved: bool = False,
    title_prefix: str = "2Q Randomized Benchmarking",
    use_input_stream: bool | None = None,
    plot_style: RbPlotStyle = "error_bars",
    log_x: bool = False,
) -> Dict[str, Figure]:
    """Plot RB survival curves on a chip-topology grid, one panel per qubit pair."""
    grid_names, pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, pair_names)

    for ax, qubit in grid_iter(grid):
        qp_name = qubit["qubit"]
        plot_individual_data_with_fit(
            ax,
            ds_fit,
            qp_name,
            interleaved=interleaved,
            plot_style=plot_style,
            log_x=log_x,
        )

    if use_input_stream is not None:
        stream_label = "(with input stream)" if use_input_stream else "(without input stream)"
        title_prefix = f"{title_prefix}\n{stream_label}"

    grid.fig.suptitle(title_prefix)
    grid.fig.tight_layout()
    return {"rb": grid.fig}


def plot_individual_data_with_fit(
    ax: Axes,
    ds_fit: xr.Dataset,
    qp_name: str,
    *,
    interleaved: bool = False,
    plot_style: RbPlotStyle = "error_bars",
    log_x: bool = False,
) -> None:
    """Plot one qubit-pair RB survival curve using only the fitted dataset."""
    if qp_name not in ds_fit.qubit_pair.values:
        ax.text(0.5, 0.5, "No RB data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Qubit pair: {qp_name}")
        return

    fr = ds_fit.sel(qubit_pair=qp_name)
    depths = np.asarray(fr.circuit_depth.values, dtype=float)
    survival = fr.survival_probability.values
    success = bool(np.asarray(fr.success.values).item()) if "success" in fr else True

    if "standard_rb_overlay_survival" in fr and np.isfinite(fr.standard_rb_overlay_survival.values).any():
        overlay_survival = fr.standard_rb_overlay_survival.values
        if plot_style == "per_sequence" and "standard_rb_overlay_survival_per_sequence" in fr:
            _plot_per_sequence_scatter(
                ax,
                depths,
                fr.standard_rb_overlay_survival_per_sequence.values,
                color="0.82",
                zorder=1,
            )
            _plot_mean_marker(
                ax,
                depths,
                overlay_survival,
                color="green",
                marker="^",
                label="StandardRB Mean",
                zorder=3,
            )
        else:
            _plot_mean_marker(
                ax,
                depths,
                overlay_survival,
                color="green",
                marker="^",
                label="StandardRB Mean",
                zorder=3,
            )
        overlay_fitted = fr.standard_rb_overlay_fitted.values
        if success and np.isfinite(overlay_fitted).any():
            ax.plot(
                depths,
                overlay_fitted,
                color="green",
                linestyle="--",
                zorder=4,
                label=f"StandardRB Fit (alpha={float(fr.standard_rb_fit_alpha.values):.4f})",
            )

    mean_color = "C1" if interleaved else "C0"
    mean_label = "Interleaved Mean" if interleaved else "Mean"
    if plot_style == "per_sequence" and "survival_per_sequence" in fr:
        _plot_per_sequence_scatter(ax, depths, fr.survival_per_sequence.values, zorder=2)
        _plot_mean_marker(
            ax,
            depths,
            survival,
            color=mean_color,
            marker="o",
            label=mean_label,
            zorder=5,
        )
    else:
        stderr = fr.survival_stderr.values if "survival_stderr" in fr else np.zeros_like(survival)
        _plot_mean_with_sem(
            ax,
            depths,
            survival,
            stderr,
            color=mean_color,
            marker="o",
            label=mean_label,
            zorder=5,
        )

    if success:
        smooth_depths = np.linspace(depths[0], depths[-1], 100)
        ax.plot(
            smooth_depths,
            rb_decay_curve(
                smooth_depths,
                float(fr.fit_amplitude.values),
                float(fr.fit_alpha.values),
                float(fr.fit_offset.values),
            ),
            color="red",
            linestyle="--",
            zorder=6,
            label="Exponential Fit",
        )

    if success:
        fidelity = float(fr.fidelity.values)
        epc = float(fr.epc.values) if "epc" in fr else np.nan
        epg = float(fr.epg.values) if "epg" in fr else np.nan
        if interleaved:
            stats = (
                f"CZ Fidelity = {100 * fidelity:.2f}%\n"
                f"EPG = {format_error_rate(epg)}\n"
                f"EPC = {format_error_rate(epc)}"
            )
        else:
            avg_gate_fid = fr.average_gate_fidelity.values if "average_gate_fidelity" in fr else np.nan
            stats = (
                f"2Q Clifford Fidelity = {100 * fidelity:.2f}%, "
                f"EPC = {format_error_rate(epc)}\n"
                f"Single 2Q Gate Fidelity = {100 * float(avg_gate_fid):.2f}%, "
                f"EPG = {format_error_rate(epg)}"
            )
    else:
        stats = "Fit failed — see logs for validation issues"

    title = f"Qubit pair: {qp_name}" if success else f"Qubit pair: {qp_name} - fit failed"
    ax.set_title(f"{title}\n{stats}", fontsize=9, linespacing=1.3, pad=10)

    ax.set_xlabel("Circuit Depth")
    ax.set_ylabel(r"Probability to recover to $|00\rangle$")
    if log_x:
        ax.set_xscale("log")
    ax.legend(loc="upper right", framealpha=0.8, fontsize=8)
