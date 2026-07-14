"""Plotting utilities for two-qubit randomized benchmarking experiments."""

from __future__ import annotations

from typing import Dict, Literal

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibrate import QualibrationNode
from qualibration_libs.plotting import grid_iter

from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names
from calibration_utils.two_qubit_rb.fit_utils import rb_decay_curve
from calibration_utils.two_qubit_rb.reporting import format_fraction_pm

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
        alpha=0.4,
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
    node: QualibrationNode,
    *,
    title_prefix: str = "2Q Randomized Benchmarking",
    interleaved: bool | None = None,
) -> Dict[str, Figure]:
    """Plot RB survival curves on a chip-topology grid, one panel per qubit pair."""
    if interleaved is None:
        interleaved = "interleaved" in node.name.lower()

    ds_fit = node.results["ds_fit"]
    qubit_pairs = node.namespace["qubit_pairs"]
    use_input_stream = node.parameters.use_input_stream
    reset_type = node.parameters.reset_type
    plot_style = node.parameters.rb_plot_style
    log_x = node.parameters.rb_plot_log_x

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

    run_notes: list[str] = []
    if use_input_stream is not None:
        run_notes.append("with input stream" if use_input_stream else "without input stream")
    if reset_type is not None:
        run_notes.append(f"{reset_type} reset")

    suptitle = f"{title_prefix}\n({', '.join(run_notes)})" if run_notes else title_prefix
    grid.fig.suptitle(suptitle)
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
    color = "C1" if interleaved else "C0"
    marker = "^" if interleaved else "o"

    if "standard_rb_overlay_survival" in fr and np.isfinite(fr.standard_rb_overlay_survival.values).any():
        overlay_survival = fr.standard_rb_overlay_survival.values
        if plot_style == "per_sequence" and "standard_rb_overlay_survival_per_sequence" in fr:
            _plot_per_sequence_scatter(
                ax,
                depths,
                fr.standard_rb_overlay_survival_per_sequence.values,
                color="C0",
                zorder=1,
            )
            _plot_mean_marker(
                ax,
                depths,
                overlay_survival,
                color="C0",
                marker="o",
                label="StandardRB Mean",
                zorder=3,
            )
        else:
            _plot_mean_marker(
                ax,
                depths,
                overlay_survival,
                color="C0",
                marker="o",
                label="StandardRB Mean",
                zorder=3,
            )
        overlay_fitted = fr.standard_rb_overlay_fitted.values
        if success and np.isfinite(overlay_fitted).any():
            overlay_alpha = float(fr.standard_rb_fit_alpha.values)
            overlay_alpha_stderr = (
                float(fr.standard_rb_fit_alpha_stderr.values)
                if "standard_rb_fit_alpha_stderr" in fr and np.isfinite(fr.standard_rb_fit_alpha_stderr.values)
                else None
            )
            ax.plot(
                depths,
                overlay_fitted,
                color="C0",
                linestyle="--",
                zorder=4,
                label=(
                    f"StandardRB Fit (α = {overlay_alpha:.4f} ± {overlay_alpha_stderr:.4f})"
                    if overlay_alpha_stderr is not None
                    and np.isfinite(overlay_alpha_stderr)
                    and overlay_alpha_stderr > 0
                    else f"StandardRB Fit (α = {overlay_alpha:.4f})"
                ),
            )

    mean_label = "Interleaved Mean" if interleaved else "Mean"
    if plot_style == "per_sequence" and "survival_per_sequence" in fr:
        _plot_per_sequence_scatter(
            ax,
            depths,
            fr.survival_per_sequence.values,
            color=color,
            zorder=2,
        )
        _plot_mean_marker(
            ax,
            depths,
            survival,
            color=color,
            marker=marker,
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
            color=color,
            marker=marker,
            label=mean_label,
            zorder=5,
        )

    if success:
        smooth_depths = np.linspace(depths[0], depths[-1], 100)
        fit_alpha = float(fr.fit_alpha.values)
        alpha_stderr = (
            float(fr.alpha_stderr.values)
            if "alpha_stderr" in fr and np.isfinite(fr.alpha_stderr.values)
            else None
        )
        ax.plot(
            smooth_depths,
            rb_decay_curve(
                smooth_depths,
                float(fr.fit_amplitude.values),
                fit_alpha,
                float(fr.fit_offset.values),
            ),
            color=color,
            linestyle="--",
            zorder=6,
            label=(
                f"Exponential fit (α = {fit_alpha:.4f} ± {alpha_stderr:.4f})"
                if alpha_stderr is not None and np.isfinite(alpha_stderr) and alpha_stderr > 0
                else f"Exponential fit (α = {fit_alpha:.4f})"
            ),
        )

    if success:
        fidelity = float(fr.fidelity.values)
        fidelity_stderr = (
            float(fr.fidelity_stderr.values)
            if "fidelity_stderr" in fr and np.isfinite(fr.fidelity_stderr.values)
            else np.nan
        )
        epc = float(fr.epc.values) if "epc" in fr else np.nan
        if interleaved:
            epg = float(fr.epg.values) if "epg" in fr else np.nan
            coh_limit = (
                float(fr.coherence_limit_epg.values)
                if "coherence_limit_epg" in fr and np.isfinite(fr.coherence_limit_epg.values)
                else np.nan
            )
            stats = (
                f"CZ Gate Fidelity ($f_{{CZ}}$) = "
                f"{format_fraction_pm(fidelity, fidelity_stderr)}\n"
                f"Error Per Gate (EPG) = $1 - f_{{CZ}}$ = "
                f"{format_fraction_pm(epg, fidelity_stderr, as_error_rate=True)}\n"
                f"EPG (coherence limit) = {format_fraction_pm(coh_limit, as_error_rate=True)}"
            )
        else:
            stats = (
                f"2Q Clifford Fidelity ($f_c$) = {format_fraction_pm(fidelity, fidelity_stderr)}\n"
                f"Error Per Clifford (EPC) = $1 - f_c$ = "
                f"{format_fraction_pm(epc, fidelity_stderr, as_error_rate=True)}"
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
