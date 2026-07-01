"""Plotting utilities for CZ phase compensation with error amplification."""

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibration_libs.plotting import grid_iter

from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names

FRAME_XLABEL = r"Virtual-Z frame [$2\pi$]"

_PANEL_STYLES = {
    "control": {
        "data": {"color": "#3b82f6", "markeredgecolor": "#1e3a8a"},
        "fit": {"color": "#ea580c"},
        "optimum": {"color": "#15803d"},
    },
    "target": {
        "data": {"color": "#ef4444", "markeredgecolor": "#991b1b"},
        "fit": {"color": "#7c3aed"},
        "optimum": {"color": "#15803d"},
    },
}


def _signal_labels(ds_raw: xr.Dataset) -> tuple[str, str, str]:
    """Return y-axis label, mean-curve legend, and peak legend for state vs IQ readout."""
    if "state_control" in ds_raw.data_vars:
        return "Measured State", r"$\langle state \rangle_N$", r"peak $\langle state \rangle_N$"
    return "Rotated I Quadrature (V)", r"$\langle I \rangle_N$", r"peak $\langle I \rangle_N$"


def _split_axis_vertically(ax: Axes, hspace: float = 0.35) -> tuple[Axes, Axes]:
    """Replace ``ax`` with stacked top (control) and bottom (target) subplots."""
    sub_gs = ax.get_subplotspec().subgridspec(2, 1, hspace=hspace)
    fig = ax.figure
    ax.remove()
    ax_control = fig.add_subplot(sub_gs[0])
    ax_target = fig.add_subplot(sub_gs[1], sharex=ax_control)
    return ax_control, ax_target


def plot_raw_data_with_fit(
    ds_raw: xr.Dataset,
    qubit_pairs: list,
    ds_fit: xr.Dataset = None,
) -> Figure:
    """Plot N-averaged signal vs frame for control and target on a chip-topology grid.

    Each qubit-pair cell is split into two stacked panels (control on top, target below),
    with sinc fits and fitted phase markers when available.

    Parameters
    ----------
    ds_raw : xr.Dataset
        Raw dataset containing ``state_control`` / ``state_target`` or
        ``I_control`` / ``I_target`` variables with ``frame`` and
        ``number_of_operations`` dimensions.
    qubit_pairs : list
        Qubit pair objects used for grid placement.
    ds_fit : xr.Dataset, optional
        Fit dataset containing ``control_mean_vs_frame``, ``target_mean_vs_frame``,
        ``control_sinc_fit``, ``target_sinc_fit``, and ``success``.

    Returns
    -------
    Figure
        Matplotlib figure with two panels per qubit pair on the chip grid.
    """
    grid_names, pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, pair_names, size=4)
    width, height = grid.fig.get_size_inches()
    grid.fig.set_size_inches(width, height * 2)

    for ax, qubit in grid_iter(grid):
        qp_name = qubit["qubit"]
        ax_control, ax_target = _split_axis_vertically(ax)
        plot_individual_channel_with_fit(
            ax_control,
            ds_raw,
            qp_name,
            ds_fit,
            channel="control",
            title=f"{qp_name} — control",
            show_xlabel=False,
        )
        plot_individual_channel_with_fit(
            ax_target,
            ds_raw,
            qp_name,
            ds_fit,
            channel="target",
            title=f"{qp_name} — target",
            show_xlabel=True,
        )

    grid.fig.suptitle("CZ phase compensation - error amplification (sinc fit)", y=0.98)
    grid.fig.tight_layout(rect=(0, 0, 1, 0.96))
    return grid.fig


def plot_individual_channel_with_fit(
    ax: Axes,
    ds_raw: xr.Dataset,
    qp_name: str,
    ds_fit: xr.Dataset | None,
    channel: str,
    title: str | None = None,
    show_xlabel: bool = True,
):
    """Plot one qubit's N-averaged signal, sinc fit, and fitted phase for a single pair."""
    style = _PANEL_STYLES[channel]
    ylabel, mean_label, peak_label = _signal_labels(ds_raw)
    panel_title = title if title is not None else f"{qp_name} - {channel}"

    if ds_fit is None:
        ax.set_title(panel_title)
        if show_xlabel:
            ax.set_xlabel(FRAME_XLABEL)
        else:
            ax.tick_params(labelbottom=False)
        ax.set_ylabel(ylabel)
        ax.text(0.5, 0.5, "no fit", ha="center", va="center", transform=ax.transAxes)
        return

    qp_fit = ds_fit.sel(qubit_pair=qp_name)
    if not bool(qp_fit.success.values):
        ax.set_title(panel_title)
        if show_xlabel:
            ax.set_xlabel(FRAME_XLABEL)
        else:
            ax.tick_params(labelbottom=False)
        ax.set_ylabel(ylabel)
        ax.text(0.5, 0.5, "fit failed", ha="center", va="center", transform=ax.transAxes)
        return

    frames = ds_raw.sel(qubit_pair=qp_name).frame.values
    mean_var = f"{channel}_mean_vs_frame"
    sinc_var = f"{channel}_sinc_fit"
    phase_var = f"fitted_{channel}_phase"
    peak_mean_var = f"{channel}_mean_at_peak"
    method_var = f"{channel}_fit_method"

    if mean_var in qp_fit.data_vars:
        mean_curve = qp_fit[mean_var].values
        fitted_phase = float(qp_fit[phase_var].values)
        peak_mean = float(qp_fit[peak_mean_var].values)
        fit_method = str(qp_fit[method_var].values)

        ax.plot(
            frames,
            mean_curve,
            linestyle="none",
            marker="o",
            ms=5,
            mew=0.8,
            alpha=0.9,
            zorder=2,
            label=mean_label,
            **style["data"],
        )
        if sinc_var in qp_fit.data_vars:
            fit_vals = qp_fit[sinc_var].values
            if np.any(np.isfinite(fit_vals)):
                ax.plot(
                    frames,
                    fit_vals,
                    "-",
                    lw=2.5,
                    zorder=4,
                    label="sinc fit",
                    **style["fit"],
                )
        if np.isfinite(fitted_phase):
            ax.axvline(
                fitted_phase,
                color=style["optimum"]["color"],
                ls="--",
                lw=1.8,
                alpha=0.95,
                zorder=3,
                label=f"phase = {fitted_phase:.4f} ({fit_method})",
            )
        if np.isfinite(peak_mean) and np.isfinite(fitted_phase):
            ax.plot(
                fitted_phase,
                peak_mean,
                "*",
                ms=14,
                mew=1.2,
                zorder=5,
                markerfacecolor="white",
                markeredgecolor=style["fit"]["color"],
                label=f"{peak_label} = {peak_mean:.4f}",
            )
        ax.legend(loc="best", fontsize=9)

    ax.set_title(panel_title)
    if show_xlabel:
        ax.set_xlabel(FRAME_XLABEL)
    else:
        ax.tick_params(labelbottom=False)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, color="#94a3b8")
