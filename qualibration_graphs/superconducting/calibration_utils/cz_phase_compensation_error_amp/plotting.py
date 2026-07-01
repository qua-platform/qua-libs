"""Plotting utilities for CZ phase compensation with error amplification."""

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibration_libs.plotting import grid_iter

from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names


def plot_raw_data_with_fit(
    ds_raw: xr.Dataset,
    qubit_pairs: list,
    ds_fit: xr.Dataset = None,
) -> Figure:
    """Plot averaged signal vs frame with sinc fits for every qubit pair on a chip-topology grid.

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
        If None, only axis labels and titles are shown.

    Returns
    -------
    Figure
        Matplotlib figure with one panel per qubit pair.
    """
    grid_names, pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, pair_names)

    for ax, qubit in grid_iter(grid):
        qp_name = qubit["qubit"]
        plot_individual_data_with_fit(ax, ds_raw, qp_name, ds_fit)

    grid.fig.suptitle("CZ 1Q phase compensation - error amplification")
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(
    ax: Axes,
    ds_raw: xr.Dataset,
    qp_name: str,
    ds_fit: xr.Dataset = None,
):
    """Plot averaged signal and sinc fits for control and target on one axis.

    Parameters
    ----------
    ax : Axes
        Axis to draw on.
    ds_raw : xr.Dataset
        Raw dataset with ``state_control`` / ``state_target`` or
        ``I_control`` / ``I_target`` variables.
    qp_name : str
        Qubit pair name used to select data.
    ds_fit : xr.Dataset, optional
        Fit dataset containing averaged curves, sinc fits, and fitted phases.
        If None or fit failed, only raw-data axis labels are applied.
    """
    if "state_control" in ds_raw.data_vars:
        ax.set_ylabel("Measured State")
    else:
        ax.set_ylabel("Rotated I Quadrature (V)")

    if ds_fit is None:
        ax.set_title(qp_name)
        ax.set_xlabel(r"x90 frame rotation [$\mathrm{rad}/2\pi$]")
        return

    qp_fit = ds_fit.sel(qubit_pair=qp_name)
    frames = ds_raw.sel(qubit_pair=qp_name).frame.values
    fit_success = bool(qp_fit.success.values) if "success" in qp_fit.data_vars else False

    plotted = False
    for label, color in (("control", "blue"), ("target", "red")):
        mean_var = f"{label}_mean_vs_frame"
        sinc_var = f"{label}_sinc_fit"
        phase_var = f"fitted_{label}_phase"

        if mean_var not in qp_fit.data_vars:
            continue

        mean_curve = qp_fit[mean_var].values
        ax.plot(
            frames,
            mean_curve,
            linestyle="none",
            marker="o",
            color=color,
            label=label.capitalize(),
        )
        plotted = True

        if sinc_var in qp_fit.data_vars:
            fit_vals = qp_fit[sinc_var].values
            if np.any(np.isfinite(fit_vals)):
                ax.plot(frames, fit_vals, color=color, alpha=0.5)

        if fit_success and phase_var in qp_fit.data_vars:
            fitted_phase = float(qp_fit[phase_var].values)
            if np.isfinite(fitted_phase):
                ax.axvline(fitted_phase, color=color, ls="--", lw=1.5, alpha=0.7)

    if not plotted:
        ax.text(
            0.5,
            0.5,
            "fit failed",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_title(qp_name)
    ax.set_xlabel(r"x90 frame rotation [$\mathrm{rad}/2\pi$]")
    if plotted:
        ax.legend(fontsize=8)
