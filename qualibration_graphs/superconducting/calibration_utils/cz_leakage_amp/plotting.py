"""Plotting module for CZ leakage amplification calibration."""

from typing import Dict

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibration_libs.plotting import grid_iter

from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names


def plot_raw_data_with_fit(
    ds_fit: xr.Dataset,
    qubit_pairs: list,
    node=None,
) -> Dict[str, Figure]:
    """Plot CZ leakage amplification data on chip-topology grids.

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fit dataset containing state-discrimination leakage data and ``optimal_amplitude``.
    qubit_pairs : list
        Qubit pair objects used for grid placement.
    node : optional
        Kept for backwards-compatible call sites.

    Returns
    -------
    dict[str, Figure]
        ``"raw"`` contains P(11) heatmaps and ``"mean"`` contains mean P(11)
        traces; both include optimal-amplitude markers.
    """
    grid_names, pair_names = grid_pair_names(qubit_pairs)
    figures = {}

    raw_grid = QubitPairGrid(grid_names, pair_names)
    for ax, qubit in grid_iter(raw_grid):
        qp_name = qubit["qubit"]
        plot_individual_raw_data_with_fit(ax, ds_fit, qp_name)
    raw_grid.fig.suptitle("CZ leakage amplification - raw P(11)")
    raw_grid.fig.tight_layout()
    figures["raw"] = raw_grid.fig

    mean_grid = QubitPairGrid(grid_names, pair_names)
    for ax, qubit in grid_iter(mean_grid):
        qp_name = qubit["qubit"]
        plot_individual_mean_data_with_fit(ax, ds_fit, qp_name)
    mean_grid.fig.suptitle("CZ leakage amplification - mean P(11)")
    mean_grid.fig.tight_layout()
    figures["mean"] = mean_grid.fig

    return figures


def plot_individual_raw_data_with_fit(
    ax: Axes,
    ds_fit: xr.Dataset,
    qp_name: str,
):
    """Plot one qubit-pair leakage-amplification heatmap."""
    fr = ds_fit.sel(qubit_pair=qp_name)

    if "state_moving_stationary" not in fr:
        ax.text(0.5, 0.5, "No leakage data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(qp_name)
        return

    data = fr.state_moving_stationary  # dims: amp, number_of_operations (or number_of_operations, amp)
    coupler_flux_mV = 1e3 * (fr.amp_full if "amp_full" in fr.coords else fr.amp).values
    n_ops = (
        fr.number_of_operations.values
        if "number_of_operations" in data.dims
        else np.arange(data.sizes.get("number_of_operations", data.shape[0]))
    )
    if "amp" in data.dims and "number_of_operations" in data.dims:
        z_data = data.transpose("number_of_operations", "amp").values
    else:
        z_data = data.values

    x_data, y_data = np.meshgrid(coupler_flux_mV, n_ops)
    pcm = ax.pcolormesh(
        x_data,
        y_data,
        z_data,
        cmap="viridis",
        shading="auto",
    )

    success = "success" in fr.coords and bool(fr.success) and np.isfinite(float(fr.optimal_amplitude))
    if success:
        opt_flux_mV = 1e3 * float(fr.optimal_amplitude)
        ax.axvline(opt_flux_mV, color="red", lw=2, label="optimal")

    ax.figure.colorbar(pcm, ax=ax, shrink=0.85).set_label("P(11)")
    ax.set_title(qp_name if success else f"{qp_name} - fit failed")
    ax.set_xlabel("Coupler flux (mV)")
    ax.set_ylabel("# CZ operations")
    if success:
        ax.legend(loc="upper right", fontsize=8)


def plot_individual_mean_data_with_fit(
    ax: Axes,
    ds_fit: xr.Dataset,
    qp_name: str,
):
    """Plot one qubit-pair mean P(11) trace."""
    fr = ds_fit.sel(qubit_pair=qp_name)

    if "mean_state_moving_stationary" not in fr:
        ax.text(0.5, 0.5, "No mean P(11) data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(qp_name)
        return

    mean_arr = fr.mean_state_moving_stationary
    coupler_flux_mV = 1e3 * (fr.amp_full if "amp_full" in fr.coords else fr.amp).values
    success = "success" in fr.coords and bool(fr.success) and np.isfinite(float(fr.optimal_amplitude))

    ax.plot(coupler_flux_mV, mean_arr.values, color="steelblue", lw=1.5, label="mean P(11)")
    if success:
        opt_flux_mV = 1e3 * float(fr.optimal_amplitude)
        opt_idx = int(np.nanargmin(np.abs(coupler_flux_mV - opt_flux_mV)))
        ax.axvline(opt_flux_mV, color="red", lw=2, label="optimal")
        ax.scatter([opt_flux_mV], [float(mean_arr.values[opt_idx])], color="red", zorder=5)
    ax.set_title(qp_name if success else f"{qp_name} - fit failed")
    ax.set_xlabel("Coupler flux (mV)")
    ax.set_ylabel("Mean P(11)")
    ax.set_ylim(
        max(0.0, float(np.nanmin(mean_arr.values)) - 0.05),
        min(1.0, float(np.nanmax(mean_arr.values)) + 0.05),
    )
    ax.legend(loc="best", fontsize=8)
