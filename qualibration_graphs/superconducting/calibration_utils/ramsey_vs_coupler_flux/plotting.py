"""Plotting functions for Ramsey versus coupler flux calibration."""

from typing import List

import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibration_libs.plotting import grid_iter

from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names


def plot_raw_data(ds: xr.Dataset, qubit_pairs) -> Figure:
    """2D heatmap of qubit state vs (idle_time, coupler_flux) for each pair.

    The subplot layout is computed automatically from the qubit-pair
    grid locations using :class:`~calibration_utils.pair_grid.QubitPairGrid`.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with ``state`` variable, dims (qubit_pair, coupler_flux, idle_times).
    qubit_pairs : list
        Qubit pair objects (must expose ``.qubit_control`` / ``.qubit_target``
        with ``.grid_location`` and ``.name``).

    Returns
    -------
    Figure
    """
    g_names, qp_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(g_names, qp_names)
    for ax, qubit in grid_iter(grid):
        _plot_raw_panel(ax, ds, qubit)
    grid.fig.suptitle("Ramsey vs coupler flux — raw data")
    grid.fig.tight_layout()
    return grid.fig


def _plot_raw_panel(ax: Axes, ds: xr.Dataset, qubit: dict) -> None:
    qp_name = qubit["qubit"]
    ds.state.sel(qubit_pair=qp_name).plot(ax=ax)
    ax.set_title(qp_name)
    ax.set_xlabel("Idle time (ns)")
    ax.set_ylabel("Coupler flux (V)")


def plot_frequency_vs_coupler_flux(ds_fit: xr.Dataset, qubit_pairs) -> Figure:
    """Absolute qubit frequency vs coupler flux for each pair.

    The subplot layout is computed automatically from the qubit-pair
    grid locations using :class:`~calibration_utils.pair_grid.QubitPairGrid`.

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fitted dataset containing ``qubit_frequency`` (Hz).
    qubit_pairs : list
        Qubit pair objects (must expose ``.qubit_control`` / ``.qubit_target``
        with ``.grid_location`` and ``.name``).

    Returns
    -------
    Figure
    """
    g_names, qp_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(g_names, qp_names)
    for ax, qubit in grid_iter(grid):
        _plot_frequency_panel(ax, ds_fit, qubit)
    grid.fig.suptitle("Ramsey vs coupler flux — qubit frequency")
    grid.fig.tight_layout()
    return grid.fig


def _plot_frequency_panel(ax: Axes, ds_fit: xr.Dataset, qubit: dict) -> None:
    qp_name = qubit["qubit"]
    freq_ghz = ds_fit.qubit_frequency.sel(qubit_pair=qp_name) / 1e9
    freq_ghz.plot(ax=ax, linestyle="--", marker=".", label="Extracted freq")
    ax.set_title(qp_name)
    ax.set_xlabel("Coupler flux (V)")
    ax.set_ylabel("Qubit frequency (GHz)")
    ax.legend(fontsize="small")
