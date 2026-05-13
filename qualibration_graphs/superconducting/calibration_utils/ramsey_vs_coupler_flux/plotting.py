"""Plotting functions for Ramsey versus coupler flux calibration."""

import numpy as np
import xarray as xr
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
        Dataset with ``state`` variable, dims (qubit, coupler_flux, idle_times).
        Must have a ``measured_qubit_name`` coordinate on the ``qubit`` dimension.
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
        qp_name = qubit["qubit"]
        measured_name = str(ds.measured_qubit_name.sel(qubit=qp_name).values)
        ds.state.loc[qubit].plot(ax=ax)
        ax.set_title(f"{qp_name}\nMeasured: {measured_name}")
        ax.set_xlabel("Idle time (ns)")
        ax.set_ylabel("Coupler flux (V)")
    grid.fig.suptitle("Ramsey vs coupler flux — raw data")
    grid.fig.tight_layout()
    return grid.fig


def plot_fit_data(ds_fit: xr.Dataset, qubit_pairs) -> Figure | None:
    """2D heatmap of the fitted Ramsey oscillation for each pair.

    Skips pairs where all fits failed (all ``qubit_frequency`` values are NaN)
    and returns ``None`` if there is nothing to plot.

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fitted dataset containing ``fitted_state`` and ``qubit_frequency``.
        Must have a ``measured_qubit_name`` coordinate on the ``qubit`` dimension.
    qubit_pairs : list
        Qubit pair objects (must expose ``.qubit_control`` / ``.qubit_target``
        with ``.grid_location`` and ``.name``).

    Returns
    -------
    Figure or None
    """
    valid_pairs = [qp for qp in qubit_pairs if not np.isnan(ds_fit.qubit_frequency.sel(qubit=qp.name).values).all()]
    if not valid_pairs:
        return None

    g_names, qp_names = grid_pair_names(valid_pairs)
    grid = QubitPairGrid(g_names, qp_names)
    for ax, qubit in grid_iter(grid):
        qp_name = qubit["qubit"]
        measured_name = str(ds_fit.measured_qubit_name.sel(qubit=qp_name).values)
        ds_fit.fitted_state.loc[qubit].plot(ax=ax)
        ax.set_title(f"{qp_name}\nMeasured: {measured_name}")
        ax.set_xlabel("Idle time (ns)")
        ax.set_ylabel("Coupler flux (V)")
    grid.fig.suptitle("Ramsey vs coupler flux — fit data")
    grid.fig.tight_layout()
    return grid.fig


def plot_frequency_vs_coupler_flux(ds_fit: xr.Dataset, qubit_pairs) -> Figure | None:
    """Absolute qubit frequency vs coupler flux for each pair.

    The subplot layout is computed automatically from the qubit-pair
    grid locations using :class:`~calibration_utils.pair_grid.QubitPairGrid`.
    Pairs where all fits failed are skipped. Returns ``None`` if there is
    nothing to plot.

    Parameters
    ----------
    ds_fit : xr.Dataset
        Fitted dataset containing ``qubit_frequency`` (Hz).
        Must have a ``measured_qubit_name`` coordinate on the ``qubit`` dimension.
    qubit_pairs : list
        Qubit pair objects (must expose ``.qubit_control`` / ``.qubit_target``
        with ``.grid_location`` and ``.name``).

    Returns
    -------
    Figure or None
    """
    valid_pairs = [qp for qp in qubit_pairs if not np.isnan(ds_fit.qubit_frequency.sel(qubit=qp.name).values).all()]
    if not valid_pairs:
        return None

    g_names, qp_names = grid_pair_names(valid_pairs)
    grid = QubitPairGrid(g_names, qp_names)
    for ax, qubit in grid_iter(grid):
        qp_name = qubit["qubit"]
        measured_name = str(ds_fit.measured_qubit_name.sel(qubit=qp_name).values)
        freq_ghz = ds_fit.qubit_frequency.loc[qubit] / 1e9
        freq_ghz.plot(ax=ax, linestyle="--", marker=".", label="Extracted freq")
        ax.legend(fontsize="small")
        ax.set_title(f"{qp_name}\nMeasured: {measured_name}")
        ax.set_xlabel("Coupler flux (V)")
        ax.set_ylabel("Qubit frequency (GHz)")
    grid.fig.suptitle("Ramsey vs coupler flux — qubit frequency")
    grid.fig.tight_layout()
    return grid.fig
