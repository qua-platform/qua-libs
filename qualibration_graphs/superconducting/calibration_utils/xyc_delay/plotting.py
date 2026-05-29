"""Plotting utilities for XY-Coupler delay calibration visualizations."""

from typing import List

import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualibration_libs.plotting import grid_iter
from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from quam_builder.architecture.superconducting.qubit_pair import AnyTransmonPair


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubit_pairs: List[AnyTransmonPair],
    measured_qubits: List[AnyTransmon],
    fits: xr.Dataset,
) -> Figure:
    """Plot the relative delay scans for all qubit pairs, laid out by chip topology.

    Subplots are arranged using ``QubitPairGrid``, which positions each
    coupler between its two qubits on a doubled integer grid, mirroring the
    physical chip topology.

    Parameters
    ----------
    ds:
        Raw dataset whose ``qubit`` coordinate contains coupler/pair names.
    qubit_pairs:
        Ordered list of qubit-pair objects (same order as ``ds.qubit``).
    measured_qubits:
        Ordered list of the qubit being measured in each pair.
    fits:
        Fit dataset whose ``qubit`` coordinate also contains coupler/pair names.
    """
    measured_by_pair = {qp.name: q for qp, q in zip(qubit_pairs, measured_qubits)}

    grid_names, pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, pair_names, size=6)

    for ax, qubit in grid_iter(grid):
        pair_name = qubit["qubit"]
        q = measured_by_pair[pair_name]
        fit_q = fits.sel(qubit=pair_name) if fits is not None else None
        plot_individual_data_with_fit(ax, ds.sel(qubit=pair_name), fit_q)
        title = f"Measured qubit: {q.name}\nCoupler: {pair_name}"
        if "total_coupler_flux_mv" in ds.coords:
            sp = float(ds.total_coupler_flux_mv.sel(qubit=pair_name).values)
            title += f"\nTotal coupler flux: {sp:.1f} mV"
        ax.set_title(title)

    grid.fig.suptitle("XY-Coupler Z delay calibration")
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(ax: Axes, ds_q: xr.Dataset, fit_q: xr.Dataset = None) -> None:
    """Plot data for an individual qubit pair on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds_q : xr.Dataset
        Dataset already selected for one coupler/pair.
    fit_q : xr.Dataset, optional
        Fit dataset already selected for the same coupler/pair.
    """
    ds_q.difference.plot(ax=ax)
    if fit_q is not None and fit_q.success.data:
        fit_q.fit.plot(ax=ax)
        ax.axvline(fit_q.flux_delay.data, color="red", linestyle="--", label="fitted center")
        ax.legend()
