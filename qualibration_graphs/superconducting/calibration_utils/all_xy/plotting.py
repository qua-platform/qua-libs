"""Plotting utilities for All-XY calibration."""

from typing import List, Optional

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

from .sequences import ALL_XY_LABELS, N_EXCITED, N_GROUND, N_SUPERPOSITION

# Ideal outcomes along sequence index: ground, superposition, then excited.


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: Optional[xr.Dataset] = None,
) -> Figure:
    """
    Plot All-XY state (or IQ amplitude) vs sequence index with expected ideal values.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with coord sequence_index and per-qubit state or I/Q (IQ_abs) variables.
    qubits : list of AnyTransmon
        Qubits to plot.
    fits : xr.Dataset, optional
        Fit results (unused for All-XY, kept for API compatibility).

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    del fits  # unused
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit_info in grid_iter(grid):
        _plot_one_qubit_all_xy(ax, ds, qubit_info)

    grid.fig.suptitle("All-XY")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def _plot_one_qubit_all_xy(
    ax: Axes,
    ds: xr.Dataset,
    qubit_info: dict,
) -> None:
    """Plot state or IQ amplitude vs sequence index for one qubit, with expected reference."""
    n_seq = len(ALL_XY_LABELS)
    x = np.arange(n_seq)
    qubit_name = qubit_info["qubit"]

    if "state" in ds.data_vars:
        y = ds["state"].sel(qubit=qubit_name).values
        ylabel = "State population"
        label = "State"
    else:
        y = ds["IQ_abs"].sel(qubit=qubit_name).values
        ylabel = "IQ amplitude [V]"
        label = "IQ_abs"

    ax.plot(x, y, "bo-", label=label)
    vmin, vmean, vmax = float(np.min(y)), float(np.mean(y)), float(np.max(y))
    expected = [vmin] * N_GROUND + [vmean] * N_SUPERPOSITION + [vmax] * N_EXCITED
    ax.plot(x, expected, "r-", label="Expected", alpha=0.8)
    ax.set_ylabel(ylabel)

    ax.set_xlabel("Sequence")
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_XY_LABELS, rotation=45, ha="right")
    ax.set_title(qubit_name)
    ax.legend()
