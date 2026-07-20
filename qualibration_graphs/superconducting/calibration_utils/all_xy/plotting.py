"""Plotting utilities for All-XY calibration."""

from typing import List, Optional

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

# Labels for the 21 All-XY sequences (for x-axis). Order matches ALL_XY_SEQUENCES in the node.
ALL_XY_LABELS = [
    "I,I",
    "x180,x180",
    "y180,y180",
    "x180,y180",
    "y180,x180",
    "x90,I",
    "y90,I",
    "x90,y90",
    "y90,x90",
    "x90,y180",
    "y90,x180",
    "x180,y90",
    "y180,x90",
    "x90,x180",
    "x180,x90",
    "y90,y180",
    "y180,y90",
    "x180,I",
    "y180,I",
    "x90,x90",
    "y90,y90",
]
# Ideal outcomes: 5 excited (max), 12 superposition (mean), 4 ground (min)
N_EXCITED, N_SUPERPOSITION, N_GROUND = 4, 12, 5


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: Optional[xr.Dataset] = None,
) -> Figure:
    """
    Plot All-XY I and Q (or state) vs sequence index with expected ideal values.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with coord sequence_index and per-qubit I/Q or state variables.
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
    """Plot I and Q vs sequence index for one qubit, with expected-value reference."""
    n_seq = len(ALL_XY_LABELS)
    x = np.arange(n_seq)

    qubit_name = qubit_info["qubit"]

    # Determine the numeric index of this qubit in the dataset's qubit coordinate,
    # used as a fallback for numbered variable names (state1, state2, ...).
    if "qubit" in ds.coords:
        qubit_names = list(ds.coords["qubit"].values)
        q_idx = qubit_names.index(qubit_name) if qubit_name in qubit_names else 0
    else:
        q_idx = 0

    state_var = f"state{q_idx + 1}"
    i_var = f"I{q_idx + 1}"
    q_var = f"Q{q_idx + 1}"

    if "state" in ds.data_vars and "qubit" in ds["state"].dims:
        state = ds["state"].sel(qubit=qubit_name).values
        ax.plot(x, state, "bo-", label="State")
        vmin, vmean, vmax = float(np.min(state)), float(np.mean(state)), float(np.max(state))
        expected = [vmin] * N_GROUND + [vmean] * N_SUPERPOSITION + [vmax] * N_EXCITED
        ax.plot(x, expected, "r-", label="Expected", alpha=0.8)
        ax.set_ylabel("State population")
    elif state_var in ds.data_vars:
        state = ds[state_var].values
        ax.plot(x, state, "bo-", label="State")
        vmin, vmean, vmax = float(np.min(state)), float(np.mean(state)), float(np.max(state))
        expected = [vmin] * N_GROUND + [vmean] * N_SUPERPOSITION + [vmax] * N_EXCITED
        ax.plot(x, expected, "r-", label="Expected", alpha=0.8)
        ax.set_ylabel("State population")
    else:
        I_vals = ds[i_var].values
        Q_vals = ds[q_var].values
        ax.plot(x, I_vals, "bo-", label="I")
        ax.plot(x, Q_vals, "go-", label="Q")
        imin, imean, imax = float(np.min(I_vals)), float(np.mean(I_vals)), float(np.max(I_vals))
        qmin, qmean, qmax = float(np.min(Q_vals)), float(np.mean(Q_vals)), float(np.max(Q_vals))
        expected_I = [imax] * N_GROUND + [imean] * N_SUPERPOSITION + [imin] * N_EXCITED
        expected_Q = [qmax] * N_GROUND + [qmean] * N_SUPERPOSITION + [qmin] * N_EXCITED
        ax.plot(x, expected_I, "r-", label="Expected I", alpha=0.8)
        ax.plot(x, expected_Q, "m-", label="Expected Q", alpha=0.8)
        ax.set_ylabel("Quadrature [a.u.]")

    ax.set_xlabel("Sequence")
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_XY_LABELS, rotation=45, ha="right")
    ax.set_title(qubit_name)
    ax.legend()
