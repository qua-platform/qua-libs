"""Plotting utilities for the G-E-F readout duration optimization.

``plot_fidelity_vs_duration`` shows the three-state assignment fidelity and the per-state
assignment probabilities against integration duration, with the tolerance band and the chosen
duration marked. The IQ blobs and the confusion matrix at the chosen duration are plotted by the
existing ``calibration_utils.iq_blobs_ef`` helpers, which the node calls directly.
"""

from typing import List

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def plot_fidelity_vs_duration(
    ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset, fidelity_tolerance: float
) -> Figure:
    """Grid of per-qubit assignment fidelity against integration duration."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_fidelity_vs_duration(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]), fidelity_tolerance)
    handles, labels = ax.get_legend_handles_labels()
    grid.fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    grid.fig.suptitle("G-E-F readout optimization: assignment fidelity vs integration duration")
    grid.fig.set_size_inches(15, 9)
    # Leave room at the bottom for the shared legend, which would otherwise cover the x-axis labels
    grid.fig.tight_layout(rect=(0, 0.05, 1, 1))
    return grid.fig


def plot_individual_fidelity_vs_duration(
    ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset, fidelity_tolerance: float
):
    """Per-state and mean assignment fidelity of one qubit, with the chosen duration marked."""
    del ds
    for state in fit.prepared.values:
        p = fit.confusion_matrix.sel(prepared=state, measured=state)
        ax.plot(fit.duration, 100 * p, ".-", lw=0.8, alpha=0.7, label=f"P({state}|{state})")
    ax.plot(fit.duration, 100 * fit.assignment_fidelity, "o-", color="k", label="assignment fidelity")

    best_fidelity = float(fit.best_fidelity)
    if np.isfinite(best_fidelity):
        ax.axhspan(100 * (best_fidelity - fidelity_tolerance), 100 * best_fidelity, color="grey", alpha=0.2)
    if bool(fit.success):
        ax.axvline(float(fit.optimal_duration), color="k", linestyle="dashed", label="chosen duration")
        title = f"{qubit['qubit']}  ({float(fit.optimal_duration):.0f} ns, F={100 * float(fit.optimal_fidelity):.2f}%)"
    else:
        title = f"{qubit['qubit']}  (failed)"

    ax.set_xlabel("Integration duration [ns]")
    ax.set_ylabel("Assignment probability [%]")
    ax.set_title(title, fontsize=9)
