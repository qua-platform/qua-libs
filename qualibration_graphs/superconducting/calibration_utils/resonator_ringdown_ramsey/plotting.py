from typing import List

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

STATE_LABELS = {"g": "|g>", "e": "|e>"}


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the Ramsey phase vs. the delay after the readout pulse, with the exponential ring-down fit.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    qubits : List[AnyTransmon]
        List of qubits involved in the sequence.
    fits : xr.Dataset
        Dataset containing the fit results.

    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plots.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_data_with_fit(ax, qubit["qubit"], fits)
    grid.fig.suptitle("Resonator ring-down: Ramsey phase vs. delay after readout")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(ax: Axes, qubit: str, fits: xr.Dataset):
    """Plot the Ramsey phase of a single qubit for both preparations."""
    fit = fits.sel(qubit=qubit)
    for init_state in fit.init_state.values:
        selected = fit.sel(init_state=init_state)
        (line,) = ax.plot(selected.wait_time, np.rad2deg(selected.ramsey_phase), ".", label=STATE_LABELS[init_state])
        ax.plot(
            selected.wait_time,
            np.rad2deg(selected.ramsey_phase_fit),
            "-",
            color=line.get_color(),
            label=f"tau = {float(selected.tau):.0f} ns",
        )
    ax.axvline(float(fit.depletion_time), color="k", ls="--", label=f"depletion = {float(fit.depletion_time):.0f} ns")
    ax.set_xlabel("Delay after readout [ns]")
    ax.set_ylabel("Ramsey phase [deg]")
    ax.set_title(qubit)
    ax.legend(fontsize=7)


def plot_normalized_phase(qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the Ramsey phase of |g> and |e> vs. the delay after the readout pulse, each normalized from its shortest
    delay (1) to its late-delay baseline (0), so that the two preparations share one scale despite their 180 deg
    offset and different amplitudes. The un-normalized phase is in plot_raw_data_with_fit.
    """
    grid = QubitGrid(fits, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_normalized_phase(ax, qubit["qubit"], fits)
    grid.fig.suptitle("Resonator ring-down: normalized Ramsey phase vs. delay after readout")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_normalized_phase(ax: Axes, qubit: str, fits: xr.Dataset):
    """Plot the normalized Ramsey phase of a single qubit for both preparations."""
    fit = fits.sel(qubit=qubit)
    for init_state in fit.init_state.values:
        selected = fit.sel(init_state=init_state)
        ax.plot(selected.wait_time, selected.normalized_phase, ".-", label=STATE_LABELS[init_state])
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(float(fit.depletion_time), color="k", ls="--", label=f"depletion = {float(fit.depletion_time):.0f} ns")
    ax.set_xlabel("Delay after readout [ns]")
    ax.set_ylabel("Normalized Ramsey phase")
    ax.set_title(qubit)
    ax.legend(fontsize=7)


def plot_contrast(qubits: List[AnyTransmon], fits: xr.Dataset):
    """Plots the Ramsey contrast vs. the delay after the readout pulse, for both preparations."""
    grid = QubitGrid(fits, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        fit = fits.sel(qubit=qubit["qubit"])
        for init_state in fit.init_state.values:
            ax.plot(fit.wait_time, fit.contrast.sel(init_state=init_state), ".-", label=STATE_LABELS[init_state])
        ax.set_xlabel("Delay after readout [ns]")
        ax.set_ylabel("Ramsey contrast")
        ax.set_title(qubit["qubit"])
        ax.legend(fontsize=7)
    grid.fig.suptitle("Resonator ring-down: Ramsey contrast vs. delay after readout")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig
