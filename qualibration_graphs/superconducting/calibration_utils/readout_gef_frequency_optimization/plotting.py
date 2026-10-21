"""Plotting utilities for GEF readout frequency optimization visualizations."""

from typing import List

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def _with_iq_abs(ds: xr.Dataset) -> xr.Dataset:
    """Add |IQ| for g, e, and f branches when not already present."""
    if "IQ_abs_g" in ds.data_vars:
        return ds
    return ds.assign(
        IQ_abs_g=np.sqrt(ds.Ig**2 + ds.Qg**2),
        IQ_abs_e=np.sqrt(ds.Ie**2 + ds.Qe**2),
        IQ_abs_f=np.sqrt(ds.If**2 + ds.Qf**2),
    )


def plot_distances_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """Plot pairwise g/e/f IQ centroid distances and their minimum vs readout detuning."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_distance_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("GEF readout frequency optimization (distance)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_IQ_abs_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """Plot averaged |IQ| for |g>, |e>, and |f> vs readout detuning for each qubit."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    ds = _with_iq_abs(ds)
    for ax, qubit in grid_iter(grid):
        plot_individual_IQ_abs_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("GEF readout frequency optimization (IQ amplitude)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_distance_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """Plot Dge, Def, Dgf, and min distance vs detuning with the fitted optimum."""
    fit_q = fit.assign_coords(freq_MHz=fit.frequency / 1e6)
    (1e3 * fit_q.Dge).plot(ax=ax, x="freq_MHz", label="GE")
    (1e3 * fit_q.Def).plot(ax=ax, x="freq_MHz", label="EF")
    (1e3 * fit_q.Dgf).plot(ax=ax, x="freq_MHz", label="GF")
    (1e3 * fit_q.Distance).plot(ax=ax, x="freq_MHz")
    if fit is not None and "optimal_detuning" in fit:
        ax.axvline(float(fit.optimal_detuning) / 1e6, color="red", linestyle="--")
    ax.set_title(str(qubit["qubit"]))
    ax.set_xlabel("Frequency detuning [MHz]")
    ax.set_ylabel("Distance between IQ blobs [mV]")
    ax.legend(loc="best")


def plot_individual_IQ_abs_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """Plot averaged |IQ| for g, e, and f states vs readout detuning."""
    ds_q = _with_iq_abs(ds).assign_coords(freq_MHz=ds.frequency / 1e6).loc[qubit]
    (1e3 * ds_q.IQ_abs_g).plot(ax=ax, x="freq_MHz", label="g.s.")
    (1e3 * ds_q.IQ_abs_e).plot(ax=ax, x="freq_MHz", label="e.s.")
    (1e3 * ds_q.IQ_abs_f).plot(ax=ax, x="freq_MHz", label="f.s.")
    if fit is not None and "optimal_detuning" in fit:
        ax.axvline(float(fit.optimal_detuning) / 1e6, color="red", linestyle="--")
    ax.set_title(str(qubit["qubit"]))
    ax.set_xlabel("Frequency detuning [MHz]")
    ax.set_ylabel("Resonator response [mV]")
    ax.legend(loc="best")
