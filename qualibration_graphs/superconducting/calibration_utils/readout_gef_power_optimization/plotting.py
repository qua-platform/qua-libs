"""Plotting utilities for GEF readout power optimization."""

from typing import List

import xarray as xr
from matplotlib.axes import Axes
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def plot_distances_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """Plot minimum g/e/f centroid distance vs readout amplitude for each qubit."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_distance_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("GEF readout power optimization (minimum centroid distance)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_distance_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """Plot per-qubit minimum centroid distance and mark the selected optimum."""
    del ds
    fit.Distance.plot(ax=ax, x="readout_amplitude", label="raw")
    fit.Distance_smooth.plot(ax=ax, x="readout_amplitude", ls="--", lw=1.0, alpha=0.7, label="smoothed (3pt)")
    ax.axvline(float(fit.optimal_amplitude), color="k", linestyle=":", lw=0.8, label="optimal")
    ax.set_xlabel("Readout amplitude [V]")
    ax.set_ylabel("Minimum centroid distance [V]")

    base_amp = float((fit.readout_amplitude / fit.amp_prefactor).median())
    secax = ax.secondary_xaxis("top", functions=(lambda a: a / base_amp, lambda p: p * base_amp))
    secax.set_xlabel("Amplitude prefactor")

    ax.legend(fontsize=8)
    ax.set_title(
        f"{qubit['qubit']}  (opt {float(fit.optimal_amplitude) * 1e3:.1f} mV, "
        f"x{float(fit.optimal_amp_prefactor):.3f})"
    )
