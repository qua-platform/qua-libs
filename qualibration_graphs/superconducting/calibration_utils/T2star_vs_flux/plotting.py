"""Plotting for the T2*-versus-flux (Ramsey dephasing) node."""

from typing import List

import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def _signal_name(ds: xr.Dataset) -> str:
    if hasattr(ds, "state"):
        return "state"
    if hasattr(ds, "I"):
        return "I"
    raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """Plot the 2-D Ramsey-fringe map (idle time vs flux bias) for each qubit.

    A horizontal dashed line marks the flux bias giving the longest T2*.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        _plot_individual_map(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("T2* vs flux (Ramsey-fringe map)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_t2_star_vs_flux(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """Plot the extracted T2* versus flux bias (with error bars) for each qubit."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        _plot_individual_curve(ax, fits.sel(qubit=qubit["qubit"]), qubit)

    grid.fig.suptitle("T2* vs flux")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def _plot_individual_map(ax: Axes, ds: xr.Dataset, qubit: dict, fit: xr.Dataset = None):
    signal = _signal_name(ds)
    da = ds.sel(qubit=qubit["qubit"])[signal]
    if signal == "I":
        da = da * 1e3
        cbar_label = "Trans. amp. I [mV]"
    else:
        cbar_label = "State"

    # A 2-D map needs both axes to have >1 point; otherwise (e.g. a single flux point)
    # degrade gracefully to a 1-D line of signal vs idle time instead of crashing.
    two_d = da.sizes.get("flux_bias", 1) > 1 and da.sizes.get("idle_time", 1) > 1
    if two_d:
        da.plot(ax=ax, x="idle_time", y="flux_bias", add_colorbar=True, cbar_kwargs={"label": cbar_label})
        if fit is not None:
            valid = np.isfinite(fit["T2_star"].values)
            if valid.any():
                idx = int(np.nanargmax(np.where(valid, fit["T2_star"].values, -np.inf)))
                best_flux = float(fit.flux_bias.values[idx])
                ax.axhline(best_flux, color="red", linestyle="--", linewidth=1, label="max T2*")
    else:
        da.squeeze().plot(ax=ax)

    # Set labels AFTER plotting so they override xarray's auto-generated labels
    ax.set_title(qubit["qubit"])
    ax.set_xlabel("Idle time [ns]")
    ax.set_ylabel("Flux bias [V]" if two_d else cbar_label)


def _plot_individual_curve(ax: Axes, fit: xr.Dataset, qubit: dict):
    flux = fit.flux_bias.values
    tau_us = fit["T2_star"].values * 1e-3  # ns -> us
    err_us = fit["T2_star_error"].values * 1e-3

    ax.errorbar(flux, tau_us, yerr=err_us, fmt="o-", capsize=3, markersize=4)

    valid = np.isfinite(tau_us)
    if valid.any():
        idx = int(np.nanargmax(np.where(valid, tau_us, -np.inf)))
        best_flux = float(flux[idx])
        best_t2 = float(tau_us[idx])
        ax.axvline(best_flux, color="red", linestyle="--", linewidth=1)
        ax.text(
            0.05,
            0.95,
            f"max T2* = {best_t2:.1f} us\n@ flux = {best_flux:.4f} V",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox={"facecolor": "white", "alpha": 0.5},
        )

    ax.set_title(qubit["qubit"])
    ax.set_xlabel("Flux bias [V]")
    ax.set_ylabel("T2* [us]")
    ax.set_ylim(bottom=0)  # T2* is non-negative; pin the y-axis floor to 0
