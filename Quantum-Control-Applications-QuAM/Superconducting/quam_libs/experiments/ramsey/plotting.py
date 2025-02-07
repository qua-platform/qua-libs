from typing import List, Dict

import xarray as xr

from quam_libs.lib.fit import oscillation_decay_exp
from quam_libs.components import Transmon
from quam_libs.experiments.ramsey.analysis.fitting import RamseyFit
from quam_libs.experiments.ramsey.parameters import Parameters
from quam_libs.lib.plot_utils import QubitGrid, grid_iter


def plot_ramseys_data_with_fit(ds: xr.Dataset, qubits: List[Transmon],
                               node_parameters: Parameters, fits: Dict[str, RamseyFit]):
    """
    Plot qubit data for Ramsey experiments.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])

    for ax, qubit in grid_iter(grid):
        plot_ramsey_data_with_fit(ax, ds, qubit, node_parameters, fits.get(qubit["qubit"], None))

    grid.fig.suptitle("Ramsey vs. idle time")

    return grid.fig


def plot_ramsey_data_with_fit(ax, ds, qubit, node_parameters, fit=None):
    """
    Plot individual qubit data on a given axis.

    """
    if fit:
        fitted_ramsey_data = oscillation_decay_exp(
            ds.time,
            fit.raw_fit_results.sel(fit_vals="a"),
            fit.raw_fit_results.sel(fit_vals="f"),
            fit.raw_fit_results.sel(fit_vals="phi"),
            fit.raw_fit_results.sel(fit_vals="offset"),
            fit.raw_fit_results.sel(fit_vals="decay"),
        )
    else:
        fitted_ramsey_data = None

    if node_parameters.use_state_discrimination:
        plot_state(ax, ds, qubit, fitted_ramsey_data)
        ax.set_ylabel("State Population")
    else:
        plot_transmission_amplitude(ax, ds, qubit, fitted_ramsey_data)
        ax.set_ylabel("Trans. amp. I [mV]")

    ax.set_xlabel("Idle time [ns]")
    ax.set_title(qubit["qubit"])
    if fit is not None:
        add_fit_text(ax, fit)
    ax.legend()


def plot_state(ax, ds, qubit, fitted=None):
    """Plot state data for a qubit."""
    ds.sel(sign=1).loc[qubit].state.plot(
        ax=ax, x="time", c="C0", marker=".", ms=5.0, ls="", label="$\Delta$ = +"
    )
    ds.sel(sign=-1).loc[qubit].state.plot(
        ax=ax, x="time", c="C1", marker=".", ms=5.0, ls="", label="$\Delta$ = -"
    )
    if fitted is not None:
        ax.plot(ds.time, fitted.fit.loc[qubit].sel(sign=1), c="C0", ls="-", lw=1)
        ax.plot(ds.time, fitted.fit.loc[qubit].sel(sign=-1), c="C1", ls="-", lw=1)


def plot_transmission_amplitude(ax, ds, qubit, fitted=None):
    """Plot transmission amplitude for a qubit."""
    (ds.sel(sign=1).loc[qubit].I * 1e3).plot(
        ax=ax, x="time", c="C0", marker=".", ms=5.0, ls="", label="$\Delta$ = +"
    )
    (ds.sel(sign=-1).loc[qubit].I * 1e3).plot(
        ax=ax, x="time", c="C1", marker=".", ms=5.0, ls="", label="$\Delta$ = -"
    )
    if fitted is not None:
        ax.plot(ds.time, 1e3 * fitted.fit.loc[qubit].sel(sign=1), c="C0", ls="-", lw=1)
        ax.plot(ds.time, 1e3 * fitted.fit.loc[qubit].sel(sign=-1), c="C1", ls="-", lw=1)


def add_fit_text(ax, fit):
    """Add fit results text to the axis."""
    ax.text(
        0.1,
        0.9,
        f'T2* = {1e6 * fit.decay:.1f} ± {1e6 * fit.decay_error:.1f} µs',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )
