from typing import List, Dict

import xarray as xr

from quam_experiments.analysis.fit import decay_exp
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from quam_experiments.experiments.ramsey.analysis.fitting import RamseyFit
from quam_experiments.experiments.ramsey.parameters import Parameters
from quam_libs.plot_utils import QubitGrid, grid_iter


def plot_t1s_data_with_fit(
    ds: xr.Dataset, qubits: List[AnyTransmon], node_parameters: Parameters, fits: Dict[str, RamseyFit]
):
    """
    Plot qubit data for T1 experiments.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])

    for ax, qubit in grid_iter(grid):
        plot_t1_data_with_fit(ax, ds, qubit, node_parameters, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("T1 vs. idle time")
    grid.fig.set_size_inches(15, 9)
    return grid.fig


def plot_t1_data_with_fit(ax, ds, qubit, node_parameters, fit=None):
    """
    Plot individual qubit data on a given axis.

    """
    if fit:
        fitted_t1_data = decay_exp(
            ds.idle_time,
            fit.sel(fit_vals="a"),
            fit.sel(fit_vals="offset"),
            fit.sel(fit_vals="decay"),
        )
    else:
        fitted_t1_data = None

    if node_parameters.use_state_discrimination:
        plot_state(ax, ds, qubit, fitted_t1_data)
        ax.set_ylabel("State Population")
    else:
        plot_transmission_amplitude(ax, ds, qubit, fitted_t1_data)
        ax.set_ylabel("Trans. amp. I [mV]")

    ax.set_xlabel("Idle time [ns]")
    ax.set_title(qubit["qubit"])
    if fit is not None:
        add_fit_text(ax, fit)


def plot_state(ax, ds, qubit, fitted=None):
    """Plot state data for a qubit."""
    ds.sel(qubit=qubit["qubit"]).state.plot(ax=ax)
    if fitted is not None:
        ax.plot(ds.idle_time, fitted.fit_results, "r--")

def plot_transmission_amplitude(ax, ds, qubit, fitted=None):
    """Plot transmission amplitude for a qubit."""
    (ds.sel(qubit=qubit["qubit"]).I * 1e3).plot(ax=ax)
    if fitted is not None:
        ax.plot(ds.idle_time, 1e3 * fitted.fit_results, "r--")


def add_fit_text(ax, fit):
    """Add fit results text to the axis."""
    ax.text(
        0.1,
        0.9,
        f'T1 = {fit.tau.values:.1f} ± {fit.tau_error.values:.1f} µs\nSuccess: {fit.success.values}',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )
