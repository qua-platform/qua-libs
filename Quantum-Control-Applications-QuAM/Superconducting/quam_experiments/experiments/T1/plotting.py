from typing import List
import xarray as xr
from quam_experiments.analysis.fit import decay_exp
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from quam_experiments.experiments.ramsey.parameters import Parameters
from qualibration_libs.plot_utils import QubitGrid, grid_iter


def plot_t1s_data_with_fit(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    node_parameters: Parameters,
    fits: xr.Dataset,
):
    """
    Plots T1 data with fit results for multiple qubits.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    qubits : List[AnyTransmon]
        List of qubits involved in the sequence.
    node_parameters : Parameters
        Parameters related to the node.
    fits : xr.Dataset
        Dataset containing the fit results for the T1 data.

    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plots.

    Example:
    --------
        >>> fig = plot_t1s_data_with_fit(ds, qubits, node_parameters, fits)
        >>> fig.show()
    """
    grid = QubitGrid(ds.to_dataarray(), [q.grid_location for q in qubits])

    for ax, qubit in grid_iter(grid):
        _plot_t1_data_with_fit(ax, ds, qubit, node_parameters, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("T1 vs. idle time")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def _plot_t1_data_with_fit(ax, ds, qubit, node_parameters, fit=None):
    """Plot individual qubit data on a given axis."""
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
        _plot_state(ax, ds, qubit, fitted_t1_data)
        ax.set_ylabel("State Population")
    else:
        _plot_transmission_amplitude(ax, ds, qubit, fitted_t1_data)
        ax.set_ylabel("Trans. amp. I [mV]")

    ax.set_xlabel("Idle time [ns]")
    ax.set_title(qubit["qubit"])
    if fit is not None:
        _add_fit_text(ax, fit)


def _plot_state(ax, ds, qubit, fitted=None):
    """Plot state data for a qubit."""
    ds.sel(qubit=qubit["qubit"]).state.plot(ax=ax)
    if fitted is not None:
        ax.plot(ds.idle_time, fitted.ds_fit, "r--")


def _plot_transmission_amplitude(ax, ds, qubit, fitted=None):
    """Plot transmission amplitude for a qubit."""
    (ds.sel(qubit=qubit["qubit"]).I * 1e3).plot(ax=ax)
    if fitted is not None:
        ax.plot(ds.idle_time, 1e3 * fitted.ds_fit, "r--")


def _add_fit_text(ax, fit):
    """Add fit results text to the axis."""
    ax.text(
        0.1,
        0.9,
        f"T1 = {1e-3 * fit.tau.values:.1f} ± {1e-3 * fit.tau_error.values:.1f} µs\nSuccess: {fit.success.values}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )
