from typing import List
import xarray as xr
from matplotlib.axes import Axes
from qualibration_libs.analysis import decay_exp
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from qualibration_libs.plotting import QubitGrid, grid_iter


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
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
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])

    for ax, qubit in grid_iter(grid):
        plot_individual_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("T1 vs. idle time")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """Plot individual qubit data on a given axis."""
    if fit:
        fitted = decay_exp(
            ds.idle_time,
            fit.fit_data.sel(fit_vals="a"),
            fit.fit_data.sel(fit_vals="offset"),
            fit.fit_data.sel(fit_vals="decay"),
        )
    else:
        fitted = None

    if hasattr(fit, "state"):
        ds.sel(qubit=qubit["qubit"]).state.plot(ax=ax)
        if fitted is not None:
            ax.plot(ds.idle_time, fitted, "r--")
        ax.set_ylabel("State")
    elif hasattr(fit, "I"):
        (ds.sel(qubit=qubit["qubit"]).I * 1e3).plot(ax=ax)
        if fitted is not None:
            ax.plot(ds.idle_time, fitted * 1e3, "r--")
        ax.set_ylabel("Trans. amp. I [mV]")
    else:
        raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")

    ax.set_xlabel("Idle time [ns]")
    ax.set_title(qubit["qubit"])
    if fit is not None:
        _add_fit_text(ax, fit)


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
