from typing import List
import xarray as xr
from matplotlib.axes import Axes

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis import decay_exp
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the resonator spectroscopy amplitude IQ_abs with fitted curves for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains the raw data and the fitted curve.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("T2 echo with fit")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots individual qubit data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        mapping to the qubit to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """
    # Fitted decay
    fitted = decay_exp(
        ds.idle_time,
        fit.fit_data.sel(fit_vals="a"),
        fit.fit_data.sel(fit_vals="offset"),
        fit.fit_data.sel(fit_vals="decay"),
    )

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

    ax.set_title(qubit["qubit"])
    ax.set_xlabel("Idle_time (µs)")
    ax.text(
        0.1,
        0.9,
        f'T2e = {fit["T2_echo"].values*1e-3:.1f} ± {fit["T2_echo_error"].values*1e-3:.1f} µs',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )
