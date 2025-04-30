from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_distances_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the distance between the resonator responses when the qubits are in |g> and |e>.

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
        plot_individual_distance_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Readout frequency optimization (distance)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_IQ_abs_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the resonator responses when the qubits are in |g> and |e>.

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
        plot_individual_IQ_abs_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Readout frequency optimization (IQ_abs)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_distance_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots the distance between the resonator responses when the qubit is in |g> and |e> on a given axis with optional fit.

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
    (1e3 * ds.assign_coords(full_freq_GHz=ds.full_freq / 1e9).D.loc[qubit]).plot(ax=ax, x="full_freq_GHz", label=None)
    ax.axvline(
        fit.optimal_frequency / 1e9,
        color="red",
        linestyle="--",
        label="applied detuning",
    )
    ax.set_xlabel("RF frequency [GHz]")
    ax.set_ylabel("Distance between |g> and |e> [mV]")
    ax.legend(loc="upper left")
    ax2 = ax.twiny()
    (1e3 * ds.assign_coords(freq_MHz=ds.detuning / 1e6).D.loc[qubit]).plot(ax=ax2, x="freq_MHz", label=None)
    ax2.set_xlabel("Detuning [MHz]")


def plot_individual_IQ_abs_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots individual resonator responses when the qubit is in |g> and |e> on a given axis with optional fit.

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

    (1e3 * ds.assign_coords(full_freq_GHz=ds.full_freq / 1e9).IQ_abs_g.loc[qubit]).plot(
        ax=ax, x="full_freq_GHz", label="g.s"
    )
    (1e3 * ds.assign_coords(full_freq_GHz=ds.full_freq / 1e9).IQ_abs_e.loc[qubit]).plot(
        ax=ax, x="full_freq_GHz", label="e.s"
    )
    ax.axvline(
        fit.optimal_frequency / 1e9,
        color="red",
        linestyle="--",
        label="applied detuning",
    )
    ax.set_xlabel("RF frequency [GHz]")
    ax.set_ylabel("Resonator response [mV]")
    ax.legend(loc="lower left")
    ax2 = ax.twiny()
    (1e3 * ds.assign_coords(freq_MHz=ds.detuning / 1e6).IQ_abs_g.loc[qubit]).plot(ax=ax2, x="freq_MHz", label="g.s")
    (1e3 * ds.assign_coords(freq_MHz=ds.detuning / 1e6).IQ_abs_e.loc[qubit]).plot(ax=ax2, x="freq_MHz", label="e.s")
    ax2.set_xlabel("Detuning [MHz]")
