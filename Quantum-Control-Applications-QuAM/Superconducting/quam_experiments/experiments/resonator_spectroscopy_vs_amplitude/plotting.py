from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from quam_libs.plot_utils import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the raw data with fitted curves for the given qubits.

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
        plot_individual_raw_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Resonator spectroscopy vs power")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_raw_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
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
    # Plot the data using the secondary y-axis
    ds.loc[qubit].IQ_abs_norm.plot(
        ax=ax,
        add_colorbar=False,
        x="freq_full",
        y="power_dbm",
        robust=True,
    )
    ax.set_ylabel("Power (dBm)")
    ax2 = ax.twiny()
    ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[qubit].IQ_abs_norm.plot(
        ax=ax2, add_colorbar=False, x="detuning_MHz", y="power_dbm", robust=True
    )
    ax2.set_xlabel("Detuning [MHz]")
    # Plot the resonance frequency for each amplitude
    ax.plot(
        ds.rr_min_response.loc[qubit],
        ds.power_dbm,
        color="orange",
        linewidth=0.5,
    )
    # Plot where the optimum readout power was found
    if fit.success:
        ax.axhline(
            y=fit.optimal_power.loc[qubit],
            color="g",
            linestyle="-",
        )
        ax.axvline(
            x=fit.res_freq.loc[qubit],
            color="blue",
            linestyle="--",
        )
    ax.set_title(qubit["qubit"])
