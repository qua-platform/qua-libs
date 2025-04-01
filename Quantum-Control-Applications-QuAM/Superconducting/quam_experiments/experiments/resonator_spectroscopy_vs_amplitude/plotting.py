from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plot_utils import QubitGrid, grid_iter
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
    ds.assign_coords(freq_GHz=ds.full_freq / 1e9).loc[qubit].IQ_abs.plot(
        ax=ax,
        add_colorbar=False,
        x="freq_GHz",
        y="power",
        linewidth=0.5,
    )
    ax.set_ylabel("Power (dBm)")
    ax2 = ax.twiny()
    ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[qubit].IQ_abs_norm.plot(
        ax=ax2, add_colorbar=False, x="detuning_MHz", y="power", robust=True
    )
    ax2.set_xlabel("Detuning [MHz]")
    # Plot the resonance frequency for each amplitude
    ax2.plot(
        (fit.rr_min_response) * 1e-6,
        fit.power,
        color="orange",
        linewidth=0.5,
    )
    # Plot where the optimum readout power was found
    if fit.success:
        ax2.axhline(
            y=fit.optimal_power,
            color="g",
            linestyle="-",
        )
        ax2.axvline(
            x=fit.freq_shift * 1e-6,
            color="blue",
            linestyle="--",
        )
    # ax.set_title(qubit["qubit"])
