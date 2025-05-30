from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis import lorentzian_dip
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_phase(ds: xr.Dataset, qubits: List[AnyTransmon]) -> Figure:
    """
    Plots the raw phase data for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list
        A list of qubits to plot.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains two x-axes: one for the full frequency in GHz and one for the detuning in MHz.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax1, qubit in grid_iter(grid):
        # Create a first x-axis for full_freq_GHz
        ds.assign_coords(full_freq_GHz=ds.full_freq / u.GHz).loc[qubit].phase.plot(ax=ax1, x="full_freq_GHz")
        ax1.set_xlabel("RF frequency [GHz]")
        ax1.set_ylabel("phase [rad]")
        # Create a second x-axis for detuning_MHz
        ax2 = ax1.twiny()
        ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[qubit].phase.plot(ax=ax2, x="detuning_MHz")
        ax2.set_xlabel("Detuning [MHz]")
    grid.fig.suptitle("Resonator spectroscopy (phase)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()

    return grid.fig


def plot_raw_amplitude_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
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
        plot_individual_amplitude_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Resonator spectroscopy (amplitude + fit)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_amplitude_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
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
    if fit:
        fitted_data = lorentzian_dip(
            ds.detuning,
            float(fit.amplitude.values),
            float(fit.position.values),
            float(fit.width.values) / 2,
            float(fit.base_line.mean().values),
        )
    else:
        fitted_data = None

    # Create a first x-axis for full_freq_GHz
    (ds.assign_coords(full_freq_GHz=ds.full_freq / u.GHz).loc[qubit].IQ_abs / u.mV).plot(ax=ax, x="full_freq_GHz")
    ax.set_xlabel("RF frequency [GHz]")
    ax.set_ylabel(r"$R=\sqrt{I^2 + Q^2}$ [mV]")
    # Create a second x-axis for detuning_MHz
    ax2 = ax.twiny()
    (ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[qubit].IQ_abs / u.mV).plot(ax=ax2, x="detuning_MHz")
    ax2.set_xlabel("Detuning [MHz]")
    # Plot the fitted data
    if fitted_data is not None:
        ax2.plot(ds.detuning / u.MHz, fitted_data / u.mV, "r--")
