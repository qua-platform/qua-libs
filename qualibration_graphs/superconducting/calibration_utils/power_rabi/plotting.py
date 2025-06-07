from typing import List

import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualang_tools.units import unit
from qualibration_libs.analysis import oscillation
from qualibration_libs.plotting import QubitGrid, grid_iter
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
        if len(ds.nb_of_pulses) == 1:
            plot_individual_data_with_fit_1D(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))
        else:
            plot_individual_data_with_fit_2D(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Power Rabi")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit_1D(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
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

    if len(ds.nb_of_pulses.data) == 1:
        fitted_data = None
        plot_fit = False
        if fit is not None and hasattr(fit, "outcome") and getattr(fit.outcome, "values", None) == "successful":
            fitted_data = oscillation(
                fit.amp_prefactor.data,
                fit.fit.sel(fit_vals="a").data,
                fit.fit.sel(fit_vals="f").data,
                fit.fit.sel(fit_vals="phi").data,
                fit.fit.sel(fit_vals="offset").data,
            )
            plot_fit = True

        if hasattr(ds, "I"):
            data = "I"
            label = "Rotated I quadrature [mV]"
        elif hasattr(ds, "state"):
            data = "state"
            label = "Qubit state"
        else:
            raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")

        (ds.assign_coords(amp_mV=ds.full_amp * 1e3).loc[qubit] * 1e3)[data].plot(ax=ax, x="amp_mV", alpha=0.5)
        if plot_fit:
            ax.plot(fit.full_amp * 1e3, 1e3 * fitted_data, linewidth=2, color="red")
        ax.set_ylabel(label)
        ax.set_xlabel("Pulse amplitude [mV]")
        ax2 = ax.twiny()
        (ds.assign_coords(amp_mV=ds.amp_prefactor).loc[qubit] * 1e3)[data].plot(ax=ax2, x="amp_mV", alpha=0.5)
        ax2.set_xlabel("amplitude prefactor")


def plot_individual_data_with_fit_2D(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
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

    if hasattr(ds, "I"):
        data = "I"
    elif hasattr(ds, "state"):
        data = "state"
    else:
        raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")
    (ds.assign_coords(amp_mV=ds.full_amp * 1e3).loc[qubit])[data].plot(
        ax=ax, add_colorbar=False, x="amp_mV", y="nb_of_pulses", robust=True
    )
    ax.set_ylabel(f"Number of pulses")
    ax.set_xlabel("Pulse amplitude [mV]")
    ax2 = ax.twiny()
    (ds.assign_coords(amp_mV=ds.amp_prefactor).loc[qubit])[data].plot(
        ax=ax2, add_colorbar=False, x="amp_mV", y="nb_of_pulses", robust=True
    )
    ax2.set_xlabel("amplitude prefactor")
    if fit.outcome.values == "successful":
        ax2.axvline(
            x=fit.opt_amp_prefactor,
            color="red",
            linestyle="-",
            linewidth=2,
        )
