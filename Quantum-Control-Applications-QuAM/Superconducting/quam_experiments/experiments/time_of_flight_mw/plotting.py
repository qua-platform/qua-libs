from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from qualibration_libs.plot_utils import QubitGrid, grid_iter
from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plot_utils import QubitGrid, grid_iter
from quam_experiments.analysis.fit import lorentzian_peak
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_single_run_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
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
        plot_individual_single_run_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Single run")
    grid.fig.set_size_inches(15, 9)
    grid.fig.legend(loc="upper right", ncols=4, bbox_to_anchor=(0.5, 1.35))
    grid.fig.tight_layout()
    return grid.fig


def plot_averaged_run_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
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
        plot_individual_averaged_run_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Averaged run")
    grid.fig.set_size_inches(15, 9)
    grid.fig.legend(loc="upper right", ncols=4, bbox_to_anchor=(0.5, 1.35))
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_single_run_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
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
    ds.loc[qubit].adc_single_runI.plot(ax=ax, x="readout_time", label="I", color="b")
    ds.loc[qubit].adc_single_runQ.plot(ax=ax, x="readout_time", label="Q", color="r")
    ax.axvline(fit.delay, color="k", linestyle="--", label="TOF")
    # ax.axhline(ds.loc[qubit].offsets_I, color="b", linestyle="--")
    # ax.axhline(ds.loc[qubit].offsets_Q, color="r", linestyle="--")
    ax.fill_between(
        range(ds.sizes["readout_time"]),
        -0.5,
        0.5,
        color="grey",
        alpha=0.2,
        label="ADC Range",
    )
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Readout amplitude [mV]")
    ax.set_title(qubit["qubit"])


def plot_individual_averaged_run_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
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
    ds.loc[qubit].adcI.plot(ax=ax, x="readout_time", label="I", color="b")
    ds.loc[qubit].adcQ.plot(ax=ax, x="readout_time", label="Q", color="r")
    ax.axvline(fit.delay, color="k", linestyle="--", label="TOF")
    # ax.axhline(ds.loc[qubit].offsets_I, color="b", linestyle="--")
    # ax.axhline(ds.loc[qubit].offsets_Q, color="r", linestyle="--")
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Readout amplitude [mV]")
    ax.set_title(qubit["qubit"])
