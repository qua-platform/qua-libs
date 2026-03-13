from typing import List
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from qualang_tools.units import unit
from qualibration_libs.analysis import lorentzian_peak

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List, fits: xr.Dataset):
    """
    Plots the qubit spectroscopy parity-difference with fitted curves for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the parity-difference data.
    qubits : list
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    qubit_names = [q.name for q in qubits]
    n = len(qubit_names)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)

    for i, qname in enumerate(qubit_names):
        ax = axes[0, i]
        fit = fits.sel(qubit=qname)
        plot_individual_data_with_fit(ax, ds, qname, fit)

    fig.suptitle("Qubit spectroscopy parity difference (pdiff + fit)")
    fig.tight_layout()
    return fig


def plot_individual_data_with_fit(ax: Axes, ds: xr.Dataset, qubit_name: str, fit: xr.Dataset = None):
    """
    Plots individual qubit data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the parity-difference data.
    qubit_name : str
        The qubit name to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).
    """
    if fit:
        fitted_data = lorentzian_peak(
            ds.detuning,
            float(fit.amplitude.values),
            float(fit.position.values),
            float(fit.width.values) / 2,
            float(fit.base_line.mean().values),
        )
    else:
        fitted_data = None

    # Create a first x-axis for full_freq_GHz
    (fit.assign_coords(full_freq_GHz=fit.full_freq / u.GHz).pdiff).plot(ax=ax, x="full_freq_GHz")
    ax.set_xlabel("RF frequency [GHz]")
    ax.set_ylabel("Parity difference")
    ax.set_title(qubit_name)
    # Create a second x-axis for detuning_MHz
    ax2 = ax.twiny()
    (fit.assign_coords(detuning_MHz=fit.detuning / u.MHz).pdiff).plot(ax=ax2, x="detuning_MHz", label="")
    ax2.set_xlabel("Detuning [MHz]")
    # Plot the fitted data
    if fitted_data is not None:
        ax2.plot(fit.detuning / u.MHz, fitted_data, "r--")
