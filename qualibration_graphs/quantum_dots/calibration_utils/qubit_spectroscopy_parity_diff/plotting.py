from typing import List
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from qualang_tools.units import unit

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits: List,
    fits: xr.Dataset,
    analysis_signal: str = "E_p2_given_p1_0",
):
    """
    Plots the qubit spectroscopy conditional signal with fitted curves for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the analysis signal (as ``pdiff``) and coordinates.
    qubits : list
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters and ``fit_curve``.
    analysis_signal : str
        Name of the conditional expectation used for ``pdiff`` (for axis labels).

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
        plot_individual_data_with_fit(
            ax, ds, qname, fit, analysis_signal=analysis_signal
        )

    fig.suptitle(f"Qubit spectroscopy ({analysis_signal} + fit)")
    fig.tight_layout()
    return fig


def plot_individual_data_with_fit(
    ax: Axes,
    ds: xr.Dataset,
    qubit_name: str,
    fit: xr.Dataset = None,
    analysis_signal: str = "E_p2_given_p1_0",
):
    """
    Plots individual qubit data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the analysis signal as ``pdiff``.
    qubit_name : str
        The qubit name to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).
    analysis_signal : str
        Label for the vertical axis (which conditional expectation ``pdiff`` encodes).
    """
    (fit.assign_coords(full_freq_GHz=fit.full_freq / u.GHz).pdiff).plot(
        ax=ax, x="full_freq_GHz"
    )
    ax.set_xlabel("RF frequency [GHz]")
    ax.set_ylabel(analysis_signal)
    ax.set_title(f"qubit={qubit_name}", pad=30)

    ax2 = ax.twiny()
    (fit.assign_coords(detuning_MHz=fit.detuning / u.MHz).pdiff).plot(
        ax=ax2, x="detuning_MHz", label=""
    )
    ax2.set_xlabel("Detuning [MHz]")

    if fit is not None and "fit_curve" in fit.data_vars:
        ax2.plot(
            fit.detuning / u.MHz,
            fit.fit_curve.values,
            "r--",
            label="fit",
        )
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
