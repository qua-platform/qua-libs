from typing import List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit

u = unit(coerce_to_integer=True)


def plot_ramsey_detuning(ds: xr.Dataset, qubits: List, fit_results: dict = None) -> Figure:
    """
    Plots the parity difference signal as a function of frequency detuning for each qubit.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing pdiff data and optional fit curves.
    qubits : list
        A list of qubit objects to plot.
    fit_results : dict, optional
        Dictionary of fit results for each qubit (default is None).

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    num_qubits = len(qubits)
    fig, axes = plt.subplots(1, num_qubits, figsize=(5 * num_qubits, 4), squeeze=False)
    axes = axes.flatten()

    for ax, qubit in zip(axes, qubits):
        plot_individual_ramsey_detuning(ax, ds, qubit, fit_results)

    fig.suptitle("Ramsey detuning parity difference")
    fig.tight_layout()
    return fig


def plot_individual_ramsey_detuning(ax: Axes, ds: xr.Dataset, qubit, fit_results: dict = None):
    """
    Plots individual qubit parity difference data as a function of frequency detuning.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the parity difference data.
    qubit : object
        The qubit object.
    fit_results : dict, optional
        Dictionary of fit results for each qubit (default is None).
    """
    detuning_mhz = ds.detuning.values / u.MHz

    # Plot pdiff
    ax.plot(
        detuning_mhz,
        ds[f"pdiff_{qubit.name}"].values,
        "o-",
        markersize=3,
        label="P_diff",
    )

    # Plot fit if available
    fit_key = f"pdiff_fit_{qubit.name}"
    if fit_key in ds.data_vars:
        ax.plot(
            detuning_mhz,
            ds[fit_key].values,
            "r--",
            lw=1.5,
            label="Fit",
        )

    # Mark the frequency offset if fit results available
    if fit_results and qubit.name in fit_results:
        fr = fit_results[qubit.name]
        if fr["success"]:
            freq_offset_mhz = fr["freq_offset"] / u.MHz
            ax.axvline(
                freq_offset_mhz,
                color="g",
                linestyle=":",
                lw=1.5,
                label=f"Offset: {freq_offset_mhz:.3f} MHz",
            )

    ax.set_xlabel("Frequency detuning [MHz]")
    ax.set_ylabel("Parity difference")
    ax.set_title(f"Qubit: {qubit.name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
