from typing import List

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_iq_blobs(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the IQ blobs with the derived thresholds for the given qubits.

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
        plot_individual_iq_blobs(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))
    handles, labels = ax.get_legend_handles_labels()
    leg = grid.fig.legend(handles, labels, loc="lower center", ncol=2)
    leg.legend_handles[0].set_markersize(6)
    leg.legend_handles[1].set_markersize(6)
    grid.fig.suptitle("g.s. and e.s. discriminators (rotated)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_iq_blobs(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
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

    ax.plot(1e3 * fit.Ig, 1e3 * fit.Qg, ".", alpha=0.2, label="Ground", markersize=10)
    ax.plot(
        1e3 * fit.Ie,
        1e3 * fit.Qe,
        ".",
        alpha=0.2,
        label="Excited",
        markersize=10,
    )
    ax.plot(
        1e3 * fit.If,
        1e3 * fit.Qf,
        ".",
        alpha=0.2,
        label="Second Excited",
        markersize=10,
    )

    ax.plot(
        1e3 * fit.I_g_center,
        1e3 * fit.Q_g_center,
        "o",
        c="C0",
        ms=3,
        mec="k",
        label="G",
    )
    ax.plot(
        1e3 * fit.I_e_center,
        1e3 * fit.Q_e_center,
        "o",
        c="C1",
        ms=3,
        mec="k",
        label="E",
    )
    ax.plot(
        1e3 * fit.I_f_center,
        1e3 * fit.Q_f_center,
        "o",
        c="C2",
        ms=3,
        mec="k",
        label="F",
    )

    ax.axis("equal")
    ax.set_xlabel("I [mV]")
    ax.set_ylabel("Q [mV]")
    ax.set_title(qubit["qubit"])


def plot_confusion_matrices(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the confusion matrix for the given qubits.

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
        plot_individual_confusion_matrix(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("g.s., e.s. and f.s. fidelity")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_confusion_matrix(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
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

    # Derive the confusion matrix
    confusion = np.zeros((3, 3))
    for p, prep_state in enumerate(["g", "e", "f"]):
        dist_g = np.sqrt((fit.I_g_center - fit[f"I{prep_state}"]) ** 2 + (fit.Q_g_center - fit[f"Q{prep_state}"]) ** 2)
        dist_e = np.sqrt((fit.I_e_center - fit[f"I{prep_state}"]) ** 2 + (fit.Q_e_center - fit[f"Q{prep_state}"]) ** 2)
        dist_f = np.sqrt((fit.I_f_center - fit[f"I{prep_state}"]) ** 2 + (fit.Q_f_center - fit[f"Q{prep_state}"]) ** 2)
        dist = np.stack([dist_g, dist_e, dist_f], axis=0)
        counts = np.argmin(dist, axis=0)
        confusion[p][0] = np.sum(counts == 0) / len(counts)
        confusion[p][1] = np.sum(counts == 1) / len(counts)
        confusion[p][2] = np.sum(counts == 2) / len(counts)

    ax.imshow(confusion)
    ax.set_xticks([0, 1, 2], labels=["|g>", "|e>", "|f>"])
    ax.set_yticks([0, 1, 2], labels=["|g>", "|e>", "|f>"])
    ax.set_ylabel("Prepared")
    ax.set_xlabel("Measured")
    for prep in range(3):
        for meas in range(3):
            color = "k" if prep == meas else "w"
            ax.text(
                meas,
                prep,
                f"{100 * confusion[prep, meas]:.1f}%",
                ha="center",
                va="center",
                color=color,
            )
    ax.set_title(qubit["qubit"])
