from typing import List
import xarray as xr
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from calibration_utils.two_qubit_randomized_benchmarking.plot_tools import (
    QubitPairGrid,
    grid_pair_names,
)
from qualibration_libs.analysis import decay_exp
from quam_builder.architecture.superconducting.qubit_pair import AnyTransmonPair


u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(
    ds: xr.Dataset, qubit_pairs: List[AnyTransmonPair], fits: xr.Dataset
):
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
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    # grid = QubitPairGrid(ds, [q.grid_location for q in qubit_pairs])
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qp in grid_iter(grid):
        plot_individual_data_with_fit(ax, ds, qp, fits.sel(qubit_pair=qp["qubit"]))

    grid.fig.suptitle("Two qubit randomized benchmarking")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(
    ax: Axes, ds: xr.Dataset, qubit_pair: dict[str, str], fit: xr.Dataset = None
):
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
    # Fitted decay
    fitted = decay_exp(
        fit.depths,
        fit.fit_data.sel(fit_vals="a"),
        fit.fit_data.sel(fit_vals="offset"),
        fit.fit_data.sel(fit_vals="decay"),
    )
    if hasattr(fit, "state"):
        data = fit.state
        label = "qubit state P(|00>)"
    elif hasattr(fit, "I"):
        data = fit.I
        label = "I quadrature [mV]"
    else:
        raise RuntimeError(
            "The dataset must contain either 'I' or 'state' for the plotting function to work."
        )
    data_std = data.std(dim="nb_of_sequences") / np.sqrt(ds.nb_of_sequences.size)
    ax.errorbar(
        fit.depths,
        fit.averaged_data,
        yerr=data_std,
        fmt=".",
        capsize=2,
        elinewidth=0.5,
    )
    ax.grid("all")
    ax.set_title(qubit_pair["qubit"], pad=22)
    ax.set_xlabel("Circuit depth, D")
    ax.set_ylabel(label)
    ax.plot(fit.depths, fitted, "r--", label="fit")
    ax.text(
        0.15,
        0.9,
        f"2Q RB fidelity (Clifford) = {100 * (1 - float(fit.error_per_clifford.values)):.3f}%\nDecay rate: {fit.fit_data.sel(fit_vals='decay'):.3f}",
        transform=ax.transAxes,
    )
