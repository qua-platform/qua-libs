from typing import List

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubit_pairs: List[AnyTransmon], fits: xr.Dataset):
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
    fig, axs = plt.subplots(nrows=len(qubit_pairs), ncols=1, figsize=(15, 9))
    for ii, qp in enumerate(qubit_pairs):
        ax = axs[ii] if len(qubit_pairs) > 1 else axs
        plot_individual_data_with(ax, ds, qp.id, fits.sel(qubit_pair=qp.id))

    fig.suptitle("CZ Chevron")
    fig.set_size_inches(15, 9)
    fig.tight_layout()

    return fig


def plot_individual_data_with(ax: Axes, ds: xr.Dataset, qubit_pair: str, fit: xr.Dataset = None):
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

    # if hasattr(ds, "I"):
    #     data = "I"
    # elif hasattr(ds, "state"):
    #     data = "state"
    # else:
    #     raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")

    ds.state_target.sel(qubit_pair=qubit_pair).plot(y="amp_full", ax=ax)
    ax.scatter(fit.cz_len, fit.cz_amp, color="red", label="Fitted", marker="*", s=100)
