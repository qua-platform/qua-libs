from typing import List

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
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

    grid.fig.suptitle("Pi pulse vs flux")
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

    fit.I.plot(
        ax=ax,
        x="time",
        y="detuning",
        label="I quadrature",
    )

    fit.peak_freq.plot(ax=ax, linestyle="--", marker="o", color="black", label="Peak frequency")

    # Plot the exponential fit as a wide red line
    if "exp_fit" in fit:
        try:
            # Get the fit parameters
            amplitude = float(fit.exp_fit.sel(fit_vals="a").values)
            offset = float(fit.exp_fit.sel(fit_vals="offset").values)
            decay_rate = float(fit.exp_fit.sel(fit_vals="decay").values)

            # Determine the dimension that was fitted (should be 'time' for pi flux experiments)
            peak_freq_dims = list(fit.peak_freq.dims)
            fit_dim = [dim for dim in peak_freq_dims if dim != "qubit"][0] if peak_freq_dims else "time"

            # Get the coordinate values for the fitted dimension
            if fit_dim in fit.coords:
                x_vals = fit.coords[fit_dim].values

                # Calculate the exponential fit: amplitude * exp(decay_rate * x) + offset
                y_fit = amplitude * np.exp(decay_rate * x_vals) + offset

                # Plot the exponential fit as a wide red line
                ax.plot(x_vals, y_fit, "r-", linewidth=3, label="Exponential fit", alpha=0.8)

        except Exception as e:
            print(f"Could not plot exponential fit for qubit {qubit}: {e}")

    ax.legend()
