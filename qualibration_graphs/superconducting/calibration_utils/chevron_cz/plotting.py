from typing import List

import matplotlib.pyplot as plt
import numpy as np
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

        # Try to get fit data for this qubit pair, handle if missing
        try:
            fit_data = fits.sel(qubit_pair=qp.id) if fits is not None else None
        except (KeyError, ValueError):
            # If this qubit pair is not in the fit results, set fit_data to None
            fit_data = None

        plot_individual_data_with(ax, ds, qp.id, fit_data)

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

    if hasattr(ds, "state_target"):
        # If the dataset has 'state_target', use it for plotting
        data = ds.state_target
    else:
        data = ds.I_target

    data.sel(qubit_pair=qubit_pair).plot(y="amp_full", ax=ax)

    # Only plot fit results if they exist and are valid
    if fit is not None:
        try:
            # Check if fit values exist and are valid (not NaN)
            cz_len = float(fit.cz_len)
            cz_amp = float(fit.cz_amp)
            if not (np.isnan(cz_len) or np.isnan(cz_amp)):
                ax.scatter(cz_len, cz_amp, color="red", label="Fitted", marker="*", s=100)
                ax.set_title(f"{qubit_pair} - Fit Successful")
                ax.legend()
            else:
                ax.set_title(f"{qubit_pair} - Fit Failed (Invalid Parameters)")
        except (ValueError, TypeError, AttributeError):
            ax.set_title(f"{qubit_pair} - Fit Failed")
    else:
        ax.set_title(f"{qubit_pair} - No Fit Data")

    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Target State")
