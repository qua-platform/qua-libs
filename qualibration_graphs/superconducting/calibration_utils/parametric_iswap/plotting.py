from typing import List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from qualang_tools.units import unit
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
            fit_data = fits[qp.id] if fits is not None else None
        except (KeyError, ValueError):
            # If this qubit pair is not in the fit results, set fit_data to None
            fit_data = None

        plot_individual_data_with(ax, ds, qp.id, fit_data)

    fig.suptitle("coupler zero point")
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
    qubit_pair : str
        The ID of the qubit pair to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """

    if hasattr(ds, "state_target"):
        # If the dataset has 'state_target', use it for plotting
        data = ds.state_target.sel(qubit_pair=qubit_pair)
    else:
        data = ds.I_target.sel(qubit_pair=qubit_pair)

    data.assign_coords(
        {"qubit_flux_mV": 1e3 * data.qubit_flux_full, "coupler_flux_mV": 1e3 * data.coupler_flux_full}
    ).plot(x="qubit_flux_mV", y="coupler_flux_mV", ax=ax)

    # Only plot fit results if they exist and are valid
    if fit is not None:
        try:
            # Check if fit values exist and are valid (not NaN)
            optimal_qubit_flux = float(fit["optimal_qubit_flux"])
            optimal_coupler_flux = float(fit["optimal_coupler_flux"])
            if not (np.isnan(optimal_qubit_flux) or np.isnan(optimal_coupler_flux)):
                ax.axhline(1e3 * optimal_coupler_flux, color="red", lw=0.5, ls="--", label="Optimal Coupler Flux")
                ax.axvline(1e3 * optimal_qubit_flux, color="red", lw=0.5, ls="--", label="Optimal Qubit Flux")
                ax.set_title(f"{qubit_pair} - Fit Successful")
                ax.legend()
            else:
                ax.set_title(f"{qubit_pair} - Fit Failed (Invalid Parameters)")
        except (ValueError, TypeError, AttributeError):
            ax.set_title(f"{qubit_pair} - Fit Failed")
    else:
        ax.set_title(f"{qubit_pair} - No Fit Data")
    detuning_data = ds.sel(qubit_pair=qubit_pair).detuning.values * 1e-6

    def flux_to_detuning(x):
        return np.interp(x, 1e3 * data.qubit_flux_full, detuning_data)

    def detuning_to_flux(y):
        return np.interp(y, detuning_data, 1e3 * data.qubit_flux_full)

    sec_ax = ax.secondary_xaxis("top", functions=(flux_to_detuning, detuning_to_flux))
    sec_ax.set_xlabel("Detuning [MHz]")

    ax.set_xlabel("Qubit Flux [mV]")
    ax.set_ylabel("Coupler Flux [mV]")
