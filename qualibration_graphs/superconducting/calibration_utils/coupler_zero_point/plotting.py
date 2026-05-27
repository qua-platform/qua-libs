"""Plotting functions for coupler zero-point calibration.

This module provides visualization functions for displaying raw data and
fit results from coupler zero-point calibration experiments.
"""

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
    - The function creates a grid of subplots: one row per qubit pair, two columns
      (target qubit state, control qubit state).
    - Each subplot contains the raw data and the fitted curve.
    """
    n_pairs = len(qubit_pairs)
    fig, axs = plt.subplots(nrows=n_pairs, ncols=2, figsize=(15, 9))
    if n_pairs == 1:
        axs = axs.reshape(1, -1)

    for ii, qp in enumerate(qubit_pairs):
        # Try to get fit data for this qubit pair, handle if missing
        try:
            fit_data = fits[qp.id] if fits is not None else None
        except (KeyError, ValueError):
            fit_data = None

        plot_individual_data_with(axs[ii, 0], ds, qp.id, fit_data, data_var="target")
        plot_individual_data_with(axs[ii, 1], ds, qp.id, fit_data, data_var="control")

    fig.suptitle("coupler zero point")
    fig.set_size_inches(15, 9)
    fig.tight_layout()

    return fig


def plot_individual_data_with(
    ax: Axes, ds: xr.Dataset, qubit_pair: str, fit: xr.Dataset = None, data_var: str = "target"
):
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
    data_var : str, optional
        Which qubit state to plot: "target" or "control" (default is "target").

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """
    if data_var == "control":
        if hasattr(ds, "state_control"):
            data = ds.state_control.sel(qubit_pair=qubit_pair)
        else:
            data = ds.I_control.sel(qubit_pair=qubit_pair)
    else:
        if hasattr(ds, "state_target"):
            data = ds.state_target.sel(qubit_pair=qubit_pair)
        else:
            data = ds.I_target.sel(qubit_pair=qubit_pair)

    data.assign_coords(
        {"qubit_flux_mV": 1e3 * data.qubit_flux_full, "coupler_flux_amp_mV": 1e3 * data.coupler_flux}
    ).plot(x="qubit_flux_mV", y="coupler_flux_amp_mV", ax=ax)

    decouple_offset = float(data.coupler_flux_full.values[0] - data.coupler_flux.values[0])

    # Only plot fit results if they exist and are valid
    if fit is not None:
        try:
            # Check if fit values exist and are valid (not NaN)
            optimal_qubit_flux = float(fit["optimal_qubit_flux"])
            optimal_coupler_flux = float(fit["optimal_coupler_flux"])
            if not (np.isnan(optimal_qubit_flux) or np.isnan(optimal_coupler_flux)):
                optimal_coupler_amp = optimal_coupler_flux - decouple_offset
                ax.axhline(1e3 * optimal_coupler_amp, color="red", lw=0.5, ls="--", label="Optimal Coupler Flux")
                ax.axvline(1e3 * optimal_qubit_flux, color="red", lw=0.5, ls="--", label="Optimal Qubit Flux")
                ax.set_title(f"{qubit_pair} ({data_var}) - Fit Successful")
                ax.legend()
            else:
                ax.set_title(f"{qubit_pair} ({data_var}) - Fit Failed (Invalid Parameters)")
        except (ValueError, TypeError, AttributeError):
            ax.set_title(f"{qubit_pair} ({data_var}) - Fit Failed")
    else:
        ax.set_title(f"{qubit_pair} ({data_var}) - No Fit Data")
    detuning_data = ds.sel(qubit_pair=qubit_pair).detuning.values * 1e-6

    def flux_to_detuning(x):
        return np.interp(x, 1e3 * data.qubit_flux_full, detuning_data)

    def detuning_to_flux(y):
        return np.interp(y, detuning_data, 1e3 * data.qubit_flux_full)

    sec_ax = ax.secondary_xaxis("top", functions=(flux_to_detuning, detuning_to_flux))
    sec_ax.set_xlabel("Detuning [MHz]")

    ax.set_xlabel("Qubit Flux [mV]")
    ax.set_ylabel("Coupler Flux Pulse Amplitude [mV]")
