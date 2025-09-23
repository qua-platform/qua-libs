from typing import List
import xarray as xr
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

from qualang_tools.units import unit
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibrate import QualibrationNode

u = unit(coerce_to_integer=True)

def plot_control_data(ds: xr.Dataset, qubit_pairs: List[AnyTransmon], fits: xr.Dataset,node:QualibrationNode):
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

        if node.parameters.use_state_discrimination:
            values_to_plot = ds.state_control.sel(qubit_pair=qp.name)
        else:
            values_to_plot = ds.Q_control.sel(qubit_pair=qp.name)
        values_to_plot.plot(ax = ax, cmap = 'viridis', y = 'idle_time', x = 'flux_coupler')
        qubit_pair = node.machine.qubit_pairs[qp.name]
        ax.set_title(f"{qp.name}, coupler set point: {qubit_pair.coupler.decouple_offset}", fontsize = 10)
    fig.suptitle('Control')
    fig.tight_layout()

    return fig


def plot_target_data(ds: xr.Dataset, qubit_pairs: List[AnyTransmon], fits: xr.Dataset,node:QualibrationNode):
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

        if node.parameters.use_state_discrimination:
            values_to_plot = ds.state_target.sel(qubit_pair=qp.name)
        else:
            values_to_plot = ds.Q_target.sel(qubit_pair=qp.name)
        values_to_plot.plot(ax = ax, cmap = 'viridis', y = 'idle_time', x = 'flux_coupler')
        qubit_pair = node.machine.qubit_pairs[qp.name]
        ax.set_title(f"{qp.name}, coupler set point: {qubit_pair.coupler.decouple_offset}", fontsize = 10)
    fig.suptitle('Target')
    fig.tight_layout()

    return fig

def plot_domain_frequency(ds: xr.Dataset, qubit_pairs: List[AnyTransmon], fits: xr.Dataset,node:QualibrationNode):
    fig, axs = plt.subplots(nrows=len(qubit_pairs), ncols=1, figsize=(15, 9))

    for ii, qp in enumerate(qubit_pairs):
        ax = axs[ii] if len(qubit_pairs) > 1 else axs
        target_gate_time = 50  # ns (fallback)
        if node.parameters.cz_or_iswap  == "cz" and "Cz" in qp.macros:
            target_gate_time = qp.macros["Cz"].coupler_flux_pulse.length
        (1e3*ds.dominant_frequency.sel(qubit_pair=qp.name)).plot(ax = ax, marker = '.', ls = 'None', x = 'flux_coupler')
        qubit_pair = node.machine.qubit_pairs[qp.name]
        ax.axvline(x = qubit_pair.coupler.decouple_offset, color = 'black', label="Decouple offset")
        ax.axvline(x = fits[qp.name]["coupler_flux_pulse"], color = 'red', lw = 0.5, ls = '--', label=f"Optimal frequency for {target_gate_time}ns pulse")
        ax.axvline(x = fits[qp.name]["coupler_flux_min"] - qubit_pair.coupler.decouple_offset, color = 'green', lw = 0.5, ls = '--')
        ax.set_title(f"{qp.name}, coupler set point: {qubit_pair.coupler.decouple_offset}", fontsize = 10)
        ax.set_xlabel('Flux Coupler')
        ax.set_ylabel('Frequency (MHz)')
        ax.legend()
    fig.suptitle('Target')
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
        data = ds.state_target.sel(qubit_pair=qubit_pair)
    else:
        data = ds.I_target.sel(qubit_pair=qubit_pair)

    data.assign_coords({"qubit_flux_mV": 1e3*data.qubit_flux_full, "coupler_flux_mV": 1e3*data.coupler_flux_full}).plot(x='qubit_flux_mV',y='coupler_flux_mV')

    # Only plot fit results if they exist and are valid
    if fit is not None:
        try:
            # Check if fit values exist and are valid (not NaN)
            qubit_flux_max = float(fit['qubit_flux_max'])
            coupler_flux_min = float(fit['coupler_flux_min'])
            if not (np.isnan(qubit_flux_max) or np.isnan(coupler_flux_min)):
                ax.axhline(1e3*coupler_flux_min, color = 'red', lw = 0.5, ls = '--')
                #ax.axhline(1e3*machine.qubit_pairs[qp['qubit']].coupler.decouple_offset, color = 'blue', lw =0.5, ls = '--')
                ax.axvline(1e3*qubit_flux_max, color = 'red', lw =0.5, ls = '--')
            else:
                ax.set_title(f"{qubit_pair} - Fit Failed (Invalid Parameters)")
        except (ValueError, TypeError, AttributeError):
            ax.set_title(f"{qubit_pair} - Fit Failed")
    else:
        ax.set_title(f"{qubit_pair} - No Fit Data")
    detuning_data = ds.sel(qubit_pair=qubit_pair).detuning.values * 1e-6
    def flux_to_detuning(x):
        return np.interp(x, 1e3*data.qubit_flux_full, detuning_data)
    
    def detuning_to_flux(y):
        return np.interp(y, detuning_data, 1e3*data.qubit_flux_full)

    sec_ax = ax.secondary_xaxis('top', functions=(flux_to_detuning, detuning_to_flux))
    sec_ax.set_xlabel('Detuning [MHz]')

    ax.set_xlabel("qubit flux [mV]")
    ax.set_ylabel("coupler flux [mV]")
