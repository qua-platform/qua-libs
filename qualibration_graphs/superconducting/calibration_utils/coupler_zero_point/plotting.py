from typing import List
import xarray as xr
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

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

"""
# %% {Plotting}
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qp in grid_iter(grid):
        if node.parameters.use_state_discrimination:
            values_to_plot = ds.state_control.sel(qubit=qp['qubit'])
        else:
            values_to_plot = ds.I_control.sel(qubit=qp['qubit'])

        values_to_plot.assign_coords({"flux_qubit_mV": 1e3*values_to_plot.flux_qubit_full, "flux_coupler_mV": 1e3*values_to_plot.flux_coupler_full}).plot(ax = ax, cmap = 'viridis', x = 'flux_qubit_mV', y = 'flux_coupler_mV')
        qubit_pair = machine.qubit_pairs[qp['qubit']]
        ax.set_title(f"{qp['qubit']}, idle: {qubit_pair.coupler.decouple_offset}V, g=0: {node.results['results'][qp['qubit']]['flux_coupler_min']:.4f}V", fontsize = 10)
        ax.axhline(1e3*node.results["results"][qp["qubit"]]["flux_coupler_min"], color = 'red', lw = 0.5, ls = '--')
        ax.axhline(1e3*machine.qubit_pairs[qp['qubit']].coupler.decouple_offset, color = 'blue', lw =0.5, ls = '--')
        ax.axvline(1e3*node.results["results"][qp["qubit"]]["flux_qubit_max"], color = 'red', lw =0.5, ls = '--')
        # Create a secondary x-axis for detuning
        flux_qubit_data = ds.sel(qubit=qp['qubit']).flux_qubit_full.values*1e3
        detuning_data = ds.sel(qubit=qp['qubit']).detuning.values * 1e-6

        def flux_to_detuning(x):
            return np.interp(x, flux_qubit_data, detuning_data)

        def detuning_to_flux(y):
            return np.interp(y, detuning_data, flux_qubit_data)

        sec_ax = ax.secondary_xaxis('top', functions=(flux_to_detuning, detuning_to_flux))
        sec_ax.set_xlabel('Detuning [MHz]')
        ax.set_xlabel('Qubit flux pulse [mV]')
        ax.set_ylabel('Coupler flux pulse [mV]')
    grid.fig.suptitle('Control')
    plt.tight_layout()
    plt.show()
    node.results['figure_control'] = grid.fig

    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qp in grid_iter(grid):
        if node.parameters.use_state_discrimination:
            values_to_plot = ds.state_target.sel(qubit=qp['qubit'])
        else:
            values_to_plot = ds.I_target.sel(qubit=qp['qubit'])

        values_to_plot.assign_coords({"flux_qubit_mV": 1e3*values_to_plot.flux_qubit_full, "flux_coupler_mV": 1e3*values_to_plot.flux_coupler_full}).plot(ax = ax, cmap = 'viridis', x = 'flux_qubit_mV', y = 'flux_coupler_mV')
        qubit_pair = machine.qubit_pairs[qp['qubit']]
        ax.set_title(f"{qp['qubit']}, idle: {qubit_pair.coupler.decouple_offset}V, g=0: {node.results['results'][qp['qubit']]['flux_coupler_min']:.4f}V", fontsize = 10)
        ax.axhline(1e3*node.results["results"][qp["qubit"]]["flux_coupler_min"], color = 'red', lw = 0.5, ls = '--')
        ax.axhline(1e3*machine.qubit_pairs[qp['qubit']].coupler.decouple_offset, color = 'blue', lw =0.5, ls = '--')
        ax.axvline(1e3*node.results["results"][qp["qubit"]]["flux_qubit_max"], color = 'red', lw =0.5, ls = '--')
        # Create a secondary x-axis for detuning
        flux_qubit_data = ds.sel(qubit=qp['qubit']).flux_qubit_full.values*1e3
        detuning_data = ds.sel(qubit=qp['qubit']).detuning.values * 1e-6

        def flux_to_detuning(x):
            return np.interp(x, flux_qubit_data, detuning_data)

        def detuning_to_flux(y):
            return np.interp(y, detuning_data, flux_qubit_data)

        sec_ax = ax.secondary_xaxis('top', functions=(flux_to_detuning, detuning_to_flux))
        sec_ax.set_xlabel('Detuning [MHz]')
        ax.set_xlabel('Qubit flux shift [mV]')
        ax.set_ylabel('Coupler flux [mV]')
    grid.fig.suptitle('Target')
    plt.tight_layout()
    plt.show()
    node.results['figure_target'] = grid.fig
"""