
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


def plot_cascade_analysis(ds: xr.Dataset, qubits: List[AnyTransmon], fit_results: dict = None):
    """
    Plot raw spectroscopy data for each qubit, frequency shift plots, flux response plots, and fitted curves if successful.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the spectroscopy data with 'state', 'center_freqs', and 'flux_response' variables.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fit_results : dict, optional
        Dictionary containing fit results for each qubit (default is None).

    Returns
    -------
    dict
        Dictionary containing the generated figure objects.
    """
    figures = {}
    
    # Check what data is available to determine which figures to create
    has_raw_data = ('state' in ds.data_vars and 'freq_full' in ds.coords) or 'IQ_abs' in ds.data_vars or 'I' in ds.data_vars
    has_center_freqs = 'center_freqs' in ds.data_vars
    has_flux_response = 'flux_response' in ds.data_vars
    
    # Only create figures if we have meaningful data to show
    if has_raw_data:
        # Create grid for raw spectroscopy data plots
        grid = QubitGrid(ds, [q.grid_location for q in qubits])

        # Plot raw spectroscopy data for each qubit
        for ax, qubit in grid_iter(grid):
            # Check if we have spectroscopy data with 'state' variable
            if 'state' in ds.data_vars and 'freq_full' in ds.coords:
                # Plot state vs frequency and time
                im = ds.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit].state.plot(
                    ax=ax, add_colorbar=False, x="time", y="freq_GHz"
                )
                ax.set_ylabel("Freq (GHz)")
                ax.set_xlabel("Time (ns)")
                ax.set_title(qubit["qubit"])
                # Add colorbar showing qubit state
                cbar = grid.fig.colorbar(im, ax=ax)
                cbar.set_label("Qubit State")
                # Don't add legend for QuadMesh plots as it causes warnings
            elif 'IQ_abs' in ds.data_vars:
                # For I/Q data, plot amplitude vs time and detuning
                im = ds.loc[qubit].IQ_abs.plot(
                    ax=ax, add_colorbar=False, x="time", y="detuning"
                )
                ax.set_ylabel("Detuning (Hz)")
                ax.set_xlabel("Time (ns)")
                ax.set_title(qubit["qubit"])
                # Add colorbar showing amplitude
                cbar = grid.fig.colorbar(im, ax=ax)
                cbar.set_label("Amplitude (V)")
            elif 'I' in ds.data_vars:
                # For I/Q data, plot I quadrature vs time and detuning
                im = ds.loc[qubit].I.plot(
                    ax=ax, add_colorbar=False, x="time", y="detuning"
                )
                ax.set_ylabel("Detuning (Hz)")
                ax.set_xlabel("Time (ns)")
                ax.set_title(qubit["qubit"])
                # Add colorbar showing I quadrature
                cbar = grid.fig.colorbar(im, ax=ax)
                cbar.set_label("I Quadrature (V)")
        
        grid.fig.suptitle("Qubit spectroscopy vs time after flux pulse")
        grid.fig.set_size_inches(15, 9)
        grid.fig.tight_layout()
        figures["figure_raw"] = grid.fig

    if has_center_freqs:
        # Create grid for frequency shift plots
        grid = QubitGrid(ds, [q.grid_location for q in qubits])

        # Plot frequency shifts over time for each qubit
        for ax, qubit in grid_iter(grid):
            # Plot center frequency vs time
            (ds.loc[qubit].center_freqs / 1e9).plot(ax=ax)
            ax.set_ylabel("Freq (GHz)")
            ax.set_xlabel("Time (ns)")
            ax.set_title(qubit["qubit"])
        
        grid.fig.suptitle("Qubit frequency shift vs time after flux pulse")
        grid.fig.set_size_inches(15, 9)
        grid.fig.tight_layout()
        figures["figure_freqs_shift"] = grid.fig

        # Create grid for frequency shift plots (log scale)
        grid = QubitGrid(ds, [q.grid_location for q in qubits])

        for ax, qubit in grid_iter(grid):
            (ds.loc[qubit].center_freqs / 1e9).plot(ax=ax)
            ax.set_ylabel("Freq (GHz)")
            ax.set_xlabel("Time (ns)")
            ax.set_title(qubit["qubit"])
            ax.set_xscale('log')
            ax.grid(True)
        
        grid.fig.suptitle("Qubit frequency shift vs time after flux pulse (log scale)")
        grid.fig.set_size_inches(15, 9)
        grid.fig.tight_layout()
        figures["figure_freqs_shift_log"] = grid.fig

    if has_flux_response:
        # Create grid for flux response plots
        grid = QubitGrid(ds, [q.grid_location for q in qubits])

        # Plot flux response and fitted curves for each qubit
        for ax, qubit in grid_iter(grid):
            # Plot measured flux response
            ds.loc[qubit].flux_response.plot(ax=ax)
            
            # Plot fitted curves and parameters if fits were successful    
            if fit_results and qubit["qubit"] in fit_results and fit_results[qubit["qubit"]].get("fit_successful", False):
                best_a_dc = fit_results[qubit["qubit"]]["best_a_dc"]
                t_data = ds.loc[qubit].time.values
                t_offset = t_data - t_data[0]
                y_fit = np.ones_like(t_data, dtype=float) * best_a_dc  # Start with fitted constant
                fit_text = f'a_dc = {best_a_dc:.3f}\n'
                for i, (amp, tau) in enumerate(fit_results[qubit["qubit"]]["best_components"]):
                    y_fit += amp * np.exp(-t_offset/tau)
                    fit_text += f'a{i+1} = {amp / best_a_dc:.3f}, τ{i+1} = {tau:.0f}ns\n'

                ax.plot(t_data, y_fit, color='r', label='Full Fit', linewidth=2) # Plot full fit
                ax.text(0.02, 0.98, fit_text, transform=ax.transAxes, 
                        verticalalignment='top', fontsize=8)

            ax.set_ylabel("Flux Response")
            ax.set_xlabel("Time (ns)")
            ax.set_title(qubit["qubit"])
            ax.grid(True)
        
        grid.fig.suptitle("Flux response vs time")
        grid.fig.set_size_inches(15, 9)
        grid.fig.tight_layout()
        figures["figure_flux_response"] = grid.fig

        # Create grid for flux response plots (log scale)
        grid = QubitGrid(ds, [q.grid_location for q in qubits])

        # Plot flux response and fitted curves for each qubit
        for ax, qubit in grid_iter(grid):
            # Plot measured flux response
            ds.loc[qubit].flux_response.plot(ax=ax)

            # Plot fitted curves and parameters if fits were successful    
            if fit_results and qubit["qubit"] in fit_results and fit_results[qubit["qubit"]].get("fit_successful", False):
                best_a_dc = fit_results[qubit["qubit"]]["best_a_dc"]
                t_data = ds.loc[qubit].time.values
                t_offset = t_data - t_data[0]
                y_fit = np.ones_like(t_data, dtype=float) * best_a_dc  # Start with fitted constant
                fit_text = f'a_dc = {best_a_dc:.3f}\n'
                for i, (amp, tau) in enumerate(fit_results[qubit["qubit"]]["best_components"]):
                    y_fit += amp * np.exp(-t_offset/tau)
                    fit_text += f'a{i+1} = {amp / best_a_dc:.3f}, τ{i+1} = {tau:.0f}ns\n'

                ax.plot(t_data, y_fit, color='r', label='Full Fit', linewidth=2) # Plot full fit
                ax.text(0.02, 0.98, fit_text, transform=ax.transAxes, 
                        verticalalignment='top', fontsize=8)

            ax.set_ylabel("Normalized Flux Response")
            ax.set_xlabel("Time (ns)")
            ax.set_title(qubit["qubit"])
            ax.grid(True)
            ax.set_xscale('log')
        
        grid.fig.suptitle("Flux response vs time (log scale)")
        grid.fig.set_size_inches(15, 9)
        grid.fig.tight_layout()
        figures["figure_flux_response_log"] = grid.fig

    return figures