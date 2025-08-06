from typing import List, Dict

import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from calibration_utils.cryoscope import expdecay, two_expdecay
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualang_tools.units import unit
from qualibration_libs.analysis import oscillation
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from scipy.signal import lfilter

u = unit(coerce_to_integer=True)


def plot_raw_data(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
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
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_raw_data(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Raw Cryoscope Data")
    grid.fig.set_size_inches(5, 5)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_raw_data(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots individual qubit data on a given axis.

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

    if hasattr(fit, "I"):
        data = "I"
        label = "Rotated I quadrature [mV]"
    elif hasattr(fit, "state"):
        data = "state"
        label = "Qubit state"
    else:
        raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")

    # ds[data].sel(qubit=qubit["qubit"]).plot.line(ax=ax, x="time", linestyle="--", marker=".")

    ax.scatter(fit[data].sel(axis="x"), fit[data].sel(axis="y"))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"qubit = {qubit['qubit']}")


def plot_normalized_flux(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
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
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_flux(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Reconstructed normalized Flux")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_flux(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots individual qubit data on a given axis.

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
    - If the fit dataset is provided and successful, the fitted curve is plotted along with the raw data.
    - If the fit failed, only the raw data is plotted with appropriate annotations.
    """
    # Always plot the raw flux data
    fit.fit_results.flux.plot(ax=ax, linestyle="--", marker=".", label="Raw flux data")

    # Check if fits are available and successful
    fit1_success = fit.fit_results.attrs.get("fit_1exp_success", False)
    fit2_success = fit.fit_results.attrs.get("fit_2exp_success", False)

    if fit1_success:
        fit1_params = fit.fit_results.attrs.get("fit_1exp")
        if fit1_params is not None:
            ax.plot(fit.time, expdecay(fit.time, *fit1_params), label="Single exp fit", linewidth=3, alpha=0.8)

    if fit2_success:
        fit2_params = fit.fit_results.attrs.get("fit_2exp")
        if fit2_params is not None:
            ax.plot(fit.time, two_expdecay(fit.time, *fit2_params), label="Double exp fit", linewidth=3, alpha=0.8)

    # Add status annotation
    status_text = []
    if fit1_success:
        status_text.append("Single exp: ✓")
    else:
        status_text.append("Single exp: ✗")

    if fit2_success:
        status_text.append("Double exp: ✓")
    else:
        status_text.append("Double exp: ✗")

    ax.text(
        0.02,
        0.98,
        "\n".join(status_text),
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
    )

    ax.legend()
    ax.set_title(f"Qubit {qubit['qubit']}")
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Normalized flux")


def plot_raw_data_only(ds: xr.Dataset, qubits: List[AnyTransmon]):
    """
    Plots only the raw cryoscope data when fits are not available.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_raw_data_only(ax, ds, qubit)

    grid.fig.suptitle("Raw Cryoscope Data (No fits available)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_raw_data_only(ax: Axes, ds: xr.Dataset, qubit: dict[str, str]):
    """
    Plots individual raw qubit data when no fits are available.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        mapping to the qubit to plot.
    """
    try:
        if hasattr(ds, "I"):
            data = "I"
            label = "Rotated I quadrature [mV]"
        elif hasattr(ds, "state"):
            data = "state"
            label = "Qubit state"
        else:
            raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")

        # Plot raw data vs time for each frame
        qubit_data = ds[data].sel(qubit=qubit["qubit"])

        # Plot a subset of frames to avoid cluttering
        num_frames = qubit_data.sizes.get("frame", 1)
        frame_step = max(1, num_frames // 5)  # Show max 5 frames

        for i, frame_idx in enumerate(range(0, num_frames, frame_step)):
            frame_data = qubit_data.isel(frame=frame_idx)
            alpha = 0.7 if num_frames > 1 else 1.0
            ax.plot(
                frame_data.time,
                frame_data.values,
                linestyle="--",
                marker=".",
                alpha=alpha,
                label=f"Frame {frame_idx}" if num_frames > 1 and i < 3 else "",
            )

        ax.set_title(f"Qubit {qubit['qubit']} - Fit Failed")
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel(label)

        if num_frames > 1:
            ax.legend()

        ax.text(
            0.02,
            0.98,
            "Fits failed - showing raw data only",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
        )

    except Exception as e:
        # Fallback if even raw data plotting fails
        ax.text(
            0.5,
            0.5,
            f"Error plotting data:\n{str(e)}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
        )
        ax.set_title(f"Qubit {qubit['qubit']} - Error")


def plot_advanced_analysis(ds_fit: xr.Dataset, qubits: List[AnyTransmon], advanced_results: Dict):
    """
    Plots the advanced cryoscope analysis results for all qubits.
    
    Parameters
    ----------
    ds_fit : xr.Dataset
        The fitted dataset containing flux data
    qubits : List[AnyTransmon]
        List of qubits to plot
    advanced_results : Dict
        Dictionary containing advanced fit parameters for each qubit
        
    Returns
    -------
    Dict[str, Figure]
        Dictionary mapping figure names to matplotlib Figure objects
    """
    figures = {}
    
    for qubit in qubits:
        qubit_name = qubit.name
        
        if qubit_name not in advanced_results:
            continue
            
        advanced_result = advanced_results[qubit_name]
        
        # Handle both dictionary and dataclass formats
        if hasattr(advanced_result, 'success'):
            # It's a dataclass object
            if not advanced_result.success:
                continue
        else:
            # It's a dictionary
            if not advanced_result['success']:
                continue
        
        # Get the flux data
        if hasattr(ds_fit, 'fit_results') and hasattr(ds_fit.fit_results, 'flux'):
            flux_data = ds_fit.fit_results.flux.sel(qubit=qubit_name)
        else:
            continue
        
        # Create individual plots for this qubit
        qubit_figures = plot_individual_advanced_analysis(flux_data, qubit_name, advanced_result)
        figures.update(qubit_figures)
    
    return figures


def plot_individual_advanced_analysis(flux_data: xr.DataArray, qubit_name: str, advanced_result):
    """
    Plots individual advanced analysis results for a single qubit.
    
    Parameters
    ----------
    flux_data : xr.DataArray
        The flux data for the qubit
    qubit_name : str
        Name of the qubit
    advanced_result : Dict or AdvancedFitParameters
        Advanced fit parameters for the qubit (can be dict or dataclass)
        
    Returns
    -------
    Dict[str, Figure]
        Dictionary mapping figure names to matplotlib Figure objects
    """
    figures = {}
    
    # Handle both dictionary and dataclass formats
    if hasattr(advanced_result, 'rise_index'):
        # It's a dataclass object
        rise_index = advanced_result.rise_index
        drop_index = advanced_result.drop_index
        fit_parameters = advanced_result.fit_parameters
        iir_coefficients = advanced_result.iir_coefficients
        convolved_fir = advanced_result.convolved_fir
        final_vals = advanced_result.final_vals
    else:
        # It's a dictionary
        rise_index = advanced_result['rise_index']
        drop_index = advanced_result['drop_index']
        fit_parameters = advanced_result['fit_parameters']
        iir_coefficients = advanced_result['iir_coefficients']
        convolved_fir = advanced_result['convolved_fir']
        final_vals = advanced_result['final_vals']
    
    # Plot 1: Rise/Drop indices
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    flux_data.plot(ax=axs[0])
    axs[0].set_title(f'Flux vs time - {qubit_name}')
    axs[0].axvline(rise_index, color='r', lw=0.5, ls='--', label='Rise index')
    axs[0].axvline(drop_index, color='r', lw=0.5, ls='--', label='Drop index')
    axs[0].set_xlabel('')
    axs[0].legend()
    
    # Extract the rising part for zoomed view
    flux_cryoscope_tp = flux_data.sel(time=slice(rise_index, drop_index))
    flux_cryoscope_tp = flux_cryoscope_tp.assign_coords(
        time=flux_cryoscope_tp.time - rise_index + 1)
    flux_cryoscope_tp.plot(ax=axs[1])
    axs[1].set_title('Zoomed in')
    axs[1].set_xlabel('time (ns)')
    plt.tight_layout()
    figures[f'rise_drop_{qubit_name}'] = fig
    
    # Plot 2: Exponential fit
    fig, ax = plt.subplots(figsize=(10, 6))
    flux_cryoscope_tp.plot(ax=ax, marker='.', label='Data')
    ax.plot(flux_cryoscope_tp.time, expdecay(flux_cryoscope_tp.time, **fit_parameters), 
           label='Single exp fit', linewidth=2)
    fit_text = f's={fit_parameters["s"]:.6f}\na={fit_parameters["a"]:.6f}\nt={fit_parameters["t"]:.6f}'
    ax.text(0.02, 0.98, fit_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    ax.axvline(x=1, color='red', linestyle='--', label='Exponential Fit Time Interval')
    ax.axvline(x=30, color='red', linestyle='--')
    ax.set_title(f'Exponential Fit - {qubit_name}')
    ax.legend()
    figures[f'exp_fit_{qubit_name}'] = fig
    
    # Plot 3: Filtered response
    flux_cryoscope_filtered = flux_data.copy()
    flux_cryoscope_filtered.values[0] = 0
    filtered_response_long_1exp = lfilter(iir_coefficients, iir_coefficients, flux_cryoscope_filtered.values)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(flux_data.time, flux_data.values, label='Original data')
    ax.plot(flux_data.time, filtered_response_long_1exp, label='Filtered response (1exp)')
    ax.legend()
    ax.set_xlabel('time (ns)')
    ax.set_ylabel('flux')
    ax.set_title(f'Filtered Response - {qubit_name}')
    figures[f'filtered_response_{qubit_name}'] = fig
    
    # Plot 4: Final results
    response_long = filtered_response_long_1exp[1:]
    flux_q = flux_data[1:].copy()
    flux_q.values = response_long
    filtered_response_Full = lfilter(convolved_fir, iir_coefficients, flux_data[1:].values)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(flux_data.time, flux_data.values, label='Original data')
    ax.plot(flux_data.time[1:], filtered_response_long_1exp[1:], label='Filtered (long time)')
    ax.plot(flux_data.time[1:], filtered_response_Full, label='Filtered (full, deconvolved)')
    ax.axhline(final_vals * 1.001, color='k', linestyle='--', alpha=0.5)
    ax.axhline(final_vals * 0.999, color='k', linestyle='--', alpha=0.5)
    ax.set_ylim([final_vals * 0.95, final_vals * 1.05])
    ax.set_title(f'Filtered Response - {qubit_name}')
    ax.legend()
    figures[f'final_response_{qubit_name}'] = fig
    
    # Plot 5: Normalized final results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(flux_data.time[1:], flux_data[1:].values / np.mean(flux_data[1:].values[-10:]), 
           label='Original data')
    ax.plot(flux_data.time[1:], filtered_response_long_1exp[1:] / np.mean(filtered_response_long_1exp[-10:]), 
           '--', label='Slow rise correction')
    ax.plot(flux_q.time, filtered_response_Full / np.mean(filtered_response_long_1exp[-10:]), 
           '--', label='Expected corrected response')
    ax.axhline(1.001, color='k', linestyle='--', alpha=0.5)
    ax.axhline(0.999, color='k', linestyle='--', alpha=0.5)
    ax.set_ylim([0.95, 1.05])
    ax.legend()
    ax.set_xlabel('time (ns)')
    ax.set_ylabel('normalized amplitude')
    ax.set_title(f'Final Results - {qubit_name}')
    figures[f'normalized_final_{qubit_name}'] = fig
    
    return figures
