from typing import Dict
import xarray as xr
import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from quam_builder.architecture.superconducting.qubit import FluxTunableTransmon
from mpl_toolkits.axes_grid1 import make_axes_locatable

from qualang_tools.units import unit

from calibration_utils.crosstalk_spectroscopy_vs_flux.program import get_expected_frequency_at_flux_detuning

u = unit(coerce_to_integer=True)

def plot_analysis(ds: xr.Dataset, peak_results: Dict, fit_results: Dict, flux_detunings: Dict, qubits: Dict[str, FluxTunableTransmon]):
    """
    Plot the full analysis pipeline onto a 2D heatmap for each pair of qubits.
    
    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    peak_results : Dict
        A dictionary of Lorentzian peak fitting results at each detuning for each qubit pair.
    fit_results : Dict
        A dictionary of linear fit results for each qubit pair.
    qubits : list of AnyTransmon
        A list of qubits to plot.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    # Create a simple grid for qubit pairs
    target_qubits = ds.qubit.data
    aggressor_qubits = ds.aggressor.data

    fig, axes = plt.subplots(len(target_qubits), len(aggressor_qubits),
                             figsize=(5*len(target_qubits), 4*len(aggressor_qubits)))

    for i, target_qubit in enumerate(target_qubits):
        for j, aggressor_qubit in enumerate(aggressor_qubits):
            if target_qubit == aggressor_qubit:
                axes[i][j].set_visible(False)
                continue

            fit_result = fit_results[target_qubit][aggressor_qubit]
            peak_result = peak_results.sel(qubit=target_qubit, aggressor=aggressor_qubit)

            ax = axes[i][j]
            ax.set_ylim(ds.full_freq.min(), ds.full_freq.max())

            plot_individual_raw_data(ax, ds, target_qubit, aggressor_qubit)
            plot_individual_peak_frequencies(ax, fit_result, peak_result)
            plot_individual_linear_fit(ax, fit_result, peak_result)

            delta_f =  get_expected_frequency_at_flux_detuning(qubits[target_qubit], flux_detunings[target_qubit]) - \
                       qubits[target_qubit].xy.RF_frequency

            ax.set_title(f"{aggressor_qubit} acting on {target_qubit}\n"
                         f"$\Delta\phi_{{{target_qubit}}} = {1000*flux_detunings[target_qubit]:.1f}\,$mV, "
                         f"$\Delta f_{{{target_qubit}}} = {delta_f/1e6:.0f}\,$MHz")
            ax.set_xlabel(f"{aggressor_qubit} Flux Bias (V)")
    fig.suptitle("Crosstalk Spectroscopy")
    fig.tight_layout()
    return fig


def plot_individual_raw_data(ax: Axes, ds: xr.Dataset, target_qubit_name: str, aggressor_qubit_name: str):
    """
    Plot individual qubit pair raw data on a given axis.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    target_qubit_name : str
        Name of the target qubit.
    aggressor_qubit_name : str
        Name of the aggressor qubit.
    """
    # Select data for this qubit pair
    da = ds.sel(qubit=target_qubit_name).sel(aggressor=aggressor_qubit_name)
    da = da.assign_coords(freq_GHz=(da.full_freq / 1e9))

    # Plot 2D heatmap
    im = da.IQ_abs.plot(
        ax=ax,
        x="flux_bias",
        y="detuning",
        add_colorbar=False,
        robust=True,
    )

    ax.set_ylabel("Frequency (GHz)")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax.get_figure().colorbar(im, cax=cax)
    cbar.set_label(f"{target_qubit_name} Readout IQ Amplitude", fontsize=10, rotation=270, labelpad=15)

    return im


def plot_individual_peak_frequencies(ax: Axes, fit_result: dict, peak_result: xr.Dataset):
    """
    Plot individual qubit pair peak frequencies with error bars.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    results : dict
        Dictionary containing peak frequency data for the qubit pair.
    """
    # Plot with error bars - use bright colors to stand out against heatmap
    mask = xr.DataArray(fit_result["linear_fit_inlier_mask"], dims="flux_bias")
    peak_result = peak_result.dropna("flux_bias").where(mask, drop=True)
    ax.errorbar(peak_result.flux_bias, peak_result.peak_frequencies,
                yerr=peak_result.peak_frequency_errors,
                fmt='s', capsize=2, capthick=1, markersize=3,
                color='r', markerfacecolor='r', markeredgecolor='r',
                markeredgewidth=1, label='Peak Frequencies')

    ax.set_ylabel("Detuning (Hz)")
    ax.legend(loc='upper right')



def plot_individual_linear_fit(ax: Axes, fit_result: dict, peak_result: dict):
    """
    Plot individual qubit pair linear fit overlaid on peak frequency data.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    fit_result : CrosstalkFitParameters
        Fit results for the qubit pair.
    peak_result : dict
        Peak frequency results for the qubit pair.
    """
    flux_bias = peak_result["flux_bias"]

    # Plot linear fit if successful
    if fit_result["success"]:
        # Generate smooth line for fit
        flux_smooth = np.linspace(flux_bias.min(), flux_bias.max(), 100)
        freq_smooth = (fit_result["linear_fit_slope"] * flux_smooth + fit_result["linear_fit_intercept"])
        
        ax.plot(flux_smooth, freq_smooth, 'r-', linewidth=3,
                label=f'Linear Fit\nCrosstalk: {100*fit_result["crosstalk_coefficient"]:.2f}%')
    
    # Update legend to include the linear fit
    ax.legend(loc='upper right')
