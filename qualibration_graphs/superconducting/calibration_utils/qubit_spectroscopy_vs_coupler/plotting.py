import json
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
    Plots the raw data with avoided crossings marked with red crosses.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters (should include peak_frequency).

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains the raw 2D data and marks the avoided crossings with red crosses.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_raw_data_with_fit(ax, ds, qubit, fits)

    grid.fig.suptitle("Qubit spectroscopy vs coupler flux - Avoided crossings")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_raw_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots individual qubit data on a given axis with avoided crossings marked with red crosses.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        mapping to the qubit to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters including peak_frequency and fit results (default is None).

    Notes
    -----
    - If the fit dataset is provided, the peak frequency curve is plotted along with red crosses
      marking the positions of avoided crossings.
    """
    import matplotlib.pyplot as plt

    qubit_name = qubit["qubit"]

    # Plot the 2D spectroscopy data
    ds_plot = ds.assign_coords(freq_GHz=ds.full_freq / 1e9).loc[qubit]
    im = ds_plot.I.plot(ax=ax, add_colorbar=True, x="flux_bias", y="freq_GHz", robust=True, cmap="viridis")

    # Plot peak frequency curve if available
    if fit is not None and "peak_frequency" in fit.data_vars:
        try:
            # Select the peak frequency for this specific qubit
            # Try different methods to access the qubit dimension
            peak_freq_da = fit.peak_frequency
            if "qubit" in peak_freq_da.dims:
                # Find the index of this qubit in the dimension
                if "qubit" in peak_freq_da.coords:
                    # Has coordinate, can use sel
                    peak_freq = peak_freq_da.sel(qubit=qubit_name)
                else:
                    # No coordinate, find by matching dimension values
                    qubit_dim = peak_freq_da.dims[peak_freq_da.dims.index("qubit")]
                    qubit_values = peak_freq_da[qubit_dim].values
                    if isinstance(qubit_values[0], str):
                        qubit_idx = np.where(qubit_values == qubit_name)[0]
                    else:
                        # Try to match by name attribute or convert to string
                        qubit_idx = np.where([str(q) == qubit_name for q in qubit_values])[0]

                    if len(qubit_idx) > 0:
                        peak_freq = peak_freq_da.isel(qubit=qubit_idx[0])
                    else:
                        raise KeyError(f"Qubit {qubit_name} not found in peak_frequency")
            else:
                raise KeyError("qubit dimension not found in peak_frequency")

            # Remove NaN values for plotting
            valid_mask = ~np.isnan(peak_freq)
            if np.any(valid_mask):
                peak_freq_valid = peak_freq[valid_mask]
                flux_bias_vals = peak_freq_valid.flux_bias

                # Get the RF frequency for this qubit (from the dataset or use a reference)
                # peak_freq is the detuning, so we need to add the RF frequency
                # Get RF frequency from the dataset's full_freq coordinate
                rf_freq = ds_plot.full_freq.mean(dim=["detuning"]).values
                if np.isnan(rf_freq) or rf_freq == 0:
                    # Fallback: use the first flux bias point's mean full_freq
                    rf_freq = float(ds_plot.full_freq.isel(flux_bias=0).mean(dim="detuning").values)

                # Calculate full frequency: detuning + RF_frequency
                peak_freq_full = peak_freq_valid + rf_freq
                peak_freq_GHz = peak_freq_full / 1e9

                # Plot peak frequency as red scatter points
                ax.scatter(
                    flux_bias_vals,
                    peak_freq_GHz,
                    c="red",
                    s=30,
                    marker="o",
                    alpha=0.7,
                    label="Peak frequency",
                    zorder=5,
                )
        except (KeyError, ValueError, IndexError):
            # If selection fails, skip plotting peak frequency
            pass

    # Mark avoided crossings with red crosses if fit results are available
    if fit is not None:
        try:
            # Try to get success status - handle both coordinate and attribute access
            if "success" in fit.coords:
                success = bool(fit.success.sel(qubit=qubit_name).values)
            elif "success" in fit.data_vars:
                success = bool(fit.success.sel(qubit=qubit_name).values)
            else:
                success = False

            if success and "avoided_crossing_flux_biases" in fit.attrs:
                # Parse JSON string back to dict (stored as JSON for netCDF serialization)
                crossing_dict_str = fit.attrs["avoided_crossing_flux_biases"]
                if isinstance(crossing_dict_str, str):
                    crossing_dict = json.loads(crossing_dict_str)
                else:
                    # Backward compatibility: if it's already a dict (in-memory)
                    crossing_dict = crossing_dict_str
                crossing_flux_biases = crossing_dict.get(qubit_name, [])

                # Get RF frequency for conversion (used for both hyperbolic fit and crosses)
                rf_freq = ds_plot.full_freq.mean(dim=["detuning"]).values
                if np.isnan(rf_freq) or rf_freq == 0:
                    rf_freq = float(ds_plot.full_freq.isel(flux_bias=0).mean(dim="detuning").values)

                # Plot the smoothed peak frequency fit used in the analysis
                if "smoothed_peak_frequency" in fit.data_vars:
                    try:
                        # Get smoothed peak frequency for this qubit
                        smoothed_peak_freq_da = fit.smoothed_peak_frequency
                        if "qubit" in smoothed_peak_freq_da.dims:
                            if "qubit" in smoothed_peak_freq_da.coords:
                                smoothed_peak_freq = smoothed_peak_freq_da.sel(qubit=qubit_name)
                            else:
                                qubit_dim = smoothed_peak_freq_da.dims[smoothed_peak_freq_da.dims.index("qubit")]
                                qubit_values = smoothed_peak_freq_da[qubit_dim].values
                                if isinstance(qubit_values[0], str):
                                    qubit_idx = np.where(qubit_values == qubit_name)[0]
                                else:
                                    qubit_idx = np.where([str(q) == qubit_name for q in qubit_values])[0]
                                if len(qubit_idx) > 0:
                                    smoothed_peak_freq = smoothed_peak_freq_da.isel(qubit=qubit_idx[0])
                                else:
                                    raise KeyError(f"Qubit {qubit_name} not found")
                        else:
                            raise KeyError("qubit dimension not found")

                        # Remove NaN values
                        valid_mask = ~np.isnan(smoothed_peak_freq)
                        if np.any(valid_mask):
                            smoothed_peak_freq_valid = smoothed_peak_freq[valid_mask]
                            flux_bias_vals_smooth = smoothed_peak_freq_valid.flux_bias

                            # Add RF frequency to get full frequency, then convert to GHz
                            smoothed_freq_full_Hz = smoothed_peak_freq_valid + rf_freq
                            smoothed_freq_GHz = smoothed_freq_full_Hz / 1e9

                            # Plot the smoothed fit curve
                            ax.plot(
                                flux_bias_vals_smooth,
                                smoothed_freq_GHz,
                                "b-",
                                linewidth=2,
                                alpha=0.8,
                                label="Smoothed fit",
                                zorder=7,
                            )

                            # Mark avoided crossings with red crosses using the smoothed fit
                            for crossing_flux in crossing_flux_biases:
                                # Find the frequency at this flux bias by interpolating on smoothed data
                                crossing_freq_GHz = np.interp(
                                    crossing_flux, flux_bias_vals_smooth.data, smoothed_freq_GHz.data
                                )

                                # Plot red cross
                                ax.plot(
                                    crossing_flux,
                                    crossing_freq_GHz,
                                    "r+",
                                    markersize=15,
                                    markeredgewidth=3,
                                    label="Avoided crossing" if crossing_flux == crossing_flux_biases[0] else "",
                                    zorder=10,
                                )
                    except (KeyError, ValueError, IndexError):
                        # If smoothed data selection fails, mark crossings at middle of y-axis
                        y_mid = np.mean(ax.get_ylim())
                        for crossing_flux in crossing_flux_biases:
                            ax.plot(
                                crossing_flux,
                                y_mid,
                                "r+",
                                markersize=15,
                                markeredgewidth=3,
                                label="Avoided crossing" if crossing_flux == crossing_flux_biases[0] else "",
                                zorder=10,
                            )

                # If smoothed fit is not available, fallback to marking crossings at middle of y-axis
                if "smoothed_peak_frequency" not in fit.data_vars:
                    y_mid = np.mean(ax.get_ylim())
                    for crossing_flux in crossing_flux_biases:
                        ax.plot(
                            crossing_flux,
                            y_mid,
                            "r+",
                            markersize=15,
                            markeredgewidth=3,
                            label="Avoided crossing" if crossing_flux == crossing_flux_biases[0] else "",
                            zorder=10,
                        )
        except (KeyError, AttributeError, ValueError):
            # Fit results not available, skip marking
            pass

    ax.set_title(qubit_name)
    ax.set_xlabel("Coupler Flux Bias (V)")
    ax.set_ylabel("Frequency (GHz)")
    if ax.get_legend() is not None:
        ax.legend(loc="best", fontsize=8)

    return ax
