from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from plotly.graph_objs import Figure as PlotlyFigure
from plotly.subplots import make_subplots
from qualang_tools.units import unit
from qualibration_libs.analysis import lorentzian_dip
from qualibration_libs.plotting import (QubitGrid,
                                        ResonatorSpectroscopyPreparator,
                                        grid_iter)
from qualibration_libs.plotting.configs import PlotStyling
from qualibration_libs.plotting.helpers import add_plotly_top_axis
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)
styling = PlotStyling()


def plot_raw_phase(ds: xr.Dataset, qubits: List[AnyTransmon]) -> Figure:
    """
    Plots the raw phase data for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list
        A list of qubits to plot.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains two x-axes: one for the full frequency in GHz and one for the detuning in MHz.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax1, qubit in grid_iter(grid):
        # Create a first x-axis for full_freq_GHz
        ds.assign_coords(full_freq_GHz=ds.full_freq / u.GHz).loc[qubit].phase.plot(ax=ax1, x="full_freq_GHz")
        ax1.set_xlabel("RF frequency [GHz]")
        ax1.set_ylabel("phase [rad]")
        # Create a second x-axis for detuning_MHz
        ax2 = ax1.twiny()
        ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[qubit].phase.plot(ax=ax2, x="detuning_MHz")
        ax2.set_xlabel("Detuning [MHz]")
    grid.fig.suptitle("Resonator spectroscopy (phase)")
    grid.fig.set_size_inches(styling.matplotlib_fig_width, styling.matplotlib_fig_height)
    grid.fig.tight_layout()

    return grid.fig


def plot_raw_amplitude_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
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
        plot_individual_amplitude_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Resonator spectroscopy (amplitude + fit)")
    grid.fig.set_size_inches(styling.matplotlib_fig_width, styling.matplotlib_fig_height)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_amplitude_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
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
    - If the fit dataset is provided and the outcome is "successful", the fitted curve is plotted along with the raw data.
    """
    if fit and fit.outcome.values == "successful":
        fitted_data = lorentzian_dip(
            ds.detuning,
            float(fit.amplitude.values),
            float(fit.position.values),
            float(fit.width.values) / 2,
            float(fit.base_line.mean().values),
        )
    else:
        fitted_data = None

    # Create a first x-axis for full_freq_GHz
    (ds.assign_coords(full_freq_GHz=ds.full_freq / u.GHz).loc[qubit].IQ_abs / u.mV).plot(ax=ax, x="full_freq_GHz")
    ax.set_xlabel("RF frequency [GHz]")
    ax.set_ylabel(r"$R=\sqrt{I^2 + Q^2}$ [mV]")
    # Create a second x-axis for detuning_MHz
    ax2 = ax.twiny()
    (ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[qubit].IQ_abs / u.mV).plot(ax=ax2, x="detuning_MHz")
    ax2.set_xlabel("Detuning [MHz]")
    # Plot the fitted data
    if fitted_data is not None:
        ax2.plot(ds.detuning / u.MHz, fitted_data / u.mV, color=styling.fit_color, linestyle="--")

def plotly_plot_raw_phase(ds: xr.Dataset, qubits: List[AnyTransmon]) -> PlotlyFigure:
    """
    Robust Plotly version: only plot RF frequency trace, add detuning as a hover label for each point.
    Adds a visible detuning axis (top x-axis) to each subplot, with ticks/range matching detuning data.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits], create_figure=False)
    fig = make_subplots(
        rows=grid.n_rows,
        cols=grid.n_cols,
        subplot_titles=grid.get_subplot_titles(),
        shared_xaxes=False,
        shared_yaxes=False,
    )
    detuning_axes = []
    for (grid_row, grid_col), name_dict in grid.plotly_grid_iter():
        row = grid_row + 1  # Convert to 1-based indexing for Plotly
        col = grid_col + 1  # Convert to 1-based indexing for Plotly
        qubit_id = list(name_dict.values())[0]
        freq_data = ds.assign_coords(full_freq_GHz=ds.full_freq / u.GHz).loc[name_dict]
        detuning_data = ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[name_dict]
        fig.add_trace(
            go.Scatter(
                x=freq_data.full_freq_GHz,
                y=freq_data.phase,
                name=f"Qubit {qubit_id}",
                showlegend=False,
                line=dict(color=styling.raw_data_color),
                customdata=np.stack([detuning_data.detuning_MHz], axis=-1),
                hovertemplate="RF freq: %{x:.6f} GHz<br>Detuning: %{customdata[0]:.2f} MHz<br>Phase: %{y:.3f} rad<extra></extra>",
            ),
            row=row,
            col=col,
        )
        detuning_axes.append(detuning_data.detuning_MHz.values)
    
    # Add visible detuning axis (top x-axis) for each subplot that has data
    for i, ((grid_row, grid_col), name_dict) in enumerate(grid.plotly_grid_iter()):
        row = grid_row + 1  # Convert to 1-based indexing for Plotly
        col = grid_col + 1
        qubit_id = list(name_dict.values())[0]
        freq_data = ds.assign_coords(full_freq_GHz=ds.full_freq / u.GHz).loc[name_dict]
        detuning_data = ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[name_dict]
        add_plotly_top_axis(fig, row, col, grid.n_cols, freq_data.full_freq_GHz.values, detuning_data.detuning_MHz.values, "Detuning [MHz]")

        fig.update_xaxes(title_text="RF frequency [GHz]", row=row, col=col)
        fig.update_yaxes(title_text="phase [rad]", row=row, col=col)
    fig.update_layout(
        title="Resonator spectroscopy (phase)",
        height=styling.plotly_fig_height,
        width=styling.plotly_fig_width,
        showlegend=False,
    )
    return fig

def plotly_plot_raw_amplitude_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> PlotlyFigure:
    """
    Robust Plotly version: only plot RF frequency trace, add detuning as a hover label for each point. Overlay fit if present.
    Adds a visible detuning axis (top x-axis) to each subplot, with ticks/range matching detuning data.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits], create_figure=False)
    fig = make_subplots(
        rows=grid.n_rows,
        cols=grid.n_cols,
        subplot_titles=grid.get_subplot_titles(),
        shared_xaxes=False,
        shared_yaxes=False,
    )
    detuning_axes = []
    for (grid_row, grid_col), name_dict in grid.plotly_grid_iter():
        row = grid_row + 1  # Convert to 1-based indexing for Plotly
        col = grid_col + 1  # Convert to 1-based indexing for Plotly
        qubit_id = list(name_dict.values())[0]
        fit = fits.sel(qubit=qubit_id)
        freq_data = ds.assign_coords(full_freq_GHz=ds.full_freq / u.GHz).loc[name_dict]
        detuning_data = ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[name_dict]
        y_raw = (freq_data.IQ_abs / u.mV).values
        x_raw = freq_data.full_freq_GHz.values
        detuning_vals = detuning_data.detuning_MHz.values
        fig.add_trace(
            go.Scatter(
                x=x_raw,
                y=y_raw,
                name=f"Qubit {qubit_id}",
                showlegend=False,
                line=dict(color=styling.raw_data_color),
                customdata=np.stack([detuning_vals], axis=-1),
                hovertemplate="RF freq: %{x:.6f} GHz<br>Detuning: %{customdata[0]:.2f} MHz<br>R: %{y:.3f} mV<extra></extra>",
            ),
            row=row,
            col=col,
        )
        # Fit (overlay, same x as raw, on RF frequency axis)
        if fit is not None and fit.outcome.values == "successful":
            fitted_data = lorentzian_dip(
                ds.detuning.values,
                float(fit.amplitude.values),
                float(fit.position.values),
                float(fit.width.values) / 2,
                float(fit.base_line.mean().values),
            )
            fig.add_trace(
                go.Scatter(
                    x=x_raw,
                    y=fitted_data / u.mV,
                    name=f"Qubit {qubit_id} - Fit",
                    line=dict(dash=styling.plotly_fit_linestyle, color=styling.fit_color),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
        detuning_axes.append(detuning_vals)
    
    # Add visible detuning axis (top x-axis) for each subplot that has data
    for i, ((grid_row, grid_col), name_dict) in enumerate(grid.plotly_grid_iter()):
        row = grid_row + 1
        col = grid_col + 1
        qubit_id = list(name_dict.values())[0]
        freq_data = ds.assign_coords(full_freq_GHz=ds.full_freq / u.GHz).loc[name_dict]
        detuning_data = ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[name_dict]
        add_plotly_top_axis(fig, row, col, grid.n_cols, freq_data.full_freq_GHz.values, detuning_data.detuning_MHz.values, "Detuning [MHz]")
        fig.update_xaxes(title_text="RF frequency [GHz]", row=row, col=col)
        fig.update_yaxes(title_text=r"<i>R</i> = √(I² + Q²) [mV]", row=row, col=col)
    fig.update_layout(
        title="Resonator spectroscopy (amplitude + fit)",
        height=styling.plotly_fig_height,
        width=styling.plotly_fig_width,
        showlegend=False,
    )
    return fig



# -----------------------------------------------------------------------------
# Public wrapper functions expected by the refactor
# -----------------------------------------------------------------------------


def create_matplotlib_plots(
    ds_raw_prep: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit_prep: Optional[xr.Dataset] = None,
) -> Dict[str, Figure]:
    """Return a dictionary of Matplotlib figures for a resonator spectroscopy dataset."""

    figs = {
        "phase": plot_raw_phase(ds_raw_prep, qubits),
        "amplitude": plot_raw_amplitude_with_fit(ds_raw_prep, qubits, ds_fit_prep),
    }
    return figs


def create_plotly_plots(
    ds_raw_prep: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit_prep: Optional[xr.Dataset] = None,
) -> Dict[str, PlotlyFigure]:
    """Return a dictionary of Plotly figures for a resonator spectroscopy dataset."""

    figs = {
        "phase": plotly_plot_raw_phase(ds_raw_prep, qubits),
        "amplitude": plotly_plot_raw_amplitude_with_fit(ds_raw_prep, qubits, ds_fit_prep),
    }
    return figs


def create_plots(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit: Optional[xr.Dataset] = None,
) -> Tuple[Dict[str, PlotlyFigure], Dict[str, Figure]]:
    """Convenience helper that returns (plotly_figs, matplotlib_figs)."""
    ds_raw_prep, ds_fit_prep = ResonatorSpectroscopyPreparator(ds_raw, ds_fit, qubits=qubits).prepare()

    plotly_figs = {
        "phase": plotly_plot_raw_phase(ds_raw_prep, qubits),
        "amplitude": plotly_plot_raw_amplitude_with_fit(ds_raw_prep, qubits, ds_fit_prep),
    }
    matplotlib_figs = {
        "phase": plot_raw_phase(ds_raw_prep, qubits),
        "amplitude": plot_raw_amplitude_with_fit(ds_raw_prep, qubits, ds_fit_prep),
    }
    return plotly_figs, matplotlib_figs