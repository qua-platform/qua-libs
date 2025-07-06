import logging
from typing import Any, List, Optional

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from plotly.subplots import make_subplots
from qualang_tools.units import unit
from qualibration_libs.analysis import oscillation
from qualibration_libs.plotting import (PowerRabiPreparator, QubitGrid,
                                        grid_iter)
from qualibration_libs.plotting.configs import PlotStyling
from qualibration_libs.plotting.helpers import add_plotly_top_axis
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)
styling = PlotStyling()


# --- Constants ---
MV_PER_V = 1e3


def _get_data_key(ds: xr.Dataset) -> str:
    """
    Determines the data key ('I' or 'state') from the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to inspect.

    Returns
    -------
    str
        The data key, either "I" or "state".

    Raises
    ------
    RuntimeError
        If the dataset contains neither 'I' nor 'state'.
    """
    if hasattr(ds, "I"):
        return "I"
    if hasattr(ds, "state"):
        return "state"
    raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")


def _add_matplotlib_twin_axis(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], data_key: str):
    """
    Adds a twin axis for amplitude prefactor to a matplotlib plot.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes object.
    ds : xr.Dataset
        The dataset containing the data.
    qubit : dict[str, str]
        A mapping to the qubit to plot.
    data_key : str
        The key for the data to plot ('I' or 'state').
    """
    ax2 = ax.twiny()
    (ds.assign_coords(amp_mV=ds.amp_prefactor).loc[qubit] * MV_PER_V)[data_key].plot(
        ax=ax2, x="amp_mV", alpha=styling.matplotlib_raw_data_alpha
    )
    ax2.set_xlabel("amplitude prefactor")


def _plot_fit_1d_matplotlib(ax: Axes, fit: xr.Dataset):
    """
    Plots the 1D fitted data on a matplotlib axis.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes object.
    fit : xr.Dataset
        The dataset containing the fit parameters.
    """
    fitted_data = oscillation(
        fit.amp_prefactor.data,
        fit.fit.sel(fit_vals="a").data,
        fit.fit.sel(fit_vals="f").data,
        fit.fit.sel(fit_vals="phi").data,
        fit.fit.sel(fit_vals="offset").data,
    )
    ax.plot(fit.full_amp * MV_PER_V, MV_PER_V * fitted_data, linewidth=styling.matplotlib_fit_linewidth, color=styling.fit_color)


def _plot_fit_2d_matplotlib(ax: Axes, fit: xr.Dataset):
    """
    Plots the 2D fitted data (vertical line) on a matplotlib axis.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes object.
    fit : xr.Dataset
        The dataset containing the fit parameters.
    """
    ax.axvline(
        x=fit.opt_amp_prefactor,
        color=styling.fit_color,
        linestyle="-",
        linewidth=styling.matplotlib_fit_linewidth,
    )


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: Optional[xr.Dataset] = None):
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
    data_key = _get_data_key(ds)
    for ax, qubit in grid_iter(grid):
        fit_sel = fits.sel(qubit=qubit["qubit"]) if fits is not None else None
        is_1d = len(ds.nb_of_pulses) == 1

        # Plot raw data
        if is_1d:
            label = "Rotated I quadrature [mV]" if data_key == "I" else "Qubit state"
            (ds.assign_coords(amp_mV=ds.full_amp * MV_PER_V).loc[qubit] * MV_PER_V)[data_key].plot(
                ax=ax, x="amp_mV", alpha=styling.matplotlib_raw_data_alpha
            )
            ax.set_ylabel(label)
            ax.set_xlabel("Pulse amplitude [mV]")
        else:
            (ds.assign_coords(amp_mV=ds.full_amp * MV_PER_V).loc[qubit])[data_key].plot(
                ax=ax, add_colorbar=False, x="amp_mV", y="nb_of_pulses", robust=True
            )
            ax.set_ylabel("Number of pulses")
            ax.set_xlabel("Pulse amplitude [mV]")

        # Plot fit if available and successful
        if fit_sel is not None and hasattr(fit_sel, "outcome") and getattr(fit_sel.outcome, "values", None) == "successful":
            if is_1d:
                _plot_fit_1d_matplotlib(ax, fit_sel)
            else:
                _plot_fit_2d_matplotlib(ax, fit_sel)

        # Add twin axis for prefactor
        _add_matplotlib_twin_axis(ax, ds, qubit, data_key)

    grid.fig.suptitle("Power Rabi")
    grid.fig.set_size_inches(styling.matplotlib_fig_width, styling.matplotlib_fig_height)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit_1D(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
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
    data_key = _get_data_key(ds)
    label = "Rotated I quadrature [mV]" if data_key == "I" else "Qubit state"

    (ds.assign_coords(amp_mV=ds.full_amp * MV_PER_V).loc[qubit] * MV_PER_V)[data_key].plot(
            ax=ax, x="amp_mV", alpha=styling.matplotlib_raw_data_alpha
        )

    if fit is not None and hasattr(fit, "outcome") and getattr(fit.outcome, "values", None) == "successful":
        _plot_fit_1d_matplotlib(ax, fit)

        ax.set_ylabel(label)
        ax.set_xlabel("Pulse amplitude [mV]")
    _add_matplotlib_twin_axis(ax, ds, qubit, data_key)


def plot_individual_data_with_fit_2D(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
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
    data_key = _get_data_key(ds)
    (ds.assign_coords(amp_mV=ds.full_amp * MV_PER_V).loc[qubit])[data_key].plot(
        ax=ax, add_colorbar=False, x="amp_mV", y="nb_of_pulses", robust=True
    )
    ax.set_ylabel("Number of pulses")
    ax.set_xlabel("Pulse amplitude [mV]")

    if fit is not None and hasattr(fit, "outcome") and getattr(fit.outcome, "values", None) == "successful":
        _plot_fit_2d_matplotlib(ax, fit)

    _add_matplotlib_twin_axis(ax, ds, qubit, data_key)


def plotly_plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: Optional[xr.Dataset] = None) -> go.Figure:
    """
    Creates an interactive Plotly figure with the power Rabi data and fitted curves.

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
    go.Figure
        The Plotly figure object containing the plots.

    Notes
    -----
    - Creates a grid of subplots, one for each qubit
    - Each subplot shows raw data and fitted curve if available
    - For 1D data: shows amplitude in mV and prefactor
    - For 2D data: shows heatmap with number of pulses vs amplitude
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits], create_figure=False)
    fig = make_subplots(
        rows=grid.n_rows,
        cols=grid.n_cols,
        subplot_titles=grid.get_subplot_titles(),
        shared_xaxes=False,
        shared_yaxes=False,
    )
    data_key = _get_data_key(ds)
    is_1d = len(ds.nb_of_pulses) == 1

    for (grid_row, grid_col), name_dict in grid.plotly_grid_iter():
        row = grid_row + 1  # Convert to 1-based indexing for Plotly
        col = grid_col + 1  # Convert to 1-based indexing for Plotly
        qubit_id = list(name_dict.values())[0]
        fit = fits.sel(qubit=qubit_id) if fits is not None else None

        if is_1d:
            plotly_plot_individual_data_with_fit_1D(fig, ds, qubit_id, data_key, fit, row, col, grid.n_cols)
        else:
            plotly_plot_individual_data_with_fit_2D(fig, ds, qubit_id, data_key, fit, row, col, grid.n_cols)

    fig.update_layout(
        title="Power Rabi",
        height=styling.plotly_fig_height,
        width=styling.plotly_fig_width,
        showlegend=False,
    )
    return fig


def plotly_plot_individual_data_with_fit_1D(
    fig: go.Figure,
    ds: xr.Dataset,
    qubit: str,
    data_key: str,
    fit: xr.Dataset = None,
    row: int = 1,
    col: int = 1,
    n_cols: int = 1,
):
    """
    Plots 1D power rabi data for a single qubit on a Plotly figure.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure to add the trace to.
    ds : xr.Dataset
        The dataset containing the experiment data.
    qubit : str
        The ID of the qubit to plot.
    data_key : str
        The data key to use for plotting ('I' or 'state').
    fit : xr.Dataset, optional
        Dataset with fit results. Defaults to None.
    row : int, optional
        Subplot row index (1-based). Defaults to 1.
    col : int, optional
        Subplot column index (1-based). Defaults to 1.
    n_cols : int, optional
        Total number of columns in the subplot grid. Defaults to 1.
    """
    label = "Rotated I quadrature [mV]" if data_key == "I" else "Qubit state"
    ds_qubit = ds.sel(qubit=qubit).isel(nb_of_pulses=0)

    amp_mV = ds_qubit["full_amp"].values * MV_PER_V
    amp_prefactor = ds_qubit["amp_prefactor"].values
    y_data = ds_qubit[data_key].values * MV_PER_V
    y_err_da = ds_qubit.get(f"{data_key}_std")
    y_err = y_err_da.values * MV_PER_V if y_err_da is not None else None

    # Plot raw data
    fig.add_trace(
        go.Scatter(
            x=amp_mV,
            y=y_data,
            error_y=dict(type="data", array=y_err, visible=True) if y_err is not None else None,
            name=f"Qubit {qubit} Raw",
            mode="lines+markers",
            line=dict(color=styling.raw_data_color),
            opacity=styling.matplotlib_raw_data_alpha,
            customdata=np.stack([amp_prefactor], axis=-1),
            hovertemplate="Amplitude: %{x:.3f} mV<br>Prefactor: %{customdata[0]:.3f}<br>%{y:.3f} mV<extra></extra>",
        ),
        row=row,
        col=col,
    )

    # Plot fit
    if fit is not None and hasattr(fit, "outcome") and getattr(fit.outcome, "values", None) == "successful":
        fitted_data = oscillation(
            fit.amp_prefactor.data,
            fit.fit.sel(fit_vals="a").data,
            fit.fit.sel(fit_vals="f").data,
            fit.fit.sel(fit_vals="phi").data,
            fit.fit.sel(fit_vals="offset").data,
        )
        fig.add_trace(
            go.Scatter(
                x=fit.full_amp.values * MV_PER_V,
                y=MV_PER_V * fitted_data,
                name=f"Qubit {qubit} - Fit",
                line=dict(color=styling.fit_color, width=styling.matplotlib_fit_linewidth),
            ),
            row=row,
            col=col,
        )

    fig.update_xaxes(title_text="Pulse amplitude [mV]", row=row, col=col)
    fig.update_yaxes(title_text=label, row=row, col=col)
    add_plotly_top_axis(
        fig, row, col, n_cols, amp_mV, amp_prefactor, "Amplitude prefactor"
    )


def plotly_plot_individual_data_with_fit_2D(
    fig: go.Figure,
    ds: xr.Dataset,
    qubit: str,
    data_key: str,
    fit: xr.Dataset = None,
    row: int = 1,
    col: int = 1,
    n_cols: int = 1,
):
    """
    Plots 2D power rabi data for a single qubit on a Plotly figure.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure to add the trace to.
    ds : xr.Dataset
        The dataset containing the experiment data.
    qubit : str
        The ID of the qubit to plot.
    data_key : str
        The data key to use for plotting ('I' or 'state').
    fit : xr.Dataset, optional
        Dataset with fit results. Defaults to None.
    row : int, optional
        Subplot row index (1-based). Defaults to 1.
    col : int, optional
        Subplot column index (1-based). Defaults to 1.
    n_cols : int, optional
        Total number of columns in the subplot grid. Defaults to 1.
    """
    ds_qubit = ds.sel(qubit=qubit)
    amp_mV = ds_qubit["full_amp"].values * MV_PER_V
    amp_prefactor = ds_qubit["amp_prefactor"].values
    nb_of_pulses = ds_qubit["nb_of_pulses"].values
    z_data = ds_qubit[data_key].values

    # Ensure z_data shape is (nb_of_pulses, amp_mV)
    z_plot = z_data.T if z_data.shape[0] != len(nb_of_pulses) else z_data

    # Customdata for hover: each column is the prefactor value
    customdata = np.tile(amp_prefactor, (len(nb_of_pulses), 1))

    # Plot heatmap
    fig.add_trace(
        go.Heatmap(
        z=z_plot,
        x=amp_mV,
        y=nb_of_pulses,
        colorscale=styling.heatmap_colorscale,
        showscale=False,
        customdata=customdata,
        hovertemplate="Amplitude: %{x:.3f} mV<br>Prefactor: %{customdata:.3f}<br>Pulses: %{y}<br>Value: %{z:.3f}<extra></extra>",
        ),
        row=row,
        col=col,
    )

    # Plot fit line
    if fit is not None and hasattr(fit, "outcome") and getattr(fit.outcome, "values", None) == "successful":
        try:
            opt_amp_mV = (
                float(ds_qubit["full_amp"].sel(amp_prefactor=fit.opt_amp_prefactor, method="nearest").values) * MV_PER_V
            )
        except (KeyError, ValueError) as e:
            logging.warning(
                f"Could not select optimal amplitude for qubit {qubit} using xarray, falling back to numpy. Error: {e}"
            )
            opt_amp_mV = float(amp_mV[np.argmin(np.abs(amp_prefactor - fit.opt_amp_prefactor))])

        fig.add_trace(
            go.Scatter(
                x=[opt_amp_mV, opt_amp_mV],
                y=[nb_of_pulses.min(), nb_of_pulses.max()],
                mode="lines",
                line=dict(color=styling.fit_color, width=styling.matplotlib_fit_linewidth),
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

    fig.update_xaxes(title_text="Pulse amplitude [mV]", row=row, col=col)
    fig.update_yaxes(title_text="Number of pulses", row=row, col=col)
    add_plotly_top_axis(
        fig, row, col, n_cols, amp_mV, amp_prefactor, "Amplitude prefactor"
    )


# -----------------------------------------------------------------------------
# Public wrapper functions expected by the refactor
# -----------------------------------------------------------------------------


def create_matplotlib_plot(
    ds_raw_prep: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit_prep: Optional[xr.Dataset] = None,
) -> Figure:
    """Return a Matplotlib figure for a Power-Rabi dataset (raw + optional fit)."""

    return plot_raw_data_with_fit(ds_raw_prep, qubits, ds_fit_prep)


def create_plotly_plot(
    ds_raw_prep: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit_prep: Optional[xr.Dataset] = None,
) -> go.Figure:
    """Return a Plotly figure for a Power-Rabi dataset (raw + optional fit)."""

    return plotly_plot_raw_data_with_fit(ds_raw_prep, qubits, ds_fit_prep)


def create_plots(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit: Optional[xr.Dataset] = None,
) -> tuple[go.Figure, Figure]:
    """Convenience helper that returns (plotly_fig, matplotlib_fig)."""

    ds_raw_prep, ds_fit_prep = PowerRabiPreparator(ds_raw, ds_fit, qubits=qubits).prepare()
    plotly_fig = plotly_plot_raw_data_with_fit(ds_raw_prep, qubits, ds_fit_prep)
    matplotlib_fig = plot_raw_data_with_fit(ds_raw_prep, qubits, ds_fit_prep)
    return plotly_fig, matplotlib_fig