import logging
from typing import List

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from plotly.subplots import make_subplots
from qualang_tools.units import unit
from qualibration_libs.analysis import oscillation
from qualibration_libs.plotting import (PlotlyQubitGrid, QubitGrid, grid_iter,
                                        plotly_grid_iter)
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


# --- Constants ---
MV_PER_V = 1e3
RAW_DATA_COLOR = "#1f77b4"
FIT_COLOR = "#FF0000"  # Red
FIT_LINE_STYLE = "dash"
FIT_LINE_WIDTH = 2
RAW_DATA_ALPHA = 0.5
MATPLOTLIB_FIG_WIDTH = 15
MATPLOTLIB_FIG_HEIGHT = 9
PLOTLY_FIG_HEIGHT = 900
PLOTLY_FIG_WIDTH = 1500
PLOTLY_COLORSACLE = "Viridis"
# For overlaying axes in plotly
PLOTLY_AXIS_OFFSET = 100


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
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
        if len(ds.nb_of_pulses) == 1:
            plot_individual_data_with_fit_1D(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))
        else:
            plot_individual_data_with_fit_2D(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Power Rabi")
    grid.fig.set_size_inches(MATPLOTLIB_FIG_WIDTH, MATPLOTLIB_FIG_HEIGHT)
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

    if len(ds.nb_of_pulses) == 1:
        fitted_data = None
        plot_fit = False
        if fit is not None and hasattr(fit, "outcome") and getattr(fit.outcome, "values", None) == "successful":
            fitted_data = oscillation(
                fit.amp_prefactor.data,
                fit.fit.sel(fit_vals="a").data,
                fit.fit.sel(fit_vals="f").data,
                fit.fit.sel(fit_vals="phi").data,
                fit.fit.sel(fit_vals="offset").data,
            )
            plot_fit = True

        if hasattr(ds, "I"):
            data = "I"
            label = "Rotated I quadrature [mV]"
        elif hasattr(ds, "state"):
            data = "state"
            label = "Qubit state"
        else:
            raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")

        (ds.assign_coords(amp_mV=ds.full_amp * MV_PER_V).loc[qubit] * MV_PER_V)[data].plot(
            ax=ax, x="amp_mV", alpha=RAW_DATA_ALPHA
        )
        if plot_fit:
            ax.plot(fit.full_amp * MV_PER_V, MV_PER_V * fitted_data, linewidth=FIT_LINE_WIDTH, color=FIT_COLOR)
        ax.set_ylabel(label)
        ax.set_xlabel("Pulse amplitude [mV]")
        ax2 = ax.twiny()
        (ds.assign_coords(amp_mV=ds.amp_prefactor).loc[qubit] * MV_PER_V)[data].plot(
            ax=ax2, x="amp_mV", alpha=RAW_DATA_ALPHA
        )
        ax2.set_xlabel("amplitude prefactor")


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

    if hasattr(ds, "I"):
        data = "I"
    elif hasattr(ds, "state"):
        data = "state"
    else:
        raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")
    (ds.assign_coords(amp_mV=ds.full_amp * MV_PER_V).loc[qubit])[data].plot(
        ax=ax, add_colorbar=False, x="amp_mV", y="nb_of_pulses", robust=True
    )
    ax.set_ylabel(f"Number of pulses")
    ax.set_xlabel("Pulse amplitude [mV]")
    ax2 = ax.twiny()
    (ds.assign_coords(amp_mV=ds.amp_prefactor).loc[qubit])[data].plot(
        ax=ax2, add_colorbar=False, x="amp_mV", y="nb_of_pulses", robust=True
    )
    ax2.set_xlabel("amplitude prefactor")
    if fit.outcome.values == "successful":
        ax2.axvline(
            x=fit.opt_amp_prefactor,
            color=FIT_COLOR,
            linestyle="-",
            linewidth=FIT_LINE_WIDTH,
        )


def plotly_plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> go.Figure:
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
    grid = PlotlyQubitGrid(ds, [q.grid_location for q in qubits])
    fig = make_subplots(
        rows=grid.n_rows,
        cols=grid.n_cols,
        subplot_titles=[f"Qubit {list(nd.values())[0]}" for nd in grid.name_dicts],
        shared_xaxes=False,
        shared_yaxes=False,
    )
    colorbar_traces = []
    for i, name_dict in plotly_grid_iter(grid):
        row = i // grid.n_cols + 1
        col = i % grid.n_cols + 1
        qubit_id = list(name_dict.values())[0]
        fit = fits.sel(qubit=qubit_id)
        if len(ds.nb_of_pulses) == 1:
            plotly_plot_individual_data_with_fit_1D(fig, ds, qubit_id, fit, row, col)
        else:
            plotly_plot_individual_data_with_fit_2D(fig, ds, qubit_id, fit, row, col)

    fig.update_layout(
        title="Power Rabi",
        height=PLOTLY_FIG_HEIGHT,
        width=PLOTLY_FIG_WIDTH,
        showlegend=False,
    )
    return fig

def plotly_plot_individual_data_with_fit_1D(
    fig: go.Figure, ds: xr.Dataset, qubit: str, fit: xr.Dataset = None, row: int = 1, col: int = 1, n_cols: int = 1
):
    if hasattr(ds, "I"):
        data = "I"
        label = "Rotated I quadrature [mV]"
    elif hasattr(ds, "state"):
        data = "state"
        label = "Qubit state"
    else:
        raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")
    # Select correct 1D slice for qubit and pulse
    ds_qubit = ds.sel(qubit=qubit)
    if "nb_of_pulses" in ds_qubit.dims and ds_qubit.sizes["nb_of_pulses"] == 1:
        ds_qubit = ds_qubit.isel(nb_of_pulses=0)
    amp_mV = ds_qubit["full_amp"].values * MV_PER_V
    amp_prefactor = ds_qubit["amp_prefactor"].values
    y_data = ds_qubit[data].values * MV_PER_V
    y_err = ds_qubit.get(f"{data}_std", None)
    if y_err is not None:
        y_err = y_err.values * MV_PER_V
    fig.add_trace(
        go.Scatter(
            x=amp_mV,
            y=y_data,
            error_y=dict(type="data", array=y_err, visible=True) if y_err is not None else None,
            name=f"Qubit {qubit} Raw",
            showlegend=False,
            mode="lines+markers",
            line=dict(color=RAW_DATA_COLOR),
            opacity=RAW_DATA_ALPHA,
            customdata=np.stack([amp_prefactor], axis=-1),
            hovertemplate="Amplitude: %{x:.3f} mV<br>Prefactor: %{customdata[0]:.3f}<br>%{y:.3f} mV<extra></extra>",
        ),
        row=row,
        col=col,
    )
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
                x=fit.full_amp * MV_PER_V,
                y=MV_PER_V * fitted_data,
                name=f"Qubit {qubit} - Fit",
                line=dict(color=FIT_COLOR, width=FIT_LINE_WIDTH),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    fig.update_xaxes(title_text="Pulse amplitude [mV]", row=row, col=col)
    fig.update_yaxes(title_text=label, row=row, col=col)
    subplot_index = (row - 1) * n_cols + col
    main_xaxis = f"x{subplot_index}"
    top_xaxis_layout = f"xaxis{subplot_index + PLOTLY_AXIS_OFFSET}"
    fig["layout"][top_xaxis_layout] = dict(
        overlaying=main_xaxis,
        side="top",
        title="Amplitude prefactor",
        showgrid=False,
        tickmode="array",
        tickvals=list(amp_mV),
        ticktext=[f"{v:.2f}" for v in amp_prefactor],
        range=[float(np.min(amp_prefactor)), float(np.max(amp_prefactor))],
    )


def plotly_plot_individual_data_with_fit_2D(fig: go.Figure, ds: xr.Dataset, qubit: str, fit: xr.Dataset = None, row: int = 1, col: int = 1, n_cols: int = 1):
    if hasattr(ds, "I"):
        data = "I"
    elif hasattr(ds, "state"):
        data = "state"
    else:
        raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")
    ds_qubit = ds.sel(qubit=qubit)
    amp_mV = ds_qubit['full_amp'].values * MV_PER_V
    amp_prefactor = ds_qubit['amp_prefactor'].values
    nb_of_pulses = ds_qubit['nb_of_pulses'].values
    z_data = ds_qubit[data].values
    # Ensure z_data shape is (nb_of_pulses, amp_mV)
    if z_data.shape[0] == len(nb_of_pulses) and z_data.shape[1] == len(amp_mV):
        z_plot = z_data
    elif z_data.shape[1] == len(nb_of_pulses) and z_data.shape[0] == len(amp_mV):
        z_plot = z_data.T
    else:
        z_plot = z_data
    # Customdata for hover: each column is the prefactor value
    customdata = np.tile(amp_prefactor, (len(nb_of_pulses), 1))
    hm_trace = go.Heatmap(
        z=z_plot,
        x=amp_mV,
        y=nb_of_pulses,
        colorscale=PLOTLY_COLORSACLE,
        showscale=False,
        colorbar=dict(
            title="|IQ|",
            titleside="right",
            ticks="outside",
        ),
        customdata=customdata,
        hovertemplate="Amplitude: %{x:.3f} mV<br>Prefactor: %{customdata:.3f}<br>Pulses: %{y}<br>Value: %{z:.3f}<extra></extra>",
    )
    fig.add_trace(hm_trace, row=row, col=col)
    hm_idx = len(fig.data) - 1
    # Overlay fit at correct amplitude (convert prefactor to amplitude)
    if fit is not None and hasattr(fit, "outcome") and getattr(fit.outcome, "values", None) == "successful":
        # Find amplitude in mV corresponding to fit.opt_amp_prefactor
        try:
            opt_amp_mV = (
                float(ds_qubit["full_amp"].sel(amp_prefactor=fit.opt_amp_prefactor, method="nearest").values) * MV_PER_V
            )
        except Exception as e:
            logging.warning(
                f"Could not select optimal amplitude for qubit {qubit} using xarray, falling back to numpy. Error: {e}"
            )
            opt_amp_mV = float(amp_mV[np.argmin(np.abs(amp_prefactor - fit.opt_amp_prefactor))])
        fig.add_trace(
            go.Scatter(
                x=[opt_amp_mV, opt_amp_mV],
                y=[nb_of_pulses.min(), nb_of_pulses.max()],
                mode="lines",
                line=dict(color=FIT_COLOR, width=FIT_LINE_WIDTH),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
    fig.update_xaxes(title_text="Pulse amplitude [mV]", row=row, col=col)
    fig.update_yaxes(title_text="Number of pulses", row=row, col=col)
    subplot_index = (row - 1) * n_cols + col
    main_xaxis = f"x{subplot_index}"
    top_xaxis_layout = f"xaxis{subplot_index + PLOTLY_AXIS_OFFSET}"
    fig['layout'][top_xaxis_layout] = dict(
        overlaying=main_xaxis,
        side='top',
        title="Amplitude prefactor",
        showgrid=False,
        tickmode='array',
        tickvals=list(amp_mV),
        ticktext=[f"{v:.2f}" for v in amp_prefactor],
        range=[float(np.min(amp_prefactor)), float(np.max(amp_prefactor))],
    )
    return hm_idx, row, col