import logging
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from plotly.graph_objs import Figure as PlotlyFigure
from plotly.subplots import make_subplots
from qualang_tools.units import unit
from qualibration_libs.plotting import (
    QubitGrid, ResonatorSpectroscopyVsAmplitudePreparator, grid_iter)
from qualibration_libs.plotting.configs import PlotStyling
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)
styling = PlotStyling()


# --- Constants ---
GHZ_PER_HZ = 1e-9
MHZ_PER_HZ = 1e-6


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: Optional[xr.Dataset] = None):
    """
    Plots the raw data with fitted curves for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset, optional
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
        fit_sel = fits.sel(qubit=qubit["qubit"]) if fits is not None else None
        plot_individual_raw_data_with_fit(ax, ds, qubit, fit_sel)

    grid.fig.suptitle("Resonator spectroscopy vs power")
    grid.fig.set_size_inches(styling.matplotlib_fig_width, styling.matplotlib_fig_height)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_raw_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots individual qubit data on a given axis with optional fit.
    """
    (ds.assign_coords(freq_GHz=ds.full_freq * GHZ_PER_HZ).loc[qubit].IQ_abs).plot(
        ax=ax,
        add_colorbar=False,
        x="freq_GHz",
        y="power",
        linewidth=styling.matplotlib_fit_linewidth,
        zorder=1,
    )
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Power (dBm)")

    # Now create the detuning axis
    ax2 = ax.twiny()
    (ds.assign_coords(detuning_MHz=ds.detuning * MHZ_PER_HZ).loc[qubit].IQ_abs_norm).plot(
        ax=ax2, add_colorbar=False, x="detuning_MHz", y="power", robust=True
    )
    ax2.set_xlabel("Detuning [MHz]")
    # Plot where the optimum readout power was found
    if getattr(fit, "outcome", None) == "successful":
        ax2.axhline(
            y=fit.optimal_power,
            color=styling.optimal_power_color,
            linestyle=styling.matplotlib_optimal_power_linestyle,
            linewidth=styling.matplotlib_fit_linewidth,
        )
        ax2.axvline(
            x=fit.freq_shift * MHZ_PER_HZ,
            color=styling.resonator_freq_color,
            linestyle=styling.matplotlib_resonator_freq_linestyle,
            linewidth=styling.matplotlib_fit_linewidth,
        )
    # ax3 = ax.twinx()
    # ds.assign_coords(readout_amp=ds.detuning / u.MHz).loc[qubit].IQ_abs_norm.plot(
    #     ax=ax3, add_colorbar=False, x="detuning_MHz", y="readout_amp", robust=True
    # )
    # ax.set_title(qubit["qubit"])


def plotly_plot_raw_data_with_fit(
    ds_raw: xr.Dataset, qubits: List[AnyTransmon], fits: Optional[xr.Dataset] = None
) -> go.Figure:
    """
    Plotly version of resonator spectroscopy vs power: one (n_powers×n_freqs) heatmap
    per qubit (with Detuning shown on hover), plus overlays if a fit is successful:
      • Red dashed vertical line at the center resonator frequency.
      • Magenta solid horizontal line at the optimal readout power.

    Requirements (ds_raw):
      - Must have dims (qubit, detuning, power)
      - Must have DataArray "full_freq" or "freq_full" shaping (n_qubits, n_freqs), in Hz
      - Must have DataArray "IQ_abs" (n_qubits, n_freqs, n_powers)
      - Must have coords "power" or "power_dbm" (n_powers,) in dBm
      - Must have coord "detuning" (n_freqs,) in Hz
    Requirements (fits):
      - Must have dims (detuning, power, qubit)
      - Must contain coords:
         • "res_freq"    (qubit,) in Hz
         • "rr_min_response" (qubit, power) in Hz (the detuning from res_freq)
         • "optimal_power"   (qubit,) in dBm
         • "success"         (qubit,) boolean
      - The code will do `fit_ds = fits.sel(qubit=qubit_id)`.
    """
    logger = logging.getLogger("plotly_plot_raw_data_with_fit")
    logger.setLevel(logging.DEBUG)

    # ----------------------------------------------------
    # 1) Transpose ds_raw so that its dims become (qubit, detuning, power)
    # ----------------------------------------------------
    ds2 = ds_raw.transpose("qubit", "detuning", "power")

    # ----------------------------------------------------
    # 2) Pull out the raw arrays
    # ----------------------------------------------------
    if "full_freq" in ds2:
        freq_array = ds2["full_freq"].values  # (n_qubits, n_freqs)
    elif "freq_full" in ds2:
        freq_array = ds2["freq_full"].values
    else:
        raise KeyError("After transpose, dataset must have 'full_freq' or 'freq_full'.")

    if "IQ_abs_norm" not in ds2:
        raise KeyError("After transpose, dataset must have 'IQ_abs_norm' (n_qubits, n_freqs, n_powers).")
    IQ_array = ds2["IQ_abs_norm"].values  # (n_qubits, n_freqs, n_powers)

    # Pick the power axis:
    if "power" in ds2.coords:
        power_array = ds2["power"].values  # (n_powers,) in dBm
    elif "power_dbm" in ds2.coords:
        power_array = ds2["power_dbm"].values
    else:
        raise KeyError("After transpose, dataset must have coord 'power' or 'power_dbm'.")

    # Detuning axis (Hz):
    if "detuning" not in ds2.coords:
        raise KeyError("After transpose, dataset must have coord 'detuning' (Hz).")
    detuning_array = ds2["detuning"].values  # (n_freqs,) in Hz

    n_qubits, n_freqs, n_powers = IQ_array.shape

    # ----------------------------------------------------
    # 3) Build PlotlyQubitGrid to get nrows, ncols, name_dicts
    # ----------------------------------------------------
    grid = QubitGrid(ds2, [q.grid_location for q in qubits], create_figure=False)
    ncols = grid.n_cols
    nrows = grid.n_rows
    subplot_titles = grid.get_subplot_titles()

    # ----------------------------------------------------
    # 4) Create subplots with EXTRA spacing (exactly as raw_data uses)
    # ----------------------------------------------------
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        horizontal_spacing=styling.plotly_horizontal_spacing,
        vertical_spacing=styling.plotly_vertical_spacing,
        shared_xaxes=False,
        shared_yaxes=False,
    )

    # ----------------------------------------------------
    # 5) Precompute per‐subplot zmin/zmax so each colorbar is local
    # ----------------------------------------------------
    per_zmin = []
    per_zmax = []
    for q_idx in range(n_qubits):
        z_mat = IQ_array[q_idx].T
        if z_mat.ndim == 1:
            z_mat = z_mat[np.newaxis, :]
        if z_mat.shape[0] != n_powers:
            z_mat = z_mat.T
        if np.all(np.isnan(z_mat)):
            z_mat = np.zeros_like(z_mat)
        # Use robust percentiles for color scaling, flattening the matrix
        per_zmin.append(float(np.nanpercentile(z_mat.flatten(), 2)))
        per_zmax.append(float(np.nanpercentile(z_mat.flatten(), 98)))

    # ----------------------------------------------------
    # 6) Loop over each subplot = one qubit.  Add Heatmap + overlays
    # ----------------------------------------------------
    q_labels = list(ds2.qubit.values)  # e.g. ["qC1", "qC2", "qC3"]
    heatmap_info: List[Tuple[int, int, int]] = []

    for idx, ((grid_row, grid_col), name_dict) in enumerate(grid.plotly_grid_iter()):
        row = grid_row + 1  # Convert to 1-based indexing for Plotly
        col = grid_col + 1  # Convert to 1-based indexing for Plotly
        qubit_id = list(name_dict.values())[0]
        # Only print debug for the first qubit
        if idx == 0:
            # print("[plotly debug] qubit:", qubit_id)
            pass
        # Find integer index of this qubit in ds2.qubit.values
        try:
            q_idx = q_labels.index(qubit_id)
        except ValueError:
            raise ValueError(f"Could not find qubit '{qubit_id}' in {q_labels}")

        # (a) Build x (freq GHz) and y (power dBm):
        freq_vals = freq_array[q_idx] * GHZ_PER_HZ  # (n_freqs,) in GHz
        power_vals = power_array  # (n_powers,) in dBm

        # (b) Build 2D z‐matrix for heatmap (n_powers, n_freqs):
        z_mat = IQ_array[q_idx].T
        if z_mat.ndim == 1:
            z_mat = z_mat[np.newaxis, :]
        if z_mat.shape[0] != n_powers:
            z_mat = z_mat.T
        if np.all(np.isnan(z_mat)):
            z_mat = np.zeros_like(z_mat)

        # (c) Build a 2D array of Detuning [MHz] for hover:
        detuning_MHz = (detuning_array * MHZ_PER_HZ).astype(float)  # (n_freqs,) in MHz
        det2d = np.tile(detuning_MHz[np.newaxis, :], (n_powers, 1))  # (n_powers, n_freqs)

        # (d) Grab precomputed z‐limits:
        zmin_i = per_zmin[q_idx]
        zmax_i = per_zmax[q_idx]

        # (e) Add the Viridis heatmap (with a "placeholder" colorbar):
        fig.add_trace(
            go.Heatmap(
                z=z_mat,
                x=freq_vals,
                y=power_vals,
                customdata=det2d,
                colorscale=styling.heatmap_colorscale,
                zmin=zmin_i,
                zmax=zmax_i,
                showscale=False,
                colorbar=dict(
                    x=1.0,  # placeholder (we'll move below)
                    y=0.5,
                    len=1.0,
                    thickness=styling.plotly_colorbar_thickness,
                    xanchor="left",
                    yanchor="middle",
                    ticks="outside",
                    ticklabelposition="outside",
                    title="|IQ|",
                ),
                hovertemplate=(
                    "Freq [GHz]: %{x:.3f}<br>"
                    "Power [dBm]: %{y:.2f}<br>"
                    "Detuning [MHz]: %{customdata:.2f}<br>"
                    "|IQ|: %{z:.3f}<extra>Qubit " + qubit_id + "</extra>"
                ),
                name=f"Qubit {qubit_id}",
            ),
            row=row,
            col=col,
        )
        heatmap_idx = len(fig.data) - 1
        heatmap_info.append((heatmap_idx, row, col))

        # (f) Overlay the fit if success=True:
        fit_ds = fits.sel(qubit=qubit_id) if fits is not None else None
        if fit_ds is not None and "outcome" in fit_ds.coords and fit_ds.outcome == "successful":
            res_GHz = float(fit_ds.res_freq.values) * GHZ_PER_HZ
            fig.add_trace(
                go.Scatter(
                    x=[res_GHz, res_GHz],
                    y=[power_vals.min(), power_vals.max()],
                    mode="lines",
                    line=dict(
                        color=styling.resonator_freq_color, width=styling.plotly_resonator_freq_linewidth, dash=styling.plotly_resonator_freq_linestyle
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            opt_pwr = float(fit_ds.optimal_power.values)
            fig.add_trace(
                go.Scatter(
                    x=[freq_vals.min(), freq_vals.max()],
                    y=[opt_pwr, opt_pwr],
                    mode="lines",
                    line=dict(color=styling.optimal_power_color, width=styling.plotly_optimal_power_linewidth),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

        # (g) Tidy axis titles & annotation font
        fig.update_xaxes(title_text="Frequency (GHz)", row=row, col=col)
        fig.update_yaxes(title_text="Power (dBm)", row=row, col=col)
        if idx < len(fig.layout.annotations):
            fig.layout.annotations[idx]["font"] = dict(size=styling.plotly_annotation_font_size)

    # ----------------------------------------------------
    # 7) Reposition each colorbar so it doesn't overlap
    # ----------------------------------------------------
    for hm_idx, row, col in heatmap_info:
        axis_num = (row - 1) * ncols + col
        xaxis_key = f"xaxis{axis_num}"
        yaxis_key = f"yaxis{axis_num}"

        x_dom = fig.layout[xaxis_key].domain
        y_dom = fig.layout[yaxis_key].domain

        # Move bar right by +0.03, width = 0.02, height = 90% of subplot
        x0_cb = x_dom[1] + styling.plotly_colorbar_x_offset
        x1_cb = x0_cb + styling.plotly_colorbar_width
        y0 = y_dom[0]
        y1 = y_dom[1]
        bar_len = (y1 - y0) * styling.plotly_colorbar_height_ratio
        bar_center_y = (y0 + y1) / 2

        hm_trace = fig.data[hm_idx]
        hm_trace.colorbar.update(
            {
                "x": x0_cb,
                "y": bar_center_y,
                "len": bar_len,
                "thickness": styling.plotly_colorbar_thickness,
                "xanchor": "left",
                "yanchor": "middle",
                "ticks": "outside",
                "ticklabelposition": "outside",
                "title": "|IQ|",
            }
        )

    # ----------------------------------------------------
    # 8) Final layout tweaks: set size & margins
    # ----------------------------------------------------
    fig.update_layout(
        width=max(styling.plotly_min_width, styling.plotly_subplot_width * ncols),
        height=styling.plotly_subplot_height * nrows,
        margin=styling.plotly_margin,
        title_text="Resonator Spectroscopy: Power vs Frequency (with fits)",
        showlegend=False,
    )

    return fig


def plotly_plot_raw_data(ds: xr.Dataset, qubits: List[AnyTransmon]) -> go.Figure:
    """
    Plotly version: one heatmap per qubit, each with its own colorbar,
    with 'Detuning [MHz]' included in the hover tooltip.  Subplots do not overlap.

    Requirements (ds):
      - Must have "full_freq" or "freq_full" (Hz) as a DataArray of shape (n_qubits, n_freqs).
      - Must have "IQ_abs_norm" (n_qubits × n_freqs × n_powers).
      - Must have a coordinate "power" or "power_dbm" (length n_powers, in dBm).
      - Must have a coordinate "detuning" (length n_freqs, in Hz).
    qubits:
      - List of qubit objects; we read q.grid_location to position subplots.
    Returns:
      - fig : go.Figure (Plotly) with one heatmap/subplot per qubit.
    """

    # ----------------------------------------------------
    # 1) Pull out raw NumPy arrays from ds by exact names
    # ----------------------------------------------------
    if "full_freq" in ds:
        freq_array = ds["full_freq"].values  # shape = (n_qubits, n_freqs)
    elif "freq_full" in ds:
        freq_array = ds["freq_full"].values
    else:
        raise KeyError("Dataset must have 'full_freq' or 'freq_full' (Hz).")

    if "IQ_abs_norm" not in ds:
        raise KeyError("Dataset must have 'IQ_abs_norm' (n_qubits × n_freqs × n_powers).")
    IQ_array = ds["IQ_abs_norm"].values  # shape = (n_qubits, n_freqs, n_powers)

    # Pick power coordinate:
    if "power" in ds.coords:
        power_array = ds["power"].values  # shape = (n_powers,) in dBm
    elif "power_dbm" in ds.coords:
        power_array = ds["power_dbm"].values
    else:
        raise KeyError("Dataset must have a coordinate 'power' or 'power_dbm' (n_powers).")

    # Detuning (Hz) is also a coordinate of length n_freqs:
    if "detuning" not in ds.coords:
        raise KeyError("Dataset must have a coordinate 'detuning' (Hz).")
    detuning_array = ds["detuning"].values  # shape = (n_freqs,) in Hz

    n_qubits, n_freqs, n_powers = IQ_array.shape

    # ----------------------------------------------------
    # 2) Build PlotlyQubitGrid to figure out grid shape & ordering
    # ----------------------------------------------------
    grid = QubitGrid(ds, [q.grid_location for q in qubits], create_figure=False)
    ncols = grid.n_cols
    nrows = grid.n_rows
    subplot_titles = grid.get_subplot_titles()

    # ----------------------------------------------------
    # 3) Create subplots (with extra spacing so nothing overlaps)
    # ----------------------------------------------------
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        horizontal_spacing=styling.plotly_horizontal_spacing,
        vertical_spacing=styling.plotly_vertical_spacing,
        shared_xaxes=False,
        shared_yaxes=False,
    )

    # ----------------------------------------------------
    # 4) Precompute per‐subplot zmin/zmax (so each heatmap has its own scale)
    # ----------------------------------------------------
    per_zmin = []
    per_zmax = []
    for q_idx in range(n_qubits):
        # Build the (n_powers, n_freqs) array to find min/max
        z_mat = IQ_array[q_idx].T
        if z_mat.ndim == 1:
            z_mat = z_mat[np.newaxis, :]
        if z_mat.shape[0] != n_powers:
            z_mat = z_mat.T
        if np.all(np.isnan(z_mat)):
            z_mat = np.zeros_like(z_mat)
        # Use robust percentiles for color scaling, flattening the matrix
        per_zmin.append(float(np.nanpercentile(z_mat.flatten(), 2)))
        per_zmax.append(float(np.nanpercentile(z_mat.flatten(), 98)))

    # ----------------------------------------------------
    # 5) Loop over each subplot and add a Heatmap + correctly positioned colorbar
    # ----------------------------------------------------
    for idx, ((grid_row, grid_col), name_dict) in enumerate(grid.plotly_grid_iter()):
        row = grid_row + 1  # Convert to 1-based indexing for Plotly
        col = grid_col + 1  # Convert to 1-based indexing for Plotly
        qubit_id = list(name_dict.values())[0]

        # We assume qubits in ds.qubit.values are in the same order as freq_array's first axis:
        # If that is not guaranteed, look up index by name.  For now:
        q_labels = list(ds.qubit.values)
        try:
            q_idx = q_labels.index(qubit_id)
        except ValueError:
            raise ValueError(f"Could not find qubit '{qubit_id}' in ds.qubit.values = {q_labels}")

        # Build the x (freq GHz) and y (power dBm) arrays:
        freq_vals = freq_array[q_idx] * GHZ_PER_HZ  # (n_freqs,) in GHz
        power_vals = power_array  # (n_powers,) in dBm

        # Build the 2D z‐matrix = |IQ| shaped (n_powers, n_freqs):
        z_mat = IQ_array[q_idx].T
        if z_mat.ndim == 1:
            z_mat = z_mat[np.newaxis, :]
        if z_mat.shape[0] != n_powers:
            z_mat = z_mat.T
        if np.all(np.isnan(z_mat)):
            z_mat = np.zeros_like(z_mat)

        # Build a 2D "detuning_MHz" array for hover: same shape as z_mat, each row identical:
        detuning_MHz = (detuning_array * MHZ_PER_HZ).astype(float)  # (n_freqs,) in MHz
        det2d = np.tile(detuning_MHz[np.newaxis, :], (n_powers, 1))  # shape (n_powers, n_freqs)

        # Fetch zmin/zmax just computed:
        zmin_i = per_zmin[q_idx]
        zmax_i = per_zmax[q_idx]

        # Add the heatmap with a "placeholder" colorbar (we'll reposition it in step 6)
        fig.add_trace(
            go.Heatmap(
                z=z_mat,
                x=freq_vals,
                y=power_vals,
                customdata=det2d,
                colorscale=styling.heatmap_colorscale,
                zmin=zmin_i,
                zmax=zmax_i,
                showscale=False,
                colorbar=dict(
                    x=1.0,  # placeholder, will be overwritten
                    y=0.5,
                    len=1.0,
                    thickness=styling.plotly_colorbar_thickness,
                    xanchor="left",
                    yanchor="middle",
                    ticks="outside",
                    ticklabelposition="outside",
                ),
                hovertemplate=(
                    "Freq [GHz]: %{x:.3f}<br>"
                    "Power [dBm]: %{y:.2f}<br>"
                    "Detuning [MHz]: %{customdata:.2f}<br>"
                    "|IQ|: %{z:.3f}<extra>Qubit " + qubit_id + "</extra>"
                ),
                name=f"Qubit {qubit_id}",
            ),
            row=row,
            col=col,
        )
        heatmap_idx = len(fig.data) - 1  # record index of this heatmap
        # Store (trace_index, row, col) to reposition its colorbar later
        # (We'll do that in step 6)

        # Tidy axis titles and annotation font for this subplot:
        fig.update_xaxes(title_text="Frequency (GHz)", row=row, col=col)
        fig.update_yaxes(title_text="Power (dBm)", row=row, col=col)
        if idx < len(fig.layout.annotations):
            fig.layout.annotations[idx]["font"] = dict(size=styling.plotly_annotation_font_size)

    # ----------------------------------------------------
    # 6) Reposition each colorbar so it sits to the right of its subplot
    #    and shrink it to 90% of subplot's height, with "outside" ticks/labels
    # ----------------------------------------------------
    for idx in range(n_qubits):
        axis_num = idx + 1
        xaxis_key = f"xaxis{axis_num}"
        yaxis_key = f"yaxis{axis_num}"

        # The domain of the subplot's axes:
        x_dom = fig.layout[xaxis_key].domain  # [x_start, x_end]
        y_dom = fig.layout[yaxis_key].domain  # [y_start, y_end]

        # Place the colorbar just to the right of x_end by +0.03 of the figure's width
        x0_cb = x_dom[1] + styling.plotly_colorbar_x_offset
        x1_cb = x0_cb + styling.plotly_colorbar_width  # bar width = 2% of figure width
        y0 = y_dom[0]
        y1 = y_dom[1]
        bar_len = (y1 - y0) * styling.plotly_colorbar_height_ratio  # 90% of subplot's vertical span
        bar_center_y = (y0 + y1) / 2

        hm_trace = fig.data[idx]  # this subplot's heatmap trace
        hm_trace.colorbar.update(
            {
                "x": x0_cb,
                "y": bar_center_y,
                "len": bar_len,
                "thickness": styling.plotly_colorbar_thickness,
                "xanchor": "left",
                "yanchor": "middle",
                "ticks": "outside",
                "ticklabelposition": "outside",
            }
        )

    # ----------------------------------------------------
    # 7) Final layout tweaks: set figure size & margins
    # ----------------------------------------------------
    fig.update_layout(
        width=max(styling.plotly_min_width, styling.plotly_subplot_width * ncols),  # ensure at least 1000px wide
        height=styling.plotly_subplot_height * nrows,  # 400px per row
        margin=styling.plotly_margin,
        title_text="Resonator Spectroscopy: Power vs Frequency",
        showlegend=False,
    )

    return fig


# -----------------------------------------------------------------------------
# Public wrapper functions expected by the refactor
# -----------------------------------------------------------------------------


def create_matplotlib_plot(
    ds_raw_prep: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit_prep: Optional[xr.Dataset] = None,
) -> Figure:
    """Return a Matplotlib figure for a Resonator-Spectroscopy-vs-Amplitude dataset (raw + optional fit)."""

    return plot_raw_data_with_fit(ds_raw_prep, qubits, ds_fit_prep)


def create_plotly_plot(
    ds_raw_prep: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit_prep: Optional[xr.Dataset] = None,
) -> go.Figure:
    """Return a Plotly figure for a Resonator-Spectroscopy-vs-Amplitude dataset (raw + optional fit)."""

    return plotly_plot_raw_data_with_fit(ds_raw_prep, qubits, ds_fit_prep)


def create_plots(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit: Optional[xr.Dataset] = None,
) -> tuple[go.Figure, Figure]:
    """Convenience helper that returns (plotly_fig, matplotlib_fig)."""

    ds_raw_prep, ds_fit_prep = ResonatorSpectroscopyVsAmplitudePreparator(ds_raw, ds_fit, qubits=qubits).prepare()
    plotly_fig = plotly_plot_raw_data_with_fit(ds_raw_prep, qubits, ds_fit_prep)
    matplotlib_fig = plot_raw_data_with_fit(ds_raw_prep, qubits, ds_fit_prep)
    return plotly_fig, matplotlib_fig