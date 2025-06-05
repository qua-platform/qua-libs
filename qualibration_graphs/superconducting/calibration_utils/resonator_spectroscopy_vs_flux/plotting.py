from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from plotly.subplots import make_subplots
from qualang_tools.units import unit
from qualibration_libs.plotting import (PlotlyQubitGrid, QubitGrid, grid_iter,
                                        plotly_grid_iter)
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

    grid.fig.suptitle("Resonator spectroscopy vs flux")
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

    ax2 = ax.twiny()
    # Plot using the attenuated current x-axis
    ds.assign_coords(freq_GHz=ds.full_freq / 1e9).loc[qubit].IQ_abs.plot(
        ax=ax2,
        add_colorbar=False,
        x="attenuated_current",
        y="freq_GHz",
        robust=True,
    )
    ax2.set_xlabel("Current (A)")
    ax2.set_ylabel("Freq (GHz)")
    ax2.set_title("")
    # Move ax2 behind ax
    ax2.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)
    # Plot using the flux x-axis
    ds.assign_coords(freq_GHz=ds.full_freq / 1e9).loc[qubit].IQ_abs.plot(
        ax=ax, add_colorbar=False, x="flux_bias", y="freq_GHz", robust=True
    )
    if fit.fit_results.success.values:
        ax.axvline(
            fit.fit_results.idle_offset,
            linestyle="dashed",
            linewidth=2,
            color="r",
            label="idle offset",
        )
        ax.axvline(
            fit.fit_results.flux_min,
            linestyle="dashed",
            linewidth=2,
            color="orange",
            label="min offset",
        )
        # Location of the current resonator frequency
        ax.plot(
            fit.fit_results.idle_offset.values,
            fit.fit_results.sweet_spot_frequency.values * 1e-9,
            "r*",
            markersize=10,
        )
    ax.set_title(qubit["qubit"])
    ax.set_xlabel("Flux (V)")


def plotly_plot_raw_data_with_fit(
    ds_raw: xr.Dataset,
    qubits:  List[AnyTransmon],
    fits:    xr.Dataset
) -> go.Figure:
    """
    Plotly version of resonator spectroscopy vs flux (with fits).  Each subplot:
      - x = flux_bias [V]
      - y = RF frequency [GHz]
      - color = |IQ| (Viridis)
      - hover shows (Flux [V], Current [A], Freq [GHz], Detuning [MHz], |IQ|)
    If fit_results.success is True, overlay:
      • Red dashed vertical line at idle_offset (V)
      • Orange dashed vertical line at flux_min (V)
      • Magenta "×" marker at (idle_offset, sweet_spot_frequency/1e9)

    Requirements (ds_raw):
      - dims = (qubit, detuning, flux_bias)
      - DataArray "full_freq" or "freq_full" with shape (n_qubits, n_freqs), in Hz
      - DataArray "IQ_abs" with shape (n_qubits, n_freqs, n_flux)
      - coord "flux_bias"   (n_flux,) in V
      - coord "detuning"    (n_freqs,) in Hz
      - coord "attenuated_current" (n_flux,) in A

    Requirements (fits):
      - dims = (qubit,)
      - under fits.fit_results:
          • idle_offset             (qubit,) in V
          • flux_min                (qubit,) in V
          • sweet_spot_frequency    (qubit,) in Hz
          • success                 (qubit,) boolean
    """
    # 1) Transpose so dims = (qubit, detuning, flux_bias)
    ds2 = ds_raw.transpose("qubit", "detuning", "flux_bias")

    # 2) Pull out raw numpy arrays
    if "full_freq" in ds2:
        freq_array = ds2["full_freq"].values   # (n_qubits, n_freqs)
    elif "freq_full" in ds2:
        freq_array = ds2["freq_full"].values
    else:
        raise KeyError("After transpose, dataset must have 'full_freq' or 'freq_full'.")

    if "IQ_abs" not in ds2:
        raise KeyError("After transpose, dataset must have 'IQ_abs' (n_qubits, n_freqs, n_flux).")
    IQ_array = ds2["IQ_abs"].values            # (n_qubits, n_freqs, n_flux)

    if "flux_bias" not in ds2.coords:
        raise KeyError("After transpose, dataset must have coord 'flux_bias' (V).")
    flux_array = ds2["flux_bias"].values       # (n_flux,) in V

    if "attenuated_current" not in ds2.coords:
        raise KeyError("After transpose, dataset must have coord 'attenuated_current' (A).")
    current_array = ds2["attenuated_current"].values  # (n_flux,) in A

    if "detuning" not in ds2.coords:
        raise KeyError("After transpose, dataset must have coord 'detuning' (Hz).")
    detuning_array = ds2["detuning"].values    # (n_freqs,) in Hz

    n_qubits, n_freqs, n_flux = IQ_array.shape

    # 3) Build PlotlyQubitGrid → get nrows, ncols, name_dicts
    grid = PlotlyQubitGrid(ds2, [q.grid_location for q in qubits])
    ncols = grid.n_cols
    nrows = grid.n_rows
    subplot_titles = [f"Qubit {list(nd.values())[0]}" for nd in grid.name_dicts]

    # 4) Create subplots with extra spacing
    fig = make_subplots(
        rows               = nrows,
        cols               = ncols,
        subplot_titles     = subplot_titles,
        horizontal_spacing = 0.25,   # ~15% for colorbars/overlays
        vertical_spacing   = 0.12,   # ~12% gap between rows
        shared_xaxes       = False,
        shared_yaxes       = False
    )

    # 5) Precompute per-subplot zmin/zmax (each heatmap gets its own color scale)
    per_zmin = []
    per_zmax = []
    for q_idx in range(n_qubits):
        # reshape so that z_mat has shape (n_freqs, n_flux)
        z_mat = IQ_array[q_idx]  # (n_freqs, n_flux)
        if z_mat.ndim == 1:
            z_mat = z_mat[np.newaxis, :]
        if z_mat.shape != (n_freqs, n_flux):
            # fallback: transpose if necessary
            z_mat = z_mat.T
        if np.all(np.isnan(z_mat)):
            z_mat = np.zeros_like(z_mat)
        per_zmin.append(float(np.nanmin(z_mat)))
        per_zmax.append(float(np.nanmax(z_mat)))

    # 6) Loop over each subplot = one qubit. Add Heatmap + overlays
    q_labels = list(ds2.qubit.values)  # e.g. ["qC1","qC2","qC3"]
    heatmap_info: List[Tuple[int,int,int]] = []
    x_flux_list = []
    x_current_list = []
    hovertemplate_flux_list = []
    hovertemplate_current_list = []
    y_vals_list = []
    z_mats_list = []
    customdatas_list = []
    overlay_x_flux = []
    overlay_x_current = []
    overlay_types = []
    overlay_trace_idxs = []

    for idx, name_dict in enumerate(grid.name_dicts):
        row      = (idx // ncols) + 1
        col      = (idx % ncols)  + 1
        qubit_id = list(name_dict.values())[0]

        # find integer index of qubit_id in ds2.qubit.values
        try:
            q_idx = q_labels.index(qubit_id)
        except ValueError:
            raise ValueError(f"Could not find qubit '{qubit_id}' in {q_labels}")

        # (a) Build x = flux [V], y = freq [GHz]
        freq_vals = freq_array[q_idx] / 1e9   # (n_freqs,) in GHz
        flux_vals = flux_array               # (n_flux,) in V
        current_vals = current_array         # (n_flux,) in A

        # (b) Form 2D z-matrix = |IQ| shaped (n_freqs, n_flux)
        z_mat = IQ_array[q_idx]  # (n_freqs, n_flux)
        if z_mat.ndim == 1:
            z_mat = z_mat[np.newaxis, :]
        if z_mat.shape != (n_freqs, n_flux):
            # fallback: transpose if needed
            z_mat = z_mat.T
        if np.all(np.isnan(z_mat)):
            z_mat = np.zeros_like(z_mat)

        # (c) Build a 2D "detuning_MHz" array for hover (shape = (n_freqs, n_flux))
        detuning_MHz = (detuning_array / 1e6).astype(float)        # (n_freqs,) in MHz
        det2d        = np.tile(detuning_MHz[:, None], (1, n_flux))  # shape = (n_freqs, n_flux)

        # (d) Build a 2D "current" array for hover (shape = (n_freqs, n_flux))
        current2d    = np.tile(current_array[np.newaxis, :], (n_freqs, 1))  # shape = (n_freqs, n_flux)

        # (e) Stack them so that customdata[...,0] = detuning_MHz, customdata[...,1] = current
        customdata = np.stack([det2d, current2d], axis=-1)  # shape = (n_freqs, n_flux, 2)

        # (f) Grab local zmin/zmax
        zmin_i = per_zmin[q_idx]
        zmax_i = per_zmax[q_idx]

        # (g) Add the Viridis heatmap with a "placeholder" colorbar
        fig.add_trace(
            go.Heatmap(
                z             = z_mat,
                x             = flux_vals,
                y             = freq_vals,
                customdata    = customdata,
                colorscale    = "Viridis",
                zmin          = zmin_i,
                zmax          = zmax_i,
                showscale     = True,
                colorbar      = dict(
                    x         = 1.0,
                    y         = 0.5,
                    len       = 1.0,
                    thickness = 14,
                    xanchor   = "left",
                    yanchor   = "middle",
                    ticks     = "outside",
                    ticklabelposition = "outside",
                    title     = "|IQ|"
                ),
                hovertemplate = (
                    "Flux [V]: %{x:.3f}<br>"
                    "Current [A]: %{customdata[1]:.6f}<br>"
                    "Freq [GHz]: %{y:.3f}<br>"
                    "Detuning [MHz]: %{customdata[0]:.2f}<br>"
                    "|IQ|: %{z:.3f}<extra>Qubit " + qubit_id + "</extra>"
                ),
                name          = f"Qubit {qubit_id}"
            ),
            row = row, col = col
        )
        heatmap_idx = len(fig.data) - 1
        heatmap_info.append((heatmap_idx, row, col))

        # (h) Overlay the fit if success=True
        fit_ds = fits.sel(qubit=qubit_id)
        if "success" in fit_ds.coords and bool(fit_ds.success):
            # pull from fit_results
            flux_offset = float(fit_ds.fit_results.idle_offset.values)
            min_offset  = float(fit_ds.fit_results.flux_min.values)
            sweet_spot  = float(fit_ds.fit_results.sweet_spot_frequency.values) / 1e9

            # Find corresponding current values for overlays
            # Use np.interp to map flux to current
            flux_to_current = lambda f: np.interp(f, flux_vals, current_vals)
            flux_offset_current = flux_to_current(flux_offset)
            min_offset_current = flux_to_current(min_offset)

            # —— Magenta "×" at (idle_offset, sweet_spot_freq)
            fig.add_trace(
                go.Scatter(
                    x          = [flux_offset],
                    y          = [sweet_spot],
                    mode       = "markers",
                    marker     = dict(symbol="x", color="magenta", size=12),
                    showlegend = False,
                    hoverinfo  = "skip"
                ),
                row = row, col = col
            )
            subplot_overlay_x_flux = [flux_offset]
            subplot_overlay_x_current = [flux_offset_current]
            subplot_overlay_types = [{'type': 'marker'}]
            # —— RED dashed vertical line at idle_offset
            fig.add_trace(
                go.Scatter(
                    x          = [flux_offset, flux_offset],
                    y          = [freq_vals.min(), freq_vals.max()],
                    mode       = "lines",
                    line       = dict(color="red", width=2, dash="dash"),
                    showlegend = False,
                    hoverinfo  = "skip"
                ),
                row = row, col = col
            )
            subplot_overlay_x_flux.append([flux_offset, flux_offset])
            subplot_overlay_x_current.append([flux_offset_current, flux_offset_current])
            subplot_overlay_types.append({'type': 'line'})
            # —— ORANGE dashed vertical line at flux_min
            fig.add_trace(
                go.Scatter(
                    x          = [min_offset, min_offset],
                    y          = [freq_vals.min(), freq_vals.max()],
                    mode       = "lines",
                    line       = dict(color="orange", width=2, dash="dash"),
                    showlegend = False,
                    hoverinfo  = "skip"
                ),
                row = row, col = col
            )
            subplot_overlay_x_flux.append([min_offset, min_offset])
            subplot_overlay_x_current.append([min_offset_current, min_offset_current])
            subplot_overlay_types.append({'type': 'line'})
        else:
            subplot_overlay_x_flux = []
            subplot_overlay_x_current = []
            subplot_overlay_types = []

        # (i) Tidy axis titles & annotation font
        fig.update_xaxes(title_text="Flux bias [V]",        row=row, col=col)
        fig.update_yaxes(title_text="RF frequency [GHz]",    row=row, col=col)
        fig.layout.annotations[idx]["font"] = dict(size=16)

        # Save for toggling
        x_flux_list.append(flux_vals)
        x_current_list.append(current_vals)
        y_vals_list.append(freq_vals)
        z_mats_list.append(z_mat)
        customdatas_list.append(customdata)
        hovertemplate_flux = (
            "Flux [V]: %{x:.3f}<br>"
            "Current [A]: %{customdata[1]:.6f}<br>"
            "Freq [GHz]: %{y:.3f}<br>"
            "Detuning [MHz]: %{customdata[0]:.2f}<br>"
            "|IQ|: %{z:.3f}<extra>Qubit " + qubit_id + "</extra>"
        )
        hovertemplate_flux_list.append(hovertemplate_flux)

    # 7) Reposition & shrink each colorbar so it sits to the right of its subplot
    for (hm_idx, row, col) in heatmap_info:
        axis_num  = (row - 1)*ncols + col
        xaxis_key = f"xaxis{axis_num}"
        yaxis_key = f"yaxis{axis_num}"

        x_dom = fig.layout[xaxis_key].domain
        y_dom = fig.layout[yaxis_key].domain

        x0_cb        = x_dom[1] + 0.03
        x1_cb        = x0_cb + 0.02   # bar width = 2% of figure width
        y0           = y_dom[0]
        y1           = y_dom[1]
        bar_len      = (y1 - y0) * 0.90  # 90% of subplot height
        bar_center_y = (y0 + y1) / 2

        hm_trace = fig.data[hm_idx]
        hm_trace.colorbar.update({
            "x"                   : x0_cb,
            "y"                   : bar_center_y,
            "len"                 : bar_len,
            "thickness"           : 14,
            "xanchor"             : "left",
            "yanchor"             : "middle",
            "ticks"               : "outside",
            "ticklabelposition"   : "outside"
        })

    # 8) Final layout tweaks: size & margins
    fig.update_layout(
        width       = max(1000, 400 * ncols),
        height      = 400 * nrows,
        margin      = dict(l=60, r=60, t=80, b=60),
        title_text  = "Resonator Spectroscopy: Flux vs Frequency (with fits)",
        showlegend  = False
    )

    return fig


def plotly_plot_raw_data(
    ds: xr.Dataset,
    qubits: List[AnyTransmon]
) -> go.Figure:
    """
    Plotly version: one heatmap per qubit, each with its own colorbar,
    with 'Detuning [MHz]' included in the hover tooltip.  Subplots do not overlap.

    Requirements (ds):
      - Must have "full_freq" or "freq_full" (Hz) as a DataArray of shape (n_qubits, n_freqs).
      - Must have "IQ_abs" (n_qubits × n_freqs × n_flux).
      - Must have a coordinate "flux_bias" (length n_flux, in V).
      - Must have a coordinate "detuning" (length n_freqs, in Hz).
    qubits:
      - List of qubit objects; we read q.grid_location to position subplots.
    Returns:
      - fig : go.Figure (Plotly) with one heatmap/subplot per qubit.
    """

    # 1) Pull out raw NumPy arrays from ds
    if "full_freq" in ds:
        freq_array = ds["full_freq"].values    # (n_qubits, n_freqs)
    elif "freq_full" in ds:
        freq_array = ds["freq_full"].values
    else:
        raise KeyError("Dataset must have 'full_freq' or 'freq_full' (Hz).")

    if "IQ_abs" not in ds:
        raise KeyError("Dataset must have 'IQ_abs' (n_qubits × n_freqs × n_flux).")
    IQ_array = ds["IQ_abs"].values            # (n_qubits, n_freqs, n_flux)

    if "flux_bias" not in ds.coords:
        raise KeyError("Dataset must have a coordinate 'flux_bias' (V).")
    flux_array = ds["flux_bias"].values       # (n_flux,) in V

    if "detuning" not in ds.coords:
        raise KeyError("Dataset must have a coordinate 'detuning' (Hz).")
    detuning_array = ds["detuning"].values    # (n_freqs,) in Hz

    if "attenuated_current" not in ds.coords:
        raise KeyError("Dataset must have coord 'attenuated_current' (A).")
    current_array = ds["attenuated_current"].values  # (n_flux,) in A


    n_qubits, n_freqs, n_flux = IQ_array.shape

    # 2) Build PlotlyQubitGrid for arrangement
    grid = PlotlyQubitGrid(ds, [q.grid_location for q in qubits])
    ncols = grid.n_cols
    nrows = grid.n_rows
    subplot_titles = [f"Qubit {list(nd.values())[0]}" for nd in grid.name_dicts]

    # 3) Create subplots with extra spacing
    fig = make_subplots(
        rows               = nrows,
        cols               = ncols,
        subplot_titles     = subplot_titles,
        horizontal_spacing = 0.25,
        vertical_spacing   = 0.12,
        shared_xaxes       = False,
        shared_yaxes       = False
    )

    # 4) Precompute per‐subplot zmin/zmax
    per_zmin = []
    per_zmax = []
    for q_idx in range(n_qubits):
        z_mat = IQ_array[q_idx].T
        if z_mat.ndim == 1:
            z_mat = z_mat[np.newaxis, :]
        if z_mat.shape[0] != n_flux:
            z_mat = z_mat.T
        if np.all(np.isnan(z_mat)):
            z_mat = np.zeros_like(z_mat)
        per_zmin.append(float(np.nanmin(z_mat)))
        per_zmax.append(float(np.nanmax(z_mat)))

    # 5) Add heatmaps
    x_flux_list = []
    x_current_list = []
    hovertemplate_flux_list = []
    hovertemplate_current_list = []
    y_vals_list = []
    z_mats_list = []
    customdatas_list = []
    for idx, name_dict in enumerate(grid.name_dicts):
        row      = (idx // ncols) + 1
        col      = (idx % ncols)  + 1
        qubit_id = list(name_dict.values())[0]

        q_labels = list(ds.qubit.values)
        try:
            q_idx = q_labels.index(qubit_id)
        except ValueError:
            raise ValueError(f"Could not find qubit '{qubit_id}' in ds.qubit.values = {q_labels}")

        freq_vals = freq_array[q_idx] / 1e9   # (n_freqs,) in GHz
        flux_vals = flux_array               # (n_flux,) in V
        current_vals = current_array         # (n_flux,) in A

        z_mat = IQ_array[q_idx].T
        if z_mat.ndim == 1:
            z_mat = z_mat[np.newaxis, :]
        if z_mat.shape[0] != n_flux:
            z_mat = z_mat.T
        if np.all(np.isnan(z_mat)):
            z_mat = np.zeros_like(z_mat)

        detuning_MHz = (detuning_array / 1e6).astype(float)
        det2d = np.tile(detuning_MHz[np.newaxis, :], (n_flux, 1))

        current2d = np.tile(current_array[np.newaxis, :], (n_flux, 1))

        zmin_i = per_zmin[q_idx]
        zmax_i = per_zmax[q_idx]

        fig.add_trace(
            go.Heatmap(
                z             = z_mat,
                x             = freq_vals,
                y             = flux_vals,
                customdata    = current2d,
                colorscale    = "Viridis",
                zmin          = zmin_i,
                zmax          = zmax_i,
                showscale     = True,
                colorbar      = dict(
                    x         = 1.0,
                    y         = 0.5,
                    len       = 1.0,
                    thickness = 14,
                    xanchor   = "left",
                    yanchor   = "middle",
                    ticks     = "outside",
                    ticklabelposition = "outside",
                    title     = "|IQ|"
                ),
                hovertemplate = (
                    "Freq [GHz]: %{x:.3f}<br>"
                    "Flux [V]: %{y:.3f}<br>"
                    "Current [A]: %{customdata:.6f}<br>"
                    "|IQ|: %{z:.3f}<extra>Qubit " + qubit_id + "</extra>"
                ),
                name          = f"Qubit {qubit_id}"
            ),
            row = row, col = col
        )

        fig.update_xaxes(title_text="RF frequency [GHz]", row=row, col=col)
        fig.update_yaxes(title_text="Flux bias [V]",       row=row, col=col)
        fig.layout.annotations[idx]["font"] = dict(size=16)

        # Save for toggling
        x_flux_list.append(flux_vals)
        x_current_list.append(current_vals)
        y_vals_list.append(freq_vals)
        z_mats_list.append(z_mat)
        customdatas_list.append(current2d)
        hovertemplate_flux = (
            "Freq [GHz]: %{x:.3f}<br>"
            "Flux [V]: %{y:.3f}<br>"
            "Current [A]: %{customdata:.6f}<br>"
            "|IQ|: %{z:.3f}<extra>Qubit " + qubit_id + "</extra>"
        )
        hovertemplate_flux_list.append(hovertemplate_flux)

    # 6) Reposition each colorbar
    for idx in range(n_qubits):
        axis_num  = idx + 1
        xaxis_key = f"xaxis{axis_num}"
        yaxis_key = f"yaxis{axis_num}"

        x_dom = fig.layout[xaxis_key].domain
        y_dom = fig.layout[yaxis_key].domain

        x0_cb        = x_dom[1] + 0.03
        x1_cb        = x0_cb + 0.02
        y0           = y_dom[0]
        y1           = y_dom[1]
        bar_len      = (y1 - y0) * 0.90
        bar_center_y = (y0 + y1) / 2

        hm_trace = fig.data[idx]
        hm_trace.colorbar.update({
            "x"                   : x0_cb,
            "y"                   : bar_center_y,
            "len"                 : bar_len,
            "thickness"           : 14,
            "xanchor"             : "left",
            "yanchor"             : "middle",
            "ticks"               : "outside",
            "ticklabelposition"   : "outside"
        })

    # 7) Final layout tweaks
    fig.update_layout(
        width       = max(1000, 400 * ncols),
        height      = 400 * nrows,
        margin      = dict(l=60, r=60, t=80, b=60),
        title_text  = "Resonator Spectroscopy: Flux vs Frequency",
        showlegend  = False
    )

    return fig