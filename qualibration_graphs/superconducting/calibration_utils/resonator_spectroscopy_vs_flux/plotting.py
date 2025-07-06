from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualang_tools.units import unit
from qualibration_libs.plotting import (QubitGrid,
                                        ResonatorSpectroscopyVsFluxPreparator,
                                        grid_iter)
from qualibration_libs.plotting.base import BasePlotter
from qualibration_libs.plotting.configs import PlotStyling
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)
styling = PlotStyling()


# --- Constants ---
GHZ_PER_HZ = 1e-9
MHZ_PER_HZ = 1e-6


class ResonatorSpectroscopyVsFluxPlotter(BasePlotter):
    """Plotter for resonator spectroscopy vs flux experiments."""

    def __init__(
        self,
        ds_raw: xr.Dataset,
        qubits: List[AnyTransmon],
        ds_fit: Optional[xr.Dataset] = None,
    ):
        super().__init__(ds_raw, qubits, ds_fit)
        # Precompute z-limits for consistent color scaling across subplots
        self.per_zmin = []
        self.per_zmax = []
        for qubit in self.qubits:
            qubit_id = qubit.name
            z_mat = self.ds_raw["IQ_abs"].sel(qubit=qubit_id).values
            if z_mat.ndim == 1:
                z_mat = z_mat[np.newaxis, :]
            if np.all(np.isnan(z_mat)):
                z_mat = np.zeros_like(z_mat)
            self.per_zmin.append(float(np.nanmin(z_mat)))
            self.per_zmax.append(float(np.nanmax(z_mat)))

    def get_plot_title(self) -> str:
        return "Resonator Spectroscopy: Flux vs Frequency (with fits)"

    def _get_make_subplots_kwargs(self) -> dict:
        return {
            "shared_xaxes": False,
            "shared_yaxes": False,
            "horizontal_spacing": styling.plotly_horizontal_spacing,
            "vertical_spacing": styling.plotly_vertical_spacing,
        }

    def _get_final_layout_updates(self) -> dict:
        return {
            "width": max(styling.plotly_min_width, styling.plotly_subplot_width * self.grid.n_cols),
            "height": styling.plotly_subplot_height * self.grid.n_rows,
            "margin": styling.plotly_margin,
            "showlegend": False,
        }

    def _plot_matplotlib_subplot(self, ax: Axes, qubit_dict: dict, fit_data: Optional[xr.Dataset]):
        ax2 = ax.twiny()
        # Plot using the attenuated current x-axis
        (self.ds_raw.assign_coords(freq_GHz=self.ds_raw.full_freq * GHZ_PER_HZ).loc[qubit_dict].IQ_abs).plot(
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
        (self.ds_raw.assign_coords(freq_GHz=self.ds_raw.full_freq * GHZ_PER_HZ).loc[qubit_dict].IQ_abs).plot(
            ax=ax, add_colorbar=False, x="flux_bias", y="freq_GHz", robust=True
        )
        if fit_data is not None and hasattr(fit_data, "outcome") and fit_data.outcome.values == "successful":
            ax.axvline(
                fit_data.fit_results.idle_offset,
                linestyle=styling.matplotlib_idle_offset_linestyle,
                linewidth=styling.matplotlib_idle_offset_linewidth,
                color=styling.idle_offset_color,
                label="idle offset",
            )
            ax.axvline(
                fit_data.fit_results.flux_min,
                linestyle=styling.matplotlib_idle_offset_linestyle,
                linewidth=styling.matplotlib_idle_offset_linewidth,
                color=styling.min_offset_color,
                label="min offset",
            )
            # Location of the current resonator frequency
            ax.plot(
                fit_data.fit_results.idle_offset.values,
                fit_data.fit_results.sweet_spot_frequency.values * GHZ_PER_HZ,
                marker=styling.matplotlib_sweet_spot_marker,
                color=styling.sweet_spot_color,
                markersize=styling.matplotlib_sweet_spot_markersize,
                linestyle="None",
            )
        ax.set_title(qubit_dict["qubit"])
        ax.set_xlabel("Flux (V)")

    def _plot_plotly_subplot(self, fig: go.Figure, qubit_id: str, row: int, col: int, fit_data: Optional[xr.Dataset]):
        q_idx = self.ds_raw.qubit.values.tolist().index(qubit_id)

        # (a) Build x = flux [V], y = freq [GHz]
        freq_vals = self.ds_raw["full_freq"].sel(qubit=qubit_id).values * GHZ_PER_HZ
        flux_vals = self.ds_raw["flux_bias"].values
        current_vals = self.ds_raw["attenuated_current"].values

        # (b) Form 2D z-matrix = |IQ| shaped (n_freqs, n_flux)
        z_mat = self.ds_raw["IQ_abs"].sel(qubit=qubit_id).values.T
        if z_mat.ndim == 1:
            z_mat = z_mat[np.newaxis, :]
        if np.all(np.isnan(z_mat)):
            z_mat = np.zeros_like(z_mat)

        # (c) Build a 2D "detuning_MHz" array for hover
        detuning_MHz = (self.ds_raw["detuning"].values * MHZ_PER_HZ).astype(float)
        det2d = np.tile(detuning_MHz[:, None], (1, len(flux_vals)))

        # (d) Build a 2D "current" array for hover
        current2d = np.tile(current_vals[np.newaxis, :], (len(freq_vals), 1))

        # (e) Stack them so that customdata[...,0] = detuning_MHz, customdata[...,1] = current
        customdata = np.stack([det2d, current2d], axis=-1)

        # (f) Grab local zmin/zmax
        zmin_i = self.per_zmin[q_idx]
        zmax_i = self.per_zmax[q_idx]

        fig.add_trace(
            go.Heatmap(
                z=z_mat,
                x=flux_vals,
                y=freq_vals,
                customdata=customdata,
                colorscale=styling.heatmap_colorscale,
                zmin=zmin_i,
                zmax=zmax_i,
                showscale=True,
                colorbar=dict(
                    x=1.0,
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
                    "Flux [V]: %{x:.3f}<br>"
                    "Current [A]: %{customdata[1]:.6f}<br>"
                    "Freq [GHz]: %{y:.3f}<br>"
                    "Detuning [MHz]: %{customdata[0]:.2f}<br>"
                    "|IQ|: %{z:.3f}<extra>Qubit " + qubit_id + "</extra>"
                ),
                name=f"Qubit {qubit_id}",
            ),
            row=row,
            col=col,
        )

        if fit_data is not None and "outcome" in fit_data.coords and fit_data.outcome.values == "successful":
            flux_offset = float(fit_data.fit_results.idle_offset.values)
            min_offset = float(fit_data.fit_results.flux_min.values)
            sweet_spot = float(fit_data.fit_results.sweet_spot_frequency.values) * GHZ_PER_HZ

            fig.add_trace(
                go.Scatter(
                    x=[flux_offset],
                    y=[sweet_spot],
                    mode="markers",
                    marker=dict(
                        symbol=styling.plotly_sweet_spot_marker_symbol,
                        color=styling.sweet_spot_color,
                        size=styling.plotly_sweet_spot_marker_size,
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=[flux_offset, flux_offset],
                    y=[freq_vals.min(), freq_vals.max()],
                    mode="lines",
                    line=dict(
                        color=styling.idle_offset_color,
                        width=styling.plotly_fit_linewidth,
                        dash=styling.plotly_fit_linestyle,
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=[min_offset, min_offset],
                    y=[freq_vals.min(), freq_vals.max()],
                    mode="lines",
                    line=dict(
                        color=styling.min_offset_color,
                        width=styling.plotly_fit_linewidth,
                        dash=styling.plotly_fit_linestyle,
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(title_text="Flux bias [V]", row=row, col=col)
        fig.update_yaxes(title_text="RF frequency [GHz]", row=row, col=col)
        current_annotation_index = (row - 1) * self.grid.n_cols + (col - 1)
        if current_annotation_index < len(fig.layout.annotations):
            fig.layout.annotations[current_annotation_index]["font"] = dict(
                size=styling.plotly_annotation_font_size
            )

    def _after_plotly_plotting_loop(self, fig: go.Figure):
        trace_counter = 0
        for i, ((grid_row, grid_col), name_dict) in enumerate(self.grid.plotly_grid_iter()):
            row = grid_row + 1
            col = grid_col + 1
            
            # This logic assumes the first trace for a subplot is always the heatmap
            hm_trace = fig.data[trace_counter]
            
            qubit_id = list(name_dict.values())[0]
            fit_ds = self.ds_fit.sel(qubit=qubit_id) if self.ds_fit is not None else None
            num_traces_per_subplot = 1
            if fit_ds is not None and "outcome" in fit_ds.coords and fit_ds.outcome.values == "successful":
                num_traces_per_subplot += 3 # Heatmap + 3 fit traces

            axis_num  = (row - 1)*self.grid.n_cols + col
            xaxis_key = f"xaxis{axis_num}" if axis_num > 1 else "xaxis"
            yaxis_key = f"yaxis{axis_num}" if axis_num > 1 else "yaxis"

            if xaxis_key in fig.layout and yaxis_key in fig.layout:
                x_dom = fig.layout[xaxis_key].domain
                y_dom = fig.layout[yaxis_key].domain

                x0_cb = x_dom[1] + styling.plotly_colorbar_x_offset
                y0 = y_dom[0]
                y1 = y_dom[1]
                bar_len = (y1 - y0) * styling.plotly_colorbar_height_ratio
                bar_center_y = (y0 + y1) / 2

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
            trace_counter += num_traces_per_subplot


def create_matplotlib_plot(
    ds_raw_prep: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit_prep: Optional[xr.Dataset] = None,
) -> Figure:
    """Return a Matplotlib figure for a Resonator-Spectroscopy-vs-Flux dataset (raw + optional fit)."""
    plotter = ResonatorSpectroscopyVsFluxPlotter(ds_raw_prep, qubits, ds_fit_prep)
    return plotter.create_matplotlib_plot()


def create_plotly_plot(
    ds_raw_prep: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit_prep: Optional[xr.Dataset] = None,
) -> go.Figure:
    """Return a Plotly figure for a Resonator-Spectroscopy-vs-Flux dataset (raw + optional fit)."""
    plotter = ResonatorSpectroscopyVsFluxPlotter(ds_raw_prep, qubits, ds_fit_prep)
    return plotter.create_plotly_plot()


def create_plots(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit: Optional[xr.Dataset] = None,
) -> tuple[go.Figure, Figure]:
    """Convenience helper that returns (plotly_fig, matplotlib_fig)."""
    ds_raw_prep, ds_fit_prep = ResonatorSpectroscopyVsFluxPreparator(ds_raw, ds_fit, qubits=qubits).prepare()
    plotter = ResonatorSpectroscopyVsFluxPlotter(ds_raw_prep, qubits, ds_fit_prep)
    return plotter.plot()
