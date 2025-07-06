import logging
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualang_tools.units import unit
from qualibration_libs.plotting import (
    QubitGrid, ResonatorSpectroscopyVsAmplitudePreparator, grid_iter)
from qualibration_libs.plotting.base import BasePlotter
from qualibration_libs.plotting.configs import PlotStyling
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)
styling = PlotStyling()


# --- Constants ---
GHZ_PER_HZ = 1e-9
MHZ_PER_HZ = 1e-6


class ResonatorSpectroscopyVsAmplitudePlotter(BasePlotter):
    """Plotter for resonator spectroscopy vs amplitude experiments."""

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
            z_mat = self.ds_raw["IQ_abs_norm"].sel(qubit=qubit_id).values.T
            if z_mat.ndim == 1:
                z_mat = z_mat[np.newaxis, :]
            if np.all(np.isnan(z_mat)):
                z_mat = np.zeros_like(z_mat)
            self.per_zmin.append(float(np.nanpercentile(z_mat.flatten(), 2)))
            self.per_zmax.append(float(np.nanpercentile(z_mat.flatten(), 98)))

    def get_plot_title(self) -> str:
        return "Resonator Spectroscopy: Power vs Frequency (with fits)"

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
        (self.ds_raw.assign_coords(freq_GHz=self.ds_raw.full_freq * GHZ_PER_HZ).loc[qubit_dict].IQ_abs).plot(
            ax=ax,
            add_colorbar=False,
            x="freq_GHz",
            y="power",
            linewidth=styling.matplotlib_fit_linewidth,
            zorder=1,
        )
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Power (dBm)")

        ax2 = ax.twiny()
        (self.ds_raw.assign_coords(detuning_MHz=self.ds_raw.detuning * MHZ_PER_HZ).loc[qubit_dict].IQ_abs_norm).plot(
            ax=ax2, add_colorbar=False, x="detuning_MHz", y="power", robust=True
        )
        ax2.set_xlabel("Detuning [MHz]")
        if fit_data is not None and getattr(fit_data, "outcome", None) == "successful":
            ax2.axhline(
                y=fit_data.optimal_power,
                color=styling.optimal_power_color,
                linestyle=styling.matplotlib_optimal_power_linestyle,
                linewidth=styling.matplotlib_fit_linewidth,
            )
            ax2.axvline(
                x=fit_data.freq_shift * MHZ_PER_HZ,
                color=styling.resonator_freq_color,
                linestyle=styling.matplotlib_resonator_freq_linestyle,
                linewidth=styling.matplotlib_fit_linewidth,
            )

    def _plot_plotly_subplot(self, fig: go.Figure, qubit_id: str, row: int, col: int, fit_data: Optional[xr.Dataset]):
        q_idx = self.ds_raw.qubit.values.tolist().index(qubit_id)

        # Build x (freq GHz) and y (power dBm):
        freq_vals = self.ds_raw["full_freq"].sel(qubit=qubit_id).values * GHZ_PER_HZ
        power_vals = self.ds_raw["power"].values

        # Build 2D z‐matrix for heatmap (n_powers, n_freqs):
        z_mat = self.ds_raw["IQ_abs_norm"].sel(qubit=qubit_id).values.T
        if z_mat.ndim == 1:
            z_mat = z_mat[np.newaxis, :]
        if np.all(np.isnan(z_mat)):
            z_mat = np.zeros_like(z_mat)

        # Build a 2D array of Detuning [MHz] for hover:
        detuning_MHz = (self.ds_raw["detuning"].values * MHZ_PER_HZ).astype(float)
        det2d = np.tile(detuning_MHz[np.newaxis, :], (len(power_vals), 1))

        zmin_i = self.per_zmin[q_idx]
        zmax_i = self.per_zmax[q_idx]

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

        if fit_data is not None and "outcome" in fit_data.coords and fit_data.outcome == "successful":
            res_GHz = float(fit_data.res_freq.values) * GHZ_PER_HZ
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
            opt_pwr = float(fit_data.optimal_power.values)
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

        fig.update_xaxes(title_text="Frequency (GHz)", row=row, col=col)
        fig.update_yaxes(title_text="Power (dBm)", row=row, col=col)
        
        # This assumes annotations are added in the same order as subplots
        current_annotation_index = (row - 1) * self.grid.n_cols + (col - 1)
        if current_annotation_index < len(fig.layout.annotations):
            fig.layout.annotations[current_annotation_index]["font"] = dict(size=styling.plotly_annotation_font_size)

    def _after_plotly_plotting_loop(self, fig: go.Figure):
        trace_idx = 0
        for i, ((grid_row, grid_col), name_dict) in enumerate(
            self.grid.plotly_grid_iter()
        ):
            qubit_id = list(name_dict.values())[0]
            row, col = grid_row + 1, grid_col + 1

            # The first trace for each subplot should be the heatmap.
            hm_trace = fig.data[trace_idx]

            # Determine how many traces were added for this subplot to correctly advance the index.
            fit_data = (
                self.ds_fit.sel(qubit=qubit_id) if self.ds_fit is not None else None
            )
            num_traces_this_subplot = 1  # Start with the heatmap
            if (
                fit_data is not None
                and "outcome" in fit_data.coords
                and fit_data.outcome == "successful"
            ):
                num_traces_this_subplot += 2  # Add the two fit lines

            trace_idx += num_traces_this_subplot

            axis_num = (row - 1) * self.grid.n_cols + col
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

                if not isinstance(hm_trace, go.Heatmap):
                    logging.warning(
                        f"Trace for qubit {qubit_id} is not a Heatmap, skipping colorbar update."
                    )
                    continue

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


def create_matplotlib_plot(
    ds_raw_prep: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit_prep: Optional[xr.Dataset] = None,
) -> Figure:
    """Return a Matplotlib figure for a Resonator-Spectroscopy-vs-Amplitude dataset (raw + optional fit)."""
    plotter = ResonatorSpectroscopyVsAmplitudePlotter(ds_raw_prep, qubits, ds_fit_prep)
    return plotter.create_matplotlib_plot()


def create_plotly_plot(
    ds_raw_prep: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit_prep: Optional[xr.Dataset] = None,
) -> go.Figure:
    """Return a Plotly figure for a Resonator-Spectroscopy-vs-Amplitude dataset (raw + optional fit)."""
    plotter = ResonatorSpectroscopyVsAmplitudePlotter(ds_raw_prep, qubits, ds_fit_prep)
    return plotter.create_plotly_plot()


def create_plots(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit: Optional[xr.Dataset] = None,
) -> tuple[go.Figure, Figure]:
    """Convenience helper that returns (plotly_fig, matplotlib_fig)."""
    ds_raw_prep, ds_fit_prep = ResonatorSpectroscopyVsAmplitudePreparator(ds_raw, ds_fit, qubits=qubits).prepare()
    plotter = ResonatorSpectroscopyVsAmplitudePlotter(ds_raw_prep, qubits, ds_fit_prep)
    return plotter.plot()