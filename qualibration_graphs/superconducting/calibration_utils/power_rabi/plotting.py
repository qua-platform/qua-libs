import logging
from typing import Any, List, Optional

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualang_tools.units import unit
from qualibration_libs.analysis import oscillation
from qualibration_libs.plotting import (PowerRabiPreparator, QubitGrid,
                                        grid_iter)
from qualibration_libs.plotting.base import BasePlotter
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
    """
    if hasattr(ds, "I"):
        return "I"
    if hasattr(ds, "state"):
        return "state"
    raise RuntimeError("The dataset must contain either 'I' or 'state'.")


class PowerRabiPlotter(BasePlotter):
    """
    A plotter for Power Rabi experiments.
    """

    def __init__(
        self,
        ds_raw: xr.Dataset,
        qubits: List[AnyTransmon],
        ds_fit: Optional[xr.Dataset] = None,
    ):
        super().__init__(ds_raw, qubits, ds_fit)
        self.data_key = _get_data_key(self.ds_raw)
        self.is_1d = len(self.ds_raw.nb_of_pulses) == 1

    def get_plot_title(self) -> str:
        return "Power Rabi"

    def _add_matplotlib_twin_axis(self, ax: Axes, ds: xr.Dataset, qubit: dict[str, str]):
        """
        Adds a twin axis for amplitude prefactor to a matplotlib plot.
        """
        ax2 = ax.twiny()
        ax2.set_xlabel("amplitude prefactor")

        if self.is_1d:
            (ds.assign_coords(amp_mV=ds.amp_prefactor).loc[qubit] * MV_PER_V)[self.data_key].plot(
                ax=ax2, x="amp_mV", alpha=self.styling.matplotlib_raw_data_alpha
            )
        else:
            # For 2D, we just want to set the axis, not replot the heatmap
            amp_prefactor_coord = self.ds_raw.amp_prefactor.values
            ax2.set_xlim(np.min(amp_prefactor_coord), np.max(amp_prefactor_coord))

    def _plot_fit_1d_matplotlib(self, ax: Axes, fit: xr.Dataset):
        """
        Plots the 1D fitted data on a matplotlib axis.
        """
        fitted_data = oscillation(
            fit.amp_prefactor.data,
            fit.fit.sel(fit_vals="a").data,
            fit.fit.sel(fit_vals="f").data,
            fit.fit.sel(fit_vals="phi").data,
            fit.fit.sel(fit_vals="offset").data,
        )
        ax.plot(fit.full_amp * MV_PER_V, MV_PER_V * fitted_data, linewidth=self.styling.matplotlib_fit_linewidth, color=self.styling.fit_color)

    def _plot_fit_2d_matplotlib(self, ax: Axes, fit: xr.Dataset, qubit_dict: dict):
        """
        Plots the 2D fitted data (vertical line) on a matplotlib axis.
        """
        ds_qubit = self.ds_raw.sel(qubit=qubit_dict["qubit"])
        amp_prefactor = ds_qubit["amp_prefactor"].values
        amp_mV = ds_qubit["full_amp"].values * MV_PER_V

        try:
            opt_amp_mV = (
                float(
                    ds_qubit["full_amp"]
                    .sel(amp_prefactor=fit.opt_amp_prefactor, method="nearest")
                    .values
                )
                * MV_PER_V
            )
        except (KeyError, ValueError) as e:
            logging.warning(
                f"Could not select optimal amplitude for qubit {qubit_dict['qubit']} using xarray, falling back to numpy. Error: {e}"
            )
            opt_amp_mV = float(
                amp_mV[np.argmin(np.abs(amp_prefactor - fit.opt_amp_prefactor))]
            )

        ax.axvline(
            x=opt_amp_mV,
            color=self.styling.fit_color,
            linestyle="-",
            linewidth=self.styling.matplotlib_fit_linewidth,
        )

    def _plot_matplotlib_subplot(self, ax: Axes, qubit_dict: dict, fit_data: Optional[xr.Dataset]):
        if self.is_1d:
            label = "Rotated I quadrature [mV]" if self.data_key == "I" else "Qubit state"
            (self.ds_raw.assign_coords(amp_mV=self.ds_raw.full_amp * MV_PER_V).loc[qubit_dict] * MV_PER_V)[self.data_key].plot(
                ax=ax, x="amp_mV", alpha=self.styling.matplotlib_raw_data_alpha
            )
            ax.set_ylabel(label)
            ax.set_xlabel("Pulse amplitude [mV]")
        else:
            (self.ds_raw.assign_coords(amp_mV=self.ds_raw.full_amp * MV_PER_V).loc[qubit_dict])[self.data_key].plot(
                ax=ax, add_colorbar=False, x="amp_mV", y="nb_of_pulses", robust=True
            )
            ax.set_ylabel("Number of pulses")
            ax.set_xlabel("Pulse amplitude [mV]")

        if fit_data is not None and hasattr(fit_data, "outcome") and getattr(fit_data.outcome, "values", None) == "successful":
            if self.is_1d:
                self._plot_fit_1d_matplotlib(ax, fit_data)
            else:
                self._plot_fit_2d_matplotlib(ax, fit_data, qubit_dict)

        self._add_matplotlib_twin_axis(ax, self.ds_raw, qubit_dict)

    def _plot_plotly_subplot(self, fig: go.Figure, qubit_id: str, row: int, col: int, fit_data: Optional[xr.Dataset]):
        if self.is_1d:
            self._plotly_plot_individual_data_with_fit_1D(fig, qubit_id, fit_data, row, col)
        else:
            self._plotly_plot_individual_data_with_fit_2D(fig, qubit_id, fit_data, row, col)

    def _plotly_plot_individual_data_with_fit_1D(
        self,
        fig: go.Figure,
        qubit: str,
        fit: Optional[xr.Dataset] = None,
        row: int = 1,
        col: int = 1,
    ):
        label = "Rotated I quadrature [mV]" if self.data_key == "I" else "Qubit state"
        ds_qubit = self.ds_raw.sel(qubit=qubit).isel(nb_of_pulses=0)

        amp_mV = ds_qubit["full_amp"].values * MV_PER_V
        amp_prefactor = ds_qubit["amp_prefactor"].values
        y_data = ds_qubit[self.data_key].values * MV_PER_V
        y_err_da = ds_qubit.get(f"{self.data_key}_std")
        y_err = y_err_da.values * MV_PER_V if y_err_da is not None else None

        # Plot raw data
        fig.add_trace(
            go.Scatter(
                x=amp_mV,
                y=y_data,
                error_y=dict(type="data", array=y_err, visible=True) if y_err is not None else None,
                name=f"Qubit {qubit} Raw",
                mode="lines+markers",
                line=dict(color=self.styling.raw_data_color),
                opacity=self.styling.matplotlib_raw_data_alpha,
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
                    line=dict(color=self.styling.fit_color, width=self.styling.matplotlib_fit_linewidth),
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(title_text="Pulse amplitude [mV]", row=row, col=col)
        fig.update_yaxes(title_text=label, row=row, col=col)
        add_plotly_top_axis(
            fig, row, col, self.grid.n_cols, amp_mV, amp_prefactor, "Amplitude prefactor"
        )

    def _plotly_plot_individual_data_with_fit_2D(
        self,
        fig: go.Figure,
        qubit: str,
        fit: Optional[xr.Dataset] = None,
        row: int = 1,
        col: int = 1,
    ):
        ds_qubit = self.ds_raw.sel(qubit=qubit)
        amp_mV = ds_qubit["full_amp"].values * MV_PER_V
        amp_prefactor = ds_qubit["amp_prefactor"].values
        nb_of_pulses = ds_qubit["nb_of_pulses"].values
        z_data = ds_qubit[self.data_key].values

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
            colorscale=self.styling.heatmap_colorscale,
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
                    line=dict(color=self.styling.fit_color, width=self.styling.matplotlib_fit_linewidth),
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(title_text="Pulse amplitude [mV]", row=row, col=col)
        fig.update_yaxes(title_text="Number of pulses", row=row, col=col)
        add_plotly_top_axis(
            fig, row, col, self.grid.n_cols, amp_mV, amp_prefactor, "Amplitude prefactor"
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
    return PowerRabiPlotter(ds_raw_prep, qubits, ds_fit_prep).create_matplotlib_plot()


def create_plotly_plot(
    ds_raw_prep: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit_prep: Optional[xr.Dataset] = None,
) -> go.Figure:
    """Return a Plotly figure for a Power-Rabi dataset (raw + optional fit)."""
    return PowerRabiPlotter(ds_raw_prep, qubits, ds_fit_prep).create_plotly_plot()


def create_plots(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit: Optional[xr.Dataset] = None,
) -> tuple[go.Figure, Figure]:
    """Convenience helper that returns (plotly_fig, matplotlib_fig)."""

    ds_raw_prep, ds_fit_prep = PowerRabiPreparator(ds_raw, ds_fit, qubits=qubits).prepare()
    plotter = PowerRabiPlotter(ds_raw_prep, qubits, ds_fit_prep)
    return plotter.plot()