from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from plotly.graph_objs import Figure as PlotlyFigure
from qualang_tools.units import unit
from qualibration_libs.analysis import lorentzian_dip
from qualibration_libs.plotting import ResonatorSpectroscopyPreparator
from qualibration_libs.plotting.configs import PlotStyling
from qualibration_libs.plotting.plotters import \
    BaseResonatorSpectroscopyPlotter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)
styling = PlotStyling()

class ResonatorSpectroscopyAmplitudePlotter(BaseResonatorSpectroscopyPlotter):
    """Plotter for the amplitude part of a resonator spectroscopy experiment."""

    def get_plot_title(self) -> str:
        return "Resonator spectroscopy (amplitude + fit)"

    def get_yaxis_title(self) -> str:
        return r"$R=\sqrt{I^2 + Q^2}$ [mV]"

    def _plot_matplotlib_data(
        self, ax: Axes, ax2: Axes, qubit_dict: dict, fit_data: Optional[xr.Dataset]
    ):
        # Plot raw data on main axis
        (
            self.ds_raw.assign_coords(full_freq_GHz=self.ds_raw.full_freq / u.GHz)
            .loc[qubit_dict]
            .IQ_abs
            / u.mV
        ).plot(ax=ax, x="full_freq_GHz")

        # Plot raw data on secondary axis
        (
            self.ds_raw.assign_coords(detuning_MHz=self.ds_raw.detuning / u.MHz)
            .loc[qubit_dict]
            .IQ_abs
            / u.mV
        ).plot(ax=ax2, x="detuning_MHz")

        # Plot fit on secondary axis
        if fit_data and fit_data.outcome.values == "successful":
            fitted_data = lorentzian_dip(
                self.ds_raw.detuning,
                float(fit_data.amplitude.values),
                float(fit_data.position.values),
                float(fit_data.width.values) / 2,
                float(fit_data.base_line.mean().values),
            )
            ax2.plot(
                self.ds_raw.detuning / u.MHz,
                fitted_data / u.mV,
                color=styling.fit_color,
                linestyle="--",
            )

    def _plot_plotly_data(
        self,
        fig: go.Figure,
        qubit_id: str,
        row: int,
        col: int,
        fit_data: Optional[xr.Dataset],
    ):
        y_raw = (self.ds_raw.loc[{"qubit": qubit_id}].IQ_abs / u.mV).values
        x_raw = self.get_raw_x_values(qubit_id)
        detuning_vals = self.get_secondary_x_values(qubit_id)

        # Plot raw data
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
        # Plot fit
        if fit_data is not None and fit_data.outcome.values == "successful":
            fitted_data = lorentzian_dip(
                self.ds_raw.detuning.values,
                float(fit_data.amplitude.values),
                float(fit_data.position.values),
                float(fit_data.width.values) / 2,
                float(fit_data.base_line.mean().values),
            )
            fig.add_trace(
                go.Scatter(
                    x=x_raw,
                    y=fitted_data / u.mV,
                    name=f"Qubit {qubit_id} - Fit",
                    line=dict(
                        dash=styling.plotly_fit_linestyle, color=styling.fit_color
                    ),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )


class ResonatorSpectroscopyPhasePlotter(BaseResonatorSpectroscopyPlotter):
    """Plotter for the phase part of a resonator spectroscopy experiment."""

    def get_plot_title(self) -> str:
        return "Resonator spectroscopy (phase)"

    def get_yaxis_title(self) -> str:
        return "phase [rad]"

    def _plot_matplotlib_data(
        self, ax: Axes, ax2: Axes, qubit_dict: dict, fit_data: Optional[xr.Dataset]
    ):
        # Plot raw data on main axis
        self.ds_raw.assign_coords(full_freq_GHz=self.ds_raw.full_freq / u.GHz).loc[
            qubit_dict
        ].phase.plot(ax=ax, x="full_freq_GHz")

        # Plot raw data on secondary axis
        self.ds_raw.assign_coords(detuning_MHz=self.ds_raw.detuning / u.MHz).loc[
            qubit_dict
        ].phase.plot(ax=ax2, x="detuning_MHz")

    def _plot_plotly_data(
        self,
        fig: go.Figure,
        qubit_id: str,
        row: int,
        col: int,
        fit_data: Optional[xr.Dataset],
    ):
        ds_qubit = self.ds_raw.loc[{"qubit": qubit_id}]
        x_raw = self.get_raw_x_values(qubit_id)
        detuning_vals = self.get_secondary_x_values(qubit_id)

        fig.add_trace(
            go.Scatter(
                x=x_raw,
                y=ds_qubit.phase,
                name=f"Qubit {qubit_id}",
                showlegend=False,
                line=dict(color=styling.raw_data_color),
                customdata=np.stack([detuning_vals], axis=-1),
                hovertemplate="RF freq: %{x:.6f} GHz<br>Detuning: %{customdata[0]:.2f} MHz<br>Phase: %{y:.3f} rad<extra></extra>",
            ),
            row=row,
            col=col,
        )


# -----------------------------------------------------------------------------
# Public wrapper functions expected by the refactor
# -----------------------------------------------------------------------------


def create_matplotlib_plots(
    ds_raw_prep: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit_prep: Optional[xr.Dataset] = None,
) -> Dict[str, Figure]:
    """Return a dictionary of Matplotlib figures for a resonator spectroscopy dataset."""
    phase_plotter = ResonatorSpectroscopyPhasePlotter(ds_raw_prep, qubits, ds_fit_prep)
    amplitude_plotter = ResonatorSpectroscopyAmplitudePlotter(
        ds_raw_prep, qubits, ds_fit_prep
    )
    figs = {
        "phase": phase_plotter.create_matplotlib_plot(),
        "amplitude": amplitude_plotter.create_matplotlib_plot(),
    }
    return figs


def create_plotly_plots(
    ds_raw_prep: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit_prep: Optional[xr.Dataset] = None,
) -> Dict[str, PlotlyFigure]:
    """Return a dictionary of Plotly figures for a resonator spectroscopy dataset."""
    phase_plotter = ResonatorSpectroscopyPhasePlotter(ds_raw_prep, qubits, ds_fit_prep)
    amplitude_plotter = ResonatorSpectroscopyAmplitudePlotter(
        ds_raw_prep, qubits, ds_fit_prep
    )
    figs = {
        "phase": phase_plotter.create_plotly_plot(),
        "amplitude": amplitude_plotter.create_plotly_plot(),
    }
    return figs


def create_plots(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit: Optional[xr.Dataset] = None,
) -> Tuple[Dict[str, PlotlyFigure], Dict[str, Figure]]:
    """Convenience helper that returns (plotly_figs, matplotlib_figs)."""
    ds_raw_prep, ds_fit_prep = ResonatorSpectroscopyPreparator(
        ds_raw, ds_fit, qubits=qubits
    ).prepare()

    phase_plotter = ResonatorSpectroscopyPhasePlotter(ds_raw_prep, qubits, ds_fit_prep)
    amplitude_plotter = ResonatorSpectroscopyAmplitudePlotter(
        ds_raw_prep, qubits, ds_fit_prep
    )

    plotly_figs = {
        "phase": phase_plotter.create_plotly_plot(),
        "amplitude": amplitude_plotter.create_plotly_plot(),
    }
    matplotlib_figs = {
        "phase": phase_plotter.create_matplotlib_plot(),
        "amplitude": amplitude_plotter.create_matplotlib_plot(),
    }
    return plotly_figs, matplotlib_figs