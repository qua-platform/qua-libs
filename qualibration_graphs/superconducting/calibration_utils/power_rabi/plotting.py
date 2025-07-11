import logging
from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from plotly.graph_objs import Figure as PlotlyFigure
from qualang_tools.units import unit
from qualibration_libs.plotting import PowerRabiPreparator
from qualibration_libs.plotting.plotters import (PowerRabi1DPlotter,
                                                 PowerRabi2DPlotter)
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)

def _get_data_key(ds: xr.Dataset) -> str:
    if hasattr(ds, "I"):
        return "I"
    if hasattr(ds, "state"):
        return "state"
    raise RuntimeError("The dataset must contain either 'I' or 'state'.")

class PowerRabiPlotter:
    """
    A plotter for Power Rabi experiments that delegates to a 1D or 2D
    specialized plotter based on the input data shape.
    """

    def __init__(
        self,
        ds_raw: xr.Dataset,
        qubits: List[AnyTransmon],
        ds_fit: Optional[xr.Dataset] = None,
    ):
        data_key = _get_data_key(ds_raw)
        is_1d = len(ds_raw.nb_of_pulses) == 1

        if is_1d:
            self.delegate = PowerRabi1DPlotter(ds_raw, qubits, ds_fit, data_key)
        else:
            self.delegate = PowerRabi2DPlotter(ds_raw, qubits, ds_fit, data_key)

    def create_matplotlib_plot(self) -> Figure:
        return self.delegate.create_matplotlib_plot()

    def create_plotly_plot(self) -> go.Figure:
        return self.delegate.create_plotly_plot()

    def plot(self) -> Tuple[go.Figure, Figure]:
        return self.delegate.plot()


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
) -> PlotlyFigure:
    """Return a Plotly figure for a Power-Rabi dataset (raw + optional fit)."""
    return PowerRabiPlotter(ds_raw_prep, qubits, ds_fit_prep).create_plotly_plot()


def create_plots(
    ds_raw: xr.Dataset,
    qubits: List[AnyTransmon],
    ds_fit: Optional[xr.Dataset] = None,
) -> tuple[go.Figure, Figure]:
    """Convenience helper that returns (plotly_fig, matplotlib_fig)."""
    ds_raw_prep, ds_fit_prep = PowerRabiPreparator(
        ds_raw, ds_fit, qubits=qubits
    ).prepare()
    return PowerRabiPlotter(ds_raw_prep, qubits, ds_fit_prep).plot()