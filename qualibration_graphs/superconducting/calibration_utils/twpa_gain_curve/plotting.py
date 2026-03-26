from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from quam_builder.architecture.superconducting.components.twpa import TWPA
from quam_config.my_quam import TwpaGrid
from qualibration_libs.plotting import grid_iter
from qualibrate.core import QualibrationNode

u = unit(coerce_to_integer=True)


def plot_gain(ds: xr.Dataset, twpas: List[TWPA], node: QualibrationNode):
    """
    Plots the resonator spectroscopy amplitude IQ_abs with fitted curves for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    twpas : list of TWPAI
        A list of twpas to plot.
    node : QualibrationNode
        The QUAlibrate node.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains the raw data and the fitted curve.
    """
    grid = TwpaGrid(ds, [t.grid_location for t in twpas])
    for ax, twpa in grid_iter(grid):
        plot_individual_gain(ax, ds, twpa)
        add_readout_frequencies(ax, [node.machine.qubits[q] for q in node.machine.twpas[twpa["twpa"]].qubits])

    # grid.fig.suptitle("Resonator spectroscopy (amplitude + fit)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_gain(ax: Axes, ds: xr.Dataset, twpa: dict[str, str]):
    """
    Plots individual qubit data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    twpa : dict[str, str]
        mapping to the TWPA to plot.

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """
    # Create a first x-axis for full_freq_GHz
    ds.assign_coords(full_freq_GHz=ds.full_freq).loc[twpa].gain.plot(ax=ax, x="full_freq_GHz")
    ax.set_xlabel("RF frequency [GHz]")
    # Create a second x-axis for detuning_MHz
    ax2 = ax.twiny()
    ds.assign_coords(detuning_MHz=ds.detuning / 1e6).loc[twpa].gain.plot(ax=ax2, x="detuning_MHz")
    ax2.set_xlabel("Detuning [MHz]")

def add_readout_frequencies(ax: Axes, qubits: List[AnyTransmon]):
    """
    Plots individual qubit readout lines.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    qubits : List[AnyTransmon]
        List of qubits connected to the given TWPA line.
    """
    _, _, ymin, ymax = ax.axis()
    for q in qubits:
        ax.vlines(x=q.resonator.f_01 / 1e9, ymin=ymin, ymax=ymax, linestyle="--", color="k")
    ax.set_xlabel("RF frequency [GHz]")
