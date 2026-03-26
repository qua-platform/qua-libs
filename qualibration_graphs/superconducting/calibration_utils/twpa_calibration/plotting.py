from typing import List
import xarray as xr
from matplotlib.axes import Axes
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from qualibration_libs.plotting import QubitGrid, grid_iter


def plot_gain(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset, node):
    """
    Plots T1 data with fit results for multiple qubits.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    qubits : List[AnyTransmon]
        List of qubits involved in the sequence.
    node_parameters : Parameters
        Parameters related to the node.
    fits : xr.Dataset
        Dataset containing the fit results for the T1 data.

    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plots.
    """
    grid = QubitGrid(ds, [node.machine.qubits[q].grid_location for q in qubits])

    for ax, qubit in grid_iter(grid):
        plot_individual_gain(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("TWPA Gain")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_gain(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """Plot individual qubit data on a given axis."""

    fit.gain.plot(ax=ax)
    ax.set_title(qubit["qubit"])


def plot_snr(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset, node):
    """
    Plots T1 data with fit results for multiple qubits.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    qubits : List[AnyTransmon]
        List of qubits involved in the sequence.
    node_parameters : Parameters
        Parameters related to the node.
    fits : xr.Dataset
        Dataset containing the fit results for the T1 data.

    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plots.
    """
    grid = QubitGrid(ds, [node.machine.qubits[q].grid_location for q in qubits])

    for ax, qubit in grid_iter(grid):
        plot_individual_snr(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("TWPA SNR")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_snr(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """Plot individual qubit data on a given axis."""

    fit.snr.plot(ax=ax)
    ax.set_title(qubit["qubit"])


def plot_iqblobs(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset, node):
    """
    Plots T1 data with fit results for multiple qubits.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    qubits : List[AnyTransmon]
        List of qubits involved in the sequence.
    node_parameters : Parameters
        Parameters related to the node.
    fits : xr.Dataset
        Dataset containing the fit results for the T1 data.

    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plots.
    """
    grid = QubitGrid(ds, [node.machine.qubits[q].grid_location for q in qubits])

    for ax, qubit in grid_iter(grid):
        plot_individual_iqblobs(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Best IQ blobs")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_iqblobs(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """Plot individual qubit data on a given axis."""

    detuning = fit.coords_best["detuning_p"]
    power = fit.coords_best["twpa_power_p"]
    ds_ = fit.sel(detuning_p=detuning, twpa_power_p=power, method="nearest")
    ax.scatter(ds_.Ion, ds_.Qon, label="ON")
    ax.scatter(ds_.Ioff, ds_.Qoff, label="OFF")
    ax.set_xlabel("I [V]")
    ax.set_ylabel("Q [V]")
    ax.set_title(f"detuning, power = {detuning * 1e-6:2f}MHz, {power}dBm")
    ax.legend()
