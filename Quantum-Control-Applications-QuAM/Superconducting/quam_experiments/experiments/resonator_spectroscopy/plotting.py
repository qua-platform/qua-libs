from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from quam_libs.plot_utils import QubitGrid, grid_iter
from quam_experiments.experiments.resonator_spectroscopy.fitting import lorentzian
from qualang_tools.units import unit
from typing import List
import xarray as xr
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from quam_experiments.experiments.ramsey.parameters import Parameters

u = unit(coerce_to_integer=True)


# todo: docstrings!!
def plot_raw_amplitude(ds, qubits) -> Figure:
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax1, qubit in grid_iter(grid):
        # Create a first x-axis for full_freq_GHz
        (ds.assign_coords(full_freq_GHz=ds.full_freq / u.GHz).loc[qubit].R / u.mV).plot(ax=ax1, x="full_freq_GHz")
        ax1.set_xlabel("RF frequency [GHz]")
        ax1.set_ylabel(r"$R=\sqrt{I^2 + Q^2}$ [mV]")
        # Create a second x-axis for detuning_MHz
        ax2 = ax1.twiny()
        (ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[qubit].R / u.mV).plot(ax=ax2, x="detuning_MHz")
        ax2.set_xlabel("Detuning [MHz]")
    grid.fig.suptitle("Resonator spectroscopy (raw data: amplitude)")
    plt.tight_layout()

    return grid.fig


def plot_raw_phase(ds, qubits) -> Figure:
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax1, qubit in grid_iter(grid):
        # Create a first x-axis for full_freq_GHz
        (ds.assign_coords(full_freq_GHz=ds.full_freq / u.GHz).loc[qubit].phase / u.mV).plot(ax=ax1, x="full_freq_GHz")
        ax1.set_xlabel("RF frequency [GHz]")
        ax1.set_ylabel("phase [rad]")
        # Create a second x-axis for detuning_MHz
        ax2 = ax1.twiny()
        (ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[qubit].phase / u.mV).plot(ax=ax2, x="detuning_MHz")
        ax2.set_xlabel("Detuning [MHz]")
    grid.fig.suptitle("Resonator spectroscopy (raw data: phase)")
    plt.tight_layout()

    return grid.fig


def plot_res_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], node_parameters: Parameters, fits: xr.Dataset):
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

    Example:
    --------
        >>> fig = plot_res_specs_data_with_fit(ds, qubits, node_parameters, fits)
        >>> fig.show()
    """
    grid = QubitGrid(ds.to_dataarray(), [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_res_specs_data_with_fit(ax, ds, qubit, node_parameters, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Resonator spectroscopy (fit)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_res_specs_data_with_fit(ax, ds, qubit, node_parameters, fit=None):
    """Plot individual qubit data on a given axis."""
    if fit:
        fitted_data = lorentzian(
            ds.detuning,
            fit.sel(fit_vals="a"),
            fit.sel(fit_vals="center_freq"),
            fit.sel(fit_vals="width"),
            fit.sel(fit_vals="offset"),
        )
    else:
        fitted_data = None

    _plot_transmission_amplitude(ax, ds, qubit, fitted_data)
    ax.set_ylabel("Trans. amp. I [mV]")


def _plot_transmission_amplitude(ax1, ds, qubit, fitted=None):
    """Plot transmission amplitude for a qubit."""
    # Create a first x-axis for full_freq_GHz
    (ds.assign_coords(full_freq_GHz=ds.full_freq / u.GHz).loc[qubit].R / u.mV).plot(ax=ax1, x="full_freq_GHz")
    ax1.set_xlabel("RF frequency [GHz]")
    ax1.set_ylabel(r"$R=\sqrt{I^2 + Q^2}$ [mV]")
    # Create a second x-axis for detuning_MHz
    ax2 = ax1.twiny()
    (ds.assign_coords(detuning_MHz=ds.detuning / u.MHz).loc[qubit].R / u.mV).plot(ax=ax2, x="detuning_MHz")
    ax2.set_xlabel("Detuning [MHz]")
    if fitted is not None:
        ax2.plot(ds.detuning / u.MHz, 1e3 * fitted.ds_fit, "r--")
