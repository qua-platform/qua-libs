from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from quam_libs.lib.plot_utils import QubitGrid, grid_iter


def plot_raw_amplitude(ds, qubits) -> Figure:
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        (ds.assign_coords(freq_MHz=ds.freq / 1e6).loc[qubit].IQ_abs * 1e3).plot(ax=ax, x="freq_MHz")
        ax.set_xlabel("Resonator detuning [MHz]")
        ax.set_ylabel("Trans. amp. [mV]")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle("Resonator spectroscopy (raw data)")
    plt.tight_layout()
    
    return grid.fig

def plot_raw_phase(ds, qubits) -> Figure:
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        (ds.assign_coords(freq_MHz=ds.freq / 1e6).loc[qubit].phase * 1e3).plot(ax=ax, x="freq_MHz")
        ax.set_xlabel("Resonator detuning [MHz]")
        ax.set_ylabel("Trans. phase [mrad]")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle("Resonator spectroscopy (raw data)")
    plt.tight_layout()
    return grid.fig

def plot_fit_amplitude(ds, qubits) -> Figure:
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        (ds.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit].IQ_abs * 1e3).plot(ax=ax, x="freq_GHz")
        np.abs(ds.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit].fit_evals * 1e3).plot(ax=ax, x="freq_GHz")
        ax.set_xlabel("Resonator freq [GHz]")
        ax.set_ylabel("Trans. amp. [mV]")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle("Resonator spectroscopy (fit)")
    plt.tight_layout()
    return grid.fig

