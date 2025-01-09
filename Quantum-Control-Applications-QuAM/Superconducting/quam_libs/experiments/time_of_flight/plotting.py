from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from quam_libs.lib.plot_utils import QubitGrid, grid_iter


def plot_adc_single_runs(ds, qubits) -> Figure:
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        ds.loc[qubit].adc_single_runI.plot(ax=ax, x="time", label="I", color="b")
        ds.loc[qubit].adc_single_runQ.plot(ax=ax, x="time", label="Q", color="r")
        ax.axvline(ds.loc[qubit].delays, color="k", linestyle="--", label="TOF")
        ax.axhline(ds.loc[qubit].offsets_I, color="b", linestyle="--")
        ax.axhline(ds.loc[qubit].offsets_Q, color="r", linestyle="--")
        ax.fill_between(range(ds.sizes["time"]), -0.5, 0.5, color="grey", alpha=0.2, label="ADC Range")
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Readout amplitude [mV]")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle("Single run")
    plt.tight_layout()
    plt.legend(loc="upper right", ncols=4, bbox_to_anchor=(0.5, 1.35))

    return grid.fig


def plot_adc_averaged_runs(ds, qubits) -> Figure:
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        ds.loc[qubit].adcI.plot(ax=ax, x="time", label="I", color="b")
        ds.loc[qubit].adcQ.plot(ax=ax, x="time", label="Q", color="r")
        ax.axvline(ds.loc[qubit].delays, color="k", linestyle="--", label="TOF")
        ax.axhline(ds.loc[qubit].offsets_I, color="b", linestyle="--")
        ax.axhline(ds.loc[qubit].offsets_Q, color="r", linestyle="--")
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Readout amplitude [mV]")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle("Averaged run")
    plt.tight_layout()
    plt.legend(loc="upper right", ncols=4, bbox_to_anchor=(0.5, 1.35))

    return grid.fig