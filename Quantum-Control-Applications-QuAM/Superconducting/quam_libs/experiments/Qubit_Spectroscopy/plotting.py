from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from quam_libs.lib.plot_utils import QubitGrid, grid_iter


def plot_qubit_response(ds, qubits, result) -> Figure:

    # The resonant RF frequency of the qubits
    abs_freqs = dict(
        [
            (
                q.name,
                ds.freq_full.sel(freq=result.position.sel(qubit=q.name).values).sel(qubit=q.name).values,
            )
            for q in qubits
            if not np.isnan(result.sel(qubit=q.name).position.values)
        ]
    )
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    approx_peak = result.base_line + result.amplitude * (1 / (1 + ((ds.freq - result.position) / result.width) ** 2))
    for ax, qubit in grid_iter(grid):
        # Plot the line
        (ds.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit].I_rot * 1e3).plot(ax=ax, x="freq_GHz")
        # Identify the resonance peak
        if not np.isnan(result.sel(qubit=qubit["qubit"]).position.values):
            ax.plot(
                abs_freqs[qubit["qubit"]] / 1e9,
                ds.loc[qubit].sel(freq=result.loc[qubit].position.values, method="nearest").I_rot * 1e3,
                ".r",
            )
            # # Identify the width
            (approx_peak.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit] * 1e3).plot(
                ax=ax, x="freq_GHz", linewidth=0.5, linestyle="--"
            )
        ax.set_xlabel("Qubit frequency [GHz]")
        ax.set_ylabel(r"$R=\sqrt{I^2 + Q^2}$ [mV]")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle("Qubit spectroscopy")
    plt.tight_layout()
    plt.show()
    return grid.fig
