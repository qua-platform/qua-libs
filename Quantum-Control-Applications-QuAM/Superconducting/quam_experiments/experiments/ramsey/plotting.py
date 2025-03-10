import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from typing import List
from quam_libs.plot_utils import QubitGrid, grid_iter


def plot_ramseys_data_with_fit(ds: xr.Dataset, qubits: List, params, fits: dict) -> plt.Figure:
    """
    Plot the Ramsey raw data and overlay the fitted curve.

    Parameters:
        ds : xarray.Dataset
            The dataset containing Ramsey measurements.
        qubits : list
            List of qubit objects.
        params :
            Node parameters.
        fits : dict
            Dictionary mapping qubit names to RamseyFit objects.

    Returns:
        A matplotlib Figure with the plotted data.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        qname = qubit["qubit"]
        # Plot raw data (assumed available in variables I and Q).
        ds.sel(qubit=qname).I.plot(ax=ax, label="I")
        ds.sel(qubit=qname).Q.plot(ax=ax, label="Q")
        # Generate a dummy fit curve for demonstration.
        idle_time = ds.idle_time.values  # idle_time in ns
        fit_obj = fits[qname]
        # Dummy fit: oscillatory decaying function.
        fit_curve = np.exp(-idle_time / fit_obj.decay) * np.cos(2 * np.pi * fit_obj.freq_offset * idle_time * 1e-9)
        ax.plot(idle_time, fit_curve, "r--", label="Fit")
        ax.set_title(qname)
        ax.set_xlabel("Idle time (ns)")
        ax.legend()
    grid.fig.suptitle("Ramsey with Virtual Z Rotations")
    grid.fig.tight_layout()
    return grid.fig
