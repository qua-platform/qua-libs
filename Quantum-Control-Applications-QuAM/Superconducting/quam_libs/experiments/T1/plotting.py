from typing import Dict, List
from matplotlib import pyplot as plt
import xarray as xr
from qualibrate import QualibrationNode
from quam_libs.components import Transmon
from quam_libs.lib.plot_utils import QubitGrid, grid_iter

def plot(ds: xr.Dataset, qubits: List[Transmon], fit_results: Dict, node: QualibrationNode) -> plt.Figure:
    
    tau = fit_results["tau"]
    tau_error = fit_results["tau_error"]
    fitted = fit_results["fitted"]
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        if node.parameters.use_state_discrimination:
            ds.sel(qubit=qubit["qubit"]).state.plot(ax=ax)
            ax.set_ylabel("State")
        else:
            ds.sel(qubit=qubit["qubit"]).I.plot(ax=ax)
            ax.set_ylabel("I (V)")
        ax.plot(ds.idle_time, fitted.loc[qubit], "r--")
        ax.set_title(qubit["qubit"])
        ax.set_xlabel("Idle_time (uS)")
        ax.text(
            0.1,
            0.9,
            f'T1 = {tau.sel(qubit = qubit["qubit"]).values:.1f} ± {tau_error.sel(qubit = qubit["qubit"]).values:.1f} µs',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.5),
        )
    grid.fig.suptitle("T1")
    plt.tight_layout()
    plt.show()
    
    return grid.fig