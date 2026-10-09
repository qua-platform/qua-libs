import xarray as xr
from matplotlib.figure import Figure
from qualibration_libs.plotting import QubitGrid, grid_iter

from .traces import STATES


def plot_noop_pvalue_vs_point(ds: xr.Dataset, qubits, alpha: float) -> Figure:
    """Test-vs-no-operation z-test p-value vs scan point, one subplot per qubit with ground and excited
    overlaid. p above alpha means the probe field is indistinguishable from the no-operation reference,
    i.e. the resonator is depleted."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits], size=4.5)
    for ax, qubit in grid_iter(grid):
        qname = qubit["qubit"]
        for state in STATES:
            p = ds["p_value_noop"].sel(qubit=qname, state=state)
            ax.semilogy(p["point"], p.values, marker="o", markersize=3, label=state)
        ax.axhline(alpha, color="k", linestyle="--", label=f"alpha = {alpha}")
        ax.set_title(qname)
        ax.set_xlabel("scan point")
        ax.set_ylabel("p-value vs no-operation")

    handles, labels = ax.get_legend_handles_labels()
    grid.fig.suptitle("Resonator DRACHMA — test vs no-operation z-test vs scan point", fontsize=12)
    grid.fig.tight_layout(rect=(0, 0.04, 1, 1))
    grid.fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    return grid.fig
