import xarray as xr
from matplotlib.figure import Figure
from qualibration_libs.plotting import QubitGrid, grid_iter


def plot_ge_pvalue_vs_point(ds: xr.Dataset, qubits, alpha: float) -> Figure:
    """Ground-vs-excited z-test p-value vs scan point, one subplot per qubit. p above alpha means the
    two states leave indistinguishable probe fields, i.e. the resonator is depleted."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits], size=4.5)
    for ax, qubit in grid_iter(grid):
        qname = qubit["qubit"]
        p = ds["p_value_ge"].sel(qubit=qname)
        ax.semilogy(p["point"], p.values, marker="o", markersize=3)
        ax.axhline(alpha, color="k", linestyle="--", label=f"alpha = {alpha}")
        ax.set_title(qname)
        ax.set_xlabel("scan point")
        ax.set_ylabel("g-vs-e p-value")

    handles, labels = ax.get_legend_handles_labels()
    grid.fig.suptitle("Resonator DRACHMA — ground-vs-excited z-test vs scan point", fontsize=12)
    grid.fig.tight_layout(rect=(0, 0.04, 1, 1))
    grid.fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    return grid.fig
