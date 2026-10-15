import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from qualibration_libs.plotting import QubitGrid, grid_iter

from .pairs import excess_power_cost
from .traces import STATES

STATE_COLORS = {"ground": "tab:blue", "excited": "tab:red"}

# Fixed y-limit of the p-value axis (-log10 p), as in the 08g depletion validation: anything above is
# "far from depleted" and clipped, and a common axis makes qubits directly comparable.
NEGLOG10P_MAX = 100


def plot_noop_pvalue_vs_point(ds: xr.Dataset, qubits, alpha: float) -> Figure:
    """Test-vs-no-operation z-test -log10(p-value) vs scan point, one subplot per qubit with ground and excited
    overlaid, on a linear axis fixed to [0, NEGLOG10P_MAX] (larger values are clipped). A trace below the
    dashed -log10(alpha) line means the probe field is indistinguishable from the no-operation reference,
    i.e. the resonator is depleted."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits], size=4.5)
    for ax, qubit in grid_iter(grid):
        qname = qubit["qubit"]
        ax.set_ylim(0, NEGLOG10P_MAX)
        for state in STATES:
            trace = ds["neglog10_p_noop"].sel(qubit=qname, state=state)
            ax.plot(
                trace["point"],
                np.minimum(trace.values, NEGLOG10P_MAX),
                marker="o",
                markersize=3,
                color=STATE_COLORS[state],
                label=state,
            )
        ax.axhline(-np.log10(alpha), color="k", linestyle="--", linewidth=1, label=f"alpha = {alpha}")
        ax.set_title(qname)
        ax.set_xlabel("scan point")
        ax.set_ylabel("−log₁₀(p-val)")

    handles, labels = ax.get_legend_handles_labels()
    grid.fig.suptitle("Resonator DRACHMA — test vs no-operation z-test vs scan point", fontsize=12)
    grid.fig.tight_layout(rect=(0, 0.04, 1, 1))
    grid.fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    return grid.fig


def plot_combined_objective_vs_point(ds: xr.Dataset, qubits, fits: dict, parameter: str) -> Figure:
    """Noise-weighted excess-power cost vs scan point, one subplot per qubit: the two states' costs (thin,
    state colours) and their sum (black), the objective minimised when both states respond. The selected
    point of each state is a dashed line labelled with its value; a state that keeps its current value is
    labelled "kept". `parameter` is "zeta" or "kappa" (keys of `fits` and labels)."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits], size=4.5)
    for ax, qubit in grid_iter(grid):
        qname = qubit["qubit"]
        costs = {
            s: excess_power_cost(
                ds["power"].sel(qubit=qname, state=s).values, ds["power_err"].sel(qubit=qname, state=s).values
            )
            for s in STATES
        }
        points = ds["point"].values
        for s in STATES:
            ax.plot(points, costs[s], color=STATE_COLORS[s], marker="o", markersize=3, linewidth=1, label=f"{s} cost")
        ax.plot(points, costs["ground"] + costs["excited"], color="k", marker="o", markersize=3, label="sum")
        for s in STATES:
            fit = fits[qname]
            value = 1e-3 * fit[f"{parameter}_{s}_hz"]
            kept = "" if fit[f"changed_{s}"] else ", kept"
            ax.axvline(
                int(ds[f"best_point_{s}"].sel(qubit=qname)),
                color=STATE_COLORS[s],
                linestyle="--",
                alpha=0.6,
                label=f"{s} selected: {value:.1f} kHz{kept}",
            )
        ax.set_title(qname)
        ax.set_xlabel("scan point")
        ax.set_ylabel("excess cost [σ²-like]")
        ax.legend(fontsize=7)

    grid.fig.suptitle(f"Resonator DRACHMA — noise-weighted objective vs {parameter} scan point", fontsize=12)
    grid.fig.tight_layout(rect=(0, 0, 1, 1))
    return grid.fig
