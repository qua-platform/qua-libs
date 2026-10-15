import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from qualibration_libs.plotting import QubitGrid, grid_iter

from calibration_utils.readout_drachma_common import STATE_COLORS

__all__ = ["plot_power_vs_kappa"]


def plot_power_vs_kappa(ds: xr.Dataset, qubits, fits: dict) -> Figure:
    """Residual probe power (µV²) vs kappa (kHz), one subplot per qubit (QubitGrid), ground and excited
    overlaid, each against its own kappa axis (the scan is lockstep, so both are measured at every point).
    The chosen kappa of each state is a dashed line labelled in the legend ("kept" when the state's scan is
    flat and it keeps its current value; x marks a changed value on the scan edge); the no-operation (empty
    resonator) level is a grey line."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits], size=4.5)
    for ax, qubit in grid_iter(grid):
        qname = qubit["qubit"]
        for state in ds.state.values:
            kappa = 1e-3 * ds[f"kappa_{state}_hz"].sel(qubit=qname).values
            power = 1e12 * ds["power"].sel(qubit=qname, state=state).values
            err = 1e12 * ds["power_err"].sel(qubit=qname, state=state).values
            color = STATE_COLORS[str(state)]
            ax.errorbar(kappa, power, yerr=err, color=color, label=str(state), capsize=2, marker="o", markersize=3)
            best = 1e-3 * fits[qname][f"kappa_{state}_hz"]
            kept = "" if fits[qname][f"changed_{state}"] else ", kept"
            ax.axvline(best, color=color, linestyle="--", alpha=0.6, label=f"{state} selected: {best:.1f} kHz{kept}")
            if fits[qname][f"at_edge_{state}"]:
                ax.plot(best, power[np.argmin(np.abs(kappa - best))], marker="x", color="k", markersize=8)
        ax.axhline(
            1e12 * float(ds["power_noop"].sel(qubit=qname)),
            color="gray",
            linestyle="-.",
            linewidth=1,
            label="no operation",
        )
        ax.set_title(f"{qname}  (Δκ = {1e-3 * fits[qname]['kappa_difference_hz']:+.1f} kHz)")
        ax.set_xlabel("kappa [kHz]")
        ax.set_ylabel("probe power [µV²]")

    handles, labels = ax.get_legend_handles_labels()
    grid.fig.suptitle("Resonator DRACHMA — residual probe power vs kappa", fontsize=12)
    grid.fig.tight_layout(rect=(0, 0.06, 1, 1))
    grid.fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8)
    return grid.fig
