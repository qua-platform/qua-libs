import xarray as xr
from matplotlib.figure import Figure
from qualibration_libs.plotting import QubitGrid, grid_iter

from calibration_utils.readout_drachma_common import STATE_COLORS


def plot_power_vs_zeta(ds: xr.Dataset, qubits, fits: dict) -> Figure:
    """Residual probe power (µV²) vs zeta (kHz), one subplot per qubit (QubitGrid), ground and excited
    overlaid, each against its own zeta axis (the scan is lockstep, so both are measured at every point).
    The chosen zeta of each state is a dashed line labelled in the legend ("kept" when the state's scan is
    flat and it keeps its current value); the no-operation (empty resonator) level is a grey line."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits], size=4.5)
    for ax, qubit in grid_iter(grid):
        qname = qubit["qubit"]
        for state in ds.state.values:
            zeta = 1e-3 * ds[f"zeta_{state}_hz"].sel(qubit=qname).values
            power = 1e12 * ds["power"].sel(qubit=qname, state=state).values
            err = 1e12 * ds["power_err"].sel(qubit=qname, state=state).values
            color = STATE_COLORS[str(state)]
            ax.errorbar(zeta, power, yerr=err, color=color, label=str(state), capsize=2, marker="o", markersize=3)
            best = 1e-3 * fits[qname][f"zeta_{state}_hz"]
            kept = "" if fits[qname][f"changed_{state}"] else ", kept"
            ax.axvline(best, color=color, linestyle="--", alpha=0.6, label=f"{state} selected: {best:.1f} kHz{kept}")
        ax.axhline(
            1e12 * float(ds["power_noop"].sel(qubit=qname)),
            color="gray",
            linestyle="-.",
            linewidth=1,
            label="no operation",
        )
        ax.set_title(f"{qname}  (Δζ = {1e-3 * fits[qname]['zeta_difference_hz']:+.1f} kHz)")
        ax.set_xlabel("zeta [kHz]")
        ax.set_ylabel("probe power [µV²]")

    handles, labels = ax.get_legend_handles_labels()
    grid.fig.suptitle("Resonator DRACHMA — residual probe power vs zeta", fontsize=12)
    grid.fig.tight_layout(rect=(0, 0.06, 1, 1))
    grid.fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8)
    return grid.fig
