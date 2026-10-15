import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from qualibration_libs.plotting import QubitGrid, grid_iter


def plot_residuals(ds: xr.Dataset, qubits, conditions: tuple, states: tuple, t_dep_ns: xr.DataArray = None) -> Figure:
    """Plot the residual field amplitude |IQ| (µV) vs probe time, with ±SEM (standard error of
    the mean, single-shot std / sqrt(num_shots)) error bars, one subplot per qubit (physical layout via QubitGrid). Draws every
    (condition, state) pair except "no_operation", plus a single "no_operation" trace (at the
    first state) as the empty-resonator baseline -- so the plot follows whatever conditions this
    run actually acquired, rather than a fixed hard-coded trace list. If t_dep_ns (per qubit) is
    given, the chosen depletion time is marked with a vertical line."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits], size=4.5)

    traces = [
        (f"{condition} ({state})", condition, state)
        for condition in conditions
        if condition != "no_operation"
        for state in states
    ]
    if "no_operation" in conditions:
        traces.append((f"no-op ({states[0]})", "no_operation", states[0]))

    for ax, qubit in grid_iter(grid):
        qname = qubit["qubit"]

        for label, condition, state in traces:
            trace = ds["IQ_abs"].sel(qubit=qname, condition=condition, state=state)
            sem_trace = ds["IQ_abs_sem"].sel(qubit=qname, condition=condition, state=state)
            ax.errorbar(
                trace.time_ns,
                1e6 * trace.values,
                yerr=1e6 * sem_trace.values,
                label=label,
                capsize=2,
                marker="o",
                markersize=3,
            )

        if t_dep_ns is not None:
            t_ns = float(t_dep_ns.sel(qubit=qname))
            if not np.isnan(t_ns):
                ax.axvline(t_ns, color="k", linestyle="--", linewidth=1)
                ax.text(t_ns, ax.get_ylim()[1], f"  {t_ns:.0f} ns", rotation=90, va="top", ha="left", fontsize=8)

        ax.set_title(qname)
        ax.set_xlabel("probe time [ns]")
        ax.set_ylabel("|IQ| [µV]")

    handles, labels = ax.get_legend_handles_labels()
    grid.fig.suptitle("Resonator residual |IQ| vs probe time", fontsize=12)
    grid.fig.tight_layout(rect=(0, 0.04, 1, 1))
    grid.fig.legend(handles, labels, loc="lower center", ncol=len(traces))
    return grid.fig


# Fixed y-limit shared by every p-value axis (-log10 p). Anything above is "far from depleted"
# and clipped -- how far no longer matters, and a common axis makes qubits directly comparable.
NEGLOG10P_MAX = 100


def _draw_pvalue_axis(ax, traces: list, alpha: float) -> None:
    """Shared body of plot_pvalue_grid/plot_ge_pvalue_grid: draws each (label, neglog10_p_trace,
    t_dep_ns) trace as -log10(p-value) on a linear axis fixed to [0, NEGLOG10P_MAX] (values
    above are clipped, +inf included -- chi2.logsf returns -inf for a chi2 statistic beyond
    ~1400), and marks each trace's detected depletion time (if not NaN) with a color-matched
    vertical line + rotated ns label. Also draws the p=alpha threshold line, at -log10(alpha)."""
    ax.set_ylim(0, NEGLOG10P_MAX)
    for label, trace, t_dep_ns in traces:
        plotted = np.minimum(trace.values, NEGLOG10P_MAX)
        (line,) = ax.plot(trace.time_ns, plotted, label=label, marker="o", markersize=3)

        if t_dep_ns is not None and not np.isnan(t_dep_ns):
            ax.axvline(t_dep_ns, color=line.get_color(), linestyle="--", linewidth=1)
            ax.text(
                t_dep_ns,
                NEGLOG10P_MAX,
                f"  {t_dep_ns:.0f} ns",
                rotation=90,
                va="top",
                ha="left",
                fontsize=8,
                color=line.get_color(),
            )

    ax.axhline(-np.log10(alpha), color="k", linestyle="--", linewidth=1, label=f"α = {alpha}")
    ax.set_xlabel("probe time [ns]")
    ax.set_ylabel("−log₁₀(p-value)")


def plot_pvalue_grid(
    ds: xr.Dataset,
    qubits,
    test_conditions: tuple,
    states: tuple,
    alpha: float,
    t_dep_stat_ns: xr.DataArray = None,
) -> Figure:
    """Plot the statistical depletion-time test's -log10(p-value) (chi2, df=2, vs no_operation)
    on a y-axis fixed to [0, NEGLOG10P_MAX], one subplot per qubit (physical layout via QubitGrid) -- ground and
    excited traces overlaid on the same axis, one line per (test_condition, state) pair."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits], size=4.5)

    for ax, qubit in grid_iter(grid):
        qname = qubit["qubit"]

        traces = []
        for condition in test_conditions:
            for state in states:
                trace = ds["neglog10_p"].sel(qubit=qname, test_condition=condition, state=state)
                t_dep_ns = (
                    float(t_dep_stat_ns.sel(qubit=qname, test_condition=condition, state=state))
                    if t_dep_stat_ns is not None
                    else None
                )
                traces.append((f"{condition} ({state})", trace, t_dep_ns))

        _draw_pvalue_axis(ax, traces, alpha)
        ax.set_title(qname)

    handles, labels = ax.get_legend_handles_labels()
    grid.fig.suptitle("Resonator depletion-time −log₁₀(p-value) vs probe time", fontsize=12)
    grid.fig.tight_layout(rect=(0, 0.04, 1, 1))
    grid.fig.legend(handles, labels, loc="lower center", ncol=len(handles))
    return grid.fig


def plot_ge_pvalue_grid(
    ds: xr.Dataset,
    qubits,
    test_conditions: tuple,
    alpha: float,
    t_dep_ge_ns: xr.DataArray = None,
) -> Figure:
    """Plot the ground-vs-excited distinguishability test's -log10(p-value) (chi2, df=2) on a
    y-axis fixed to [0, NEGLOG10P_MAX], one subplot per qubit (physical layout via QubitGrid) -- a single trace
    per test_condition (no state dim, since this compares ground against excited directly)."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits], size=4.5)

    for ax, qubit in grid_iter(grid):
        qname = qubit["qubit"]

        traces = []
        for condition in test_conditions:
            trace = ds["neglog10_p_ge"].sel(qubit=qname, test_condition=condition)
            t_dep_ns = (
                float(t_dep_ge_ns.sel(qubit=qname, test_condition=condition)) if t_dep_ge_ns is not None else None
            )
            traces.append((condition, trace, t_dep_ns))

        _draw_pvalue_axis(ax, traces, alpha)
        ax.set_title(qname)

    handles, labels = ax.get_legend_handles_labels()
    grid.fig.suptitle("Resonator ground-vs-excited depletion −log₁₀(p-value) vs probe time", fontsize=12)
    grid.fig.tight_layout(rect=(0, 0.04, 1, 1))
    grid.fig.legend(handles, labels, loc="lower center", ncol=len(handles))
    return grid.fig
