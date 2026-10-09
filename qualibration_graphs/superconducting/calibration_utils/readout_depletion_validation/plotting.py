import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from qualibration_libs.plotting import QubitGrid, grid_iter


def plot_residuals(ds: xr.Dataset, qubits, conditions: tuple, states: tuple, t_dep_ns: xr.DataArray = None) -> Figure:
    """Plot the residual photon amplitude |IQ| (µV) vs probe time, with ±std (shot noise)
    error bars, one subplot per qubit (physical layout via QubitGrid). Draws every
    (condition, state) pair except "no_operation", plus a single "no_operation" trace (at the
    first state) as the zero-photon baseline -- so the plot follows whatever conditions this
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
            std_trace = ds["IQ_abs_std"].sel(qubit=qname, condition=condition, state=state)
            ax.errorbar(
                trace.time_ns,
                1e6 * trace.values,
                yerr=1e6 * std_trace.values,
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
    grid.fig.suptitle("Resonator residual photon amplitude vs probe time", fontsize=12)
    grid.fig.tight_layout(rect=(0, 0.04, 1, 1))
    grid.fig.legend(handles, labels, loc="lower center", ncol=len(traces))
    return grid.fig


def _draw_pvalue_axis(ax, traces: list, alpha: float) -> None:
    """Shared body of plot_pvalue_grid/plot_ge_pvalue_grid: draws each (label, p_value_trace,
    t_dep_ns) trace on a log-scale p-value axis, floors values to 1e-300 (chi2.sf underflows
    to exact 0.0 for large chi2_stat -- a log axis silently drops those points, leaving gaps,
    without the floor), and marks each trace's detected depletion time (if not NaN) with a
    color-matched vertical line + rotated ns label. Also draws the p=alpha threshold line."""
    for label, trace, t_dep_ns in traces:
        plotted = np.clip(trace.values, 1e-300, None)
        (line,) = ax.plot(trace.time_ns, plotted, label=label, marker="o", markersize=3)

        if t_dep_ns is not None and not np.isnan(t_dep_ns):
            ax.axvline(t_dep_ns, color=line.get_color(), linestyle="--", linewidth=1)
            ax.text(
                t_dep_ns,
                ax.get_ylim()[1],
                f"  {t_dep_ns:.0f} ns",
                rotation=90,
                va="top",
                ha="left",
                fontsize=8,
                color=line.get_color(),
            )

    ax.axhline(alpha, color="k", linestyle="--", linewidth=1, label=f"α = {alpha}")
    ax.set_yscale("log")
    ax.set_xlabel("probe time [ns]")
    ax.set_ylabel("p-value")


def plot_pvalue_grid(
    ds: xr.Dataset,
    qubits,
    test_conditions: tuple,
    states: tuple,
    alpha: float,
    t_dep_stat_ns: xr.DataArray = None,
) -> Figure:
    """Plot the statistical depletion-time test's p-value (chi2, df=2, vs no_operation) on a
    log-scale y-axis, one subplot per qubit (physical layout via QubitGrid) -- ground and
    excited traces overlaid on the same axis, one line per (test_condition, state) pair."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits], size=4.5)

    for ax, qubit in grid_iter(grid):
        qname = qubit["qubit"]

        traces = []
        for condition in test_conditions:
            for state in states:
                trace = ds["p_value"].sel(qubit=qname, test_condition=condition, state=state)
                t_dep_ns = (
                    float(t_dep_stat_ns.sel(qubit=qname, test_condition=condition, state=state))
                    if t_dep_stat_ns is not None
                    else None
                )
                traces.append((f"{condition} ({state})", trace, t_dep_ns))

        _draw_pvalue_axis(ax, traces, alpha)
        ax.set_title(qname)

    handles, labels = ax.get_legend_handles_labels()
    grid.fig.suptitle("Resonator depletion-time p-value vs probe time", fontsize=12)
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
    """Plot the ground-vs-excited distinguishability test's p-value (chi2, df=2) on a
    log-scale y-axis, one subplot per qubit (physical layout via QubitGrid) -- a single trace
    per test_condition (no state dim, since this compares ground against excited directly)."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits], size=4.5)

    for ax, qubit in grid_iter(grid):
        qname = qubit["qubit"]

        traces = []
        for condition in test_conditions:
            trace = ds["p_value_ge"].sel(qubit=qname, test_condition=condition)
            t_dep_ns = (
                float(t_dep_ge_ns.sel(qubit=qname, test_condition=condition)) if t_dep_ge_ns is not None else None
            )
            traces.append((condition, trace, t_dep_ns))

        _draw_pvalue_axis(ax, traces, alpha)
        ax.set_title(qname)

    handles, labels = ax.get_legend_handles_labels()
    grid.fig.suptitle("Resonator ground-vs-excited depletion p-value vs probe time", fontsize=12)
    grid.fig.tight_layout(rect=(0, 0.04, 1, 1))
    grid.fig.legend(handles, labels, loc="lower center", ncol=len(handles))
    return grid.fig
