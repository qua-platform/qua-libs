"""Plotting for the ±δ Ramsey parity-difference analysis.

Produces a three-panel figure per qubit:

1. **+δ trace** — parity-difference vs idle time at positive detuning,
   with the damped-cosine fit overlaid and fitted frequency annotated.
2. **−δ trace** — same for the negative detuning.
3. **Triangulation summary** — textual summary of the triangulated
   residual offset, T₂*, and both fitted frequencies.
"""

from __future__ import annotations

from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _get_qubit_names_from_ds(ds: xr.Dataset) -> List[str]:
    """Extract qubit names from ``pdiff_<name>`` data-variable keys."""
    pdiff_vars = [
        v for v in ds.data_vars
        if v.startswith("pdiff_") and not v.endswith("_fit")
    ]
    return [v.replace("pdiff_", "") for v in sorted(pdiff_vars)]


def _plot_trace_ax(
    ax: "plt.Axes",
    tau_ns: np.ndarray,
    qubit_name: str,
    trace_fit: dict | None,
    label: str,
    color: str = "b",
    fit_color: str = "r",
) -> None:
    """Plot one detuning trace with its damped-cosine fit."""
    if trace_fit is None:
        ax.text(0.5, 0.5, "No fit data", transform=ax.transAxes, ha="center")
        ax.set_title(f"{qubit_name} — {label}")
        return

    pdiff = trace_fit["pdiff"]
    fitted = trace_fit.get("fitted_curve")
    t_shifted = trace_fit["tau_shifted"]
    t_plot = t_shifted + tau_ns[0]

    ax.plot(t_plot, pdiff, f"{color}-", lw=0.8, alpha=0.7)
    ax.scatter(t_plot, pdiff, c=color, s=8, alpha=0.5, zorder=3, label="Data")

    if fitted is not None:
        ax.plot(t_plot, fitted, f"{fit_color}-", lw=1.5, alpha=0.9, label="Damped cosine")

    ax.set_xlabel("Idle time (ns)")
    ax.set_ylabel("Parity difference")
    ax.set_ylim(-0.05, 1.05)

    f_hz = trace_fit.get("ramsey_freq", np.nan)
    title = f"{qubit_name} — {label}"
    if np.isfinite(f_hz):
        title += f" (f={f_hz * 1e-6:.3f} MHz)"
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7)


def _plot_summary_ax(
    ax: "plt.Axes",
    qubit_name: str,
    fit_result: dict | None = None,
) -> None:
    """Show triangulation summary as text annotations."""
    ax.set_axis_off()
    ax.set_title(f"{qubit_name} — Triangulation")

    if fit_result is None or not fit_result.get("success"):
        ax.text(0.5, 0.5, "Fit failed", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="red")
        return

    freq_off = fit_result.get("freq_offset", 0) * 1e-6
    t2 = fit_result.get("t2_star", np.nan)
    gamma = fit_result.get("decay_rate", np.nan)
    f_plus = fit_result.get("freq_plus", np.nan) * 1e-6
    f_minus = fit_result.get("freq_minus", np.nan) * 1e-6

    lines = [
        r"$f_+ = $" + f"{f_plus:.3f} MHz",
        r"$f_- = $" + f"{f_minus:.3f} MHz",
        "",
        r"$\Delta = (f_- - f_+) / 2$",
        r"$\Delta = $" + f"{freq_off:.3f} MHz",
        "",
        f"T2* = {t2:.0f} ns" if np.isfinite(t2) else "T2* = N/A",
        f"γ = {gamma:.5f} 1/ns" if np.isfinite(gamma) else "γ = N/A",
    ]
    text = "\n".join(lines)
    ax.text(
        0.5, 0.5, text,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=11,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    ds_fit: xr.Dataset | None,
    qubits: List[Any],
    fit_results: dict,
) -> "plt.Figure":
    """Plot ±δ Ramsey analysis for each qubit.

    Layout (per qubit row):
    * Column 1 — +δ trace with damped-cosine fit.
    * Column 2 — −δ trace with damped-cosine fit.
    * Column 3 — Triangulation summary (Δ, T₂*, frequencies).
    """
    qubit_names = _get_qubit_names_from_ds(ds)
    if not qubit_names:
        fig, _ = plt.subplots(figsize=(6, 4))
        return fig

    n = len(qubit_names)
    ncol = 3
    fig, axes = plt.subplots(n, ncol, figsize=(6 * ncol, 4 * n), squeeze=False)

    tau_ns = np.asarray(ds.tau.values, dtype=float)
    detuning_hz = np.asarray(ds.detuning.values, dtype=float)

    for i, qname in enumerate(qubit_names):
        fr = fit_results.get(qname, {})
        diag = fr.get("_diag", {})
        fit_plus = diag.get("fit_plus")
        fit_minus = diag.get("fit_minus")

        det_plus_mhz = detuning_hz[0] * 1e-6 if len(detuning_hz) > 0 else 0
        det_minus_mhz = detuning_hz[1] * 1e-6 if len(detuning_hz) > 1 else 0

        _plot_trace_ax(
            axes[i, 0], tau_ns, qname, fit_plus,
            label=f"+δ ({det_plus_mhz:+.1f} MHz)",
            color="b", fit_color="r",
        )
        _plot_trace_ax(
            axes[i, 1], tau_ns, qname, fit_minus,
            label=f"−δ ({det_minus_mhz:+.1f} MHz)",
            color="C2", fit_color="C3",
        )
        _plot_summary_ax(axes[i, 2], qname, fr)

    fig.suptitle("Ramsey ±δ triangulation (parity diff)")
    fig.tight_layout()
    return fig
