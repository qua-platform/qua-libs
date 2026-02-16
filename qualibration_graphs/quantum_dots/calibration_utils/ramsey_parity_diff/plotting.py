"""Plotting for the 1-D Ramsey parity-difference analysis.

Produces a two-panel figure per qubit:

1. **Ramsey time trace** — parity-difference vs idle time with the
   damped-cosine fit overlaid and key fit parameters annotated.
2. **Residuals** — difference between data and fit, useful for
   diagnosing systematic deviations from the simple exponential-decay
   model.
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


def _plot_time_trace_ax(
    ax: "plt.Axes",
    tau_ns: np.ndarray,
    qubit_name: str,
    fit_result: dict | None = None,
) -> None:
    """Plot parity-difference vs idle time with damped-cosine fit."""
    diag = (fit_result or {}).get("_diag")
    if diag is None:
        ax.text(0.5, 0.5, "No diagnostics", transform=ax.transAxes, ha="center")
        ax.set_title(f"{qubit_name} — Ramsey")
        return

    pdiff = diag["pdiff"]
    fitted = diag.get("fitted_curve")
    t_shifted = diag["tau_shifted"]
    t_plot = t_shifted + tau_ns[0]

    ax.plot(t_plot, pdiff, "b-", lw=0.8, alpha=0.7)
    ax.scatter(t_plot, pdiff, c="b", s=8, alpha=0.5, zorder=3, label="Data")

    if fitted is not None:
        ax.plot(t_plot, fitted, "r-", lw=1.5, alpha=0.9, label="Damped cosine")

    ax.set_xlabel("Idle time (ns)")
    ax.set_ylabel("Parity difference")
    ax.set_ylim(-0.05, 1.05)

    t2 = (fit_result or {}).get("t2_star", np.nan)
    ramsey_f = (fit_result or {}).get("ramsey_freq", np.nan)
    title = f"{qubit_name} — Ramsey"
    annotations = []
    if np.isfinite(t2):
        annotations.append(f"T2*={t2:.0f} ns")
    if np.isfinite(ramsey_f):
        annotations.append(f"f={ramsey_f * 1e-6:.3f} MHz")
    if annotations:
        title += f" ({', '.join(annotations)})"
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7)


def _plot_residuals_ax(
    ax: "plt.Axes",
    tau_ns: np.ndarray,
    qubit_name: str,
    fit_result: dict | None = None,
) -> None:
    """Plot fit residuals (data - model)."""
    diag = (fit_result or {}).get("_diag")
    if diag is None:
        ax.text(0.5, 0.5, "No diagnostics", transform=ax.transAxes, ha="center")
        ax.set_title(f"{qubit_name} — Residuals")
        return

    pdiff = diag["pdiff"]
    fitted = diag.get("fitted_curve")
    t_shifted = diag["tau_shifted"]
    t_plot = t_shifted + tau_ns[0]

    if fitted is not None:
        residuals = pdiff - fitted
        ax.plot(t_plot, residuals, "k-", lw=0.6, alpha=0.7)
        ax.scatter(t_plot, residuals, c="k", s=5, alpha=0.4, zorder=3)
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        ax.set_ylabel("Residual")
    else:
        ax.text(0.5, 0.5, "No fit", transform=ax.transAxes, ha="center")

    ax.set_xlabel("Idle time (ns)")
    ax.set_title(f"{qubit_name} — Residuals")


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    ds_fit: xr.Dataset | None,
    qubits: List[Any],
    fit_results: dict,
) -> "plt.Figure":
    """Plot 1-D Ramsey analysis for each qubit.

    Layout (per qubit row):
    * Column 1 — Parity-difference vs idle time with damped-cosine fit.
    * Column 2 — Fit residuals.
    """
    qubit_names = _get_qubit_names_from_ds(ds)
    if not qubit_names:
        fig, _ = plt.subplots(figsize=(6, 4))
        return fig

    n = len(qubit_names)
    ncol = 2
    fig, axes = plt.subplots(n, ncol, figsize=(6 * ncol, 4 * n), squeeze=False)

    for i, qname in enumerate(qubit_names):
        fr = fit_results.get(qname, {})
        tau_ns = np.asarray(ds.tau.values, dtype=float)

        _plot_time_trace_ax(axes[i, 0], tau_ns, qname, fr)
        _plot_residuals_ax(axes[i, 1], tau_ns, qname, fr)

    fig.suptitle("Ramsey (parity diff)")
    fig.tight_layout()
    return fig
