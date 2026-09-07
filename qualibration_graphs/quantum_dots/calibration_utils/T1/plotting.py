"""Plotting for the T₁ relaxation analysis.

Produces a single-panel figure per qubit showing:

* Thresholded state probability vs idle time τ (scatter + line).
* Exponential-decay fit overlaid.
* Extracted T₁ and amplitude annotated in the title.
"""

from __future__ import annotations

from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calibration_utils.common_utils.plot_style import apply_qubit_outcome_style, empty_figure


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    ds_fit: xr.Dataset | None,
    qubits: List[Any],
    fit_results: dict,
) -> "plt.Figure":
    """Plot T₁ decay with exponential fit for each qubit.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with ``state(qubit, tau)`` and ``tau`` coordinate.
    ds_fit : xr.Dataset or None
        Unused — kept for API consistency with other plotting modules.
    qubits : list
        Qubit objects (used only for count).
    fit_results : dict
        Qubit name → fit-result dict as returned by
        :func:`~.analysis.fit_raw_data`.
    """
    qubit_names = [str(v) for v in ds.qubit.values]
    if not qubit_names:
        return empty_figure("No qubit data available for T1.")

    tau_ns = np.asarray(ds.tau.values, dtype=float)
    # Display in µs if span is large enough
    if tau_ns[-1] > 5000:
        tau_display = tau_ns * 1e-3
        tau_unit = "µs"
    else:
        tau_display = tau_ns
        tau_unit = "ns"

    n_qubits = len(qubit_names)
    fig, axes = plt.subplots(
        n_qubits,
        1,
        figsize=(10, 3.5 * n_qubits),
        squeeze=False,
    )

    for qi, qname in enumerate(qubit_names):
        ax = axes[qi, 0]
        fr = fit_results.get(qname, {})
        diag = fr.get("_diag", {})
        fitted_curve = diag.get("fitted_curve")
        t1 = fr.get("T1", np.nan)
        amp = fr.get("amplitude", np.nan)
        offset = fr.get("offset", np.nan)
        success = fr.get("success", False)

        if "state" in ds.data_vars:
            y_trace = ds.state.sel(qubit=qname, drop=True).transpose("tau").values.astype(float)
        else:
            y_trace = np.full_like(tau_ns, np.nan)

        # Data
        ax.plot(tau_display, y_trace, "-", color="C0", lw=0.8, alpha=0.7)
        ax.scatter(
            tau_display,
            y_trace,
            c="C0",
            s=8,
            alpha=0.5,
            zorder=3,
            label="Data",
        )

        # Fit curve
        if fitted_curve is not None:
            ax.plot(
                tau_display,
                fitted_curve,
                "-",
                color="C1",
                lw=1.5,
                alpha=0.9,
                label="Exp. fit",
            )

        ax.set_xlabel(f"Idle time ({tau_unit})")
        ax.set_ylabel("State")
        ax.set_ylim(-0.05, 1.05)

        # Title with fit parameters
        title = f"{qname}"
        if success and np.isfinite(t1):
            if t1 > 5000:
                title += f"  |  T₁ = {t1 * 1e-3:.2f} µs"
            else:
                title += f"  |  T₁ = {t1:.1f} ns"
            if np.isfinite(amp):
                title += f",  A = {amp:.3f}"
            if np.isfinite(offset):
                title += f",  offset = {offset:.3f}"
        elif not success:
            title += "  |  fit failed"
        apply_qubit_outcome_style(ax, qname, success, subtitle=title[len(qname):].strip(" |"))
        ax.title.set_fontsize(10)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "T₁ relaxation — exponential decay fit",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def plot_all(
    ds_raw: xr.Dataset,
    qubits: List[Any],
    *,
    ds_fit: xr.Dataset | None = None,
    fit_results: dict | None = None,
    show: bool = True,
) -> dict[str, "plt.Figure"]:
    """Build and return all T1 figures."""
    figures = {
        "raw_data_with_fit": plot_raw_data_with_fit(
            ds_raw,
            ds_fit,
            qubits,
            fit_results or {},
        )
    }
    if show:
        plt.show()
    return figures
