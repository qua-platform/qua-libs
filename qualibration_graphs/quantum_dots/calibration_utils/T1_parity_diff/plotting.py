"""Plotting for the T₁ parity-difference relaxation analysis.

Produces a single-panel figure per qubit showing:

* Parity-difference data vs idle time τ (scatter + line).
* Exponential-decay fit overlaid.
* Extracted T₁ and amplitude annotated in the title.
"""

from __future__ import annotations

from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _get_qubit_names_from_ds(ds: xr.Dataset) -> List[str]:
    """Extract qubit names from ``pdiff_<name>`` data-variable keys."""
    pdiff_vars = [v for v in ds.data_vars if v.startswith("pdiff_") and not v.endswith("_fit")]
    return [v.replace("pdiff_", "") for v in sorted(pdiff_vars)]


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
        Raw dataset with ``pdiff_<qubit>`` and ``tau`` coordinate.
    ds_fit : xr.Dataset or None
        Unused — kept for API consistency with other plotting modules.
    qubits : list
        Qubit objects (used only for count).
    fit_results : dict
        Qubit name → fit-result dict as returned by
        :func:`~.analysis.fit_raw_data`.
    """
    qubit_names = _get_qubit_names_from_ds(ds)
    if not qubit_names:
        fig, _ = plt.subplots(figsize=(6, 4))
        return fig

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

        pdiff_var = f"pdiff_{qname}"
        if pdiff_var in ds.data_vars:
            pdiff = np.asarray(ds[pdiff_var].values, dtype=float)
        else:
            pdiff = np.full_like(tau_ns, np.nan)

        # Data
        ax.plot(tau_display, pdiff, "-", color="C0", lw=0.8, alpha=0.7)
        ax.scatter(
            tau_display,
            pdiff,
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
        ax.set_ylabel("Parity difference")
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
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("T₁ relaxation — exponential decay fit", fontsize=12)
    fig.tight_layout()
    return fig
