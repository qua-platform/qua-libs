"""Plotting utilities for the XY8 dynamical decoupling T₂ measurement.

Generates a single-panel figure per qubit showing:
  - Raw conditional readout vs half inter-pulse spacing τ.
  - Fitted exponential decay  P(τ) = offset + A·exp(−16τ / T₂_XY8).
  - Annotated T₂_XY8, amplitude, and offset in the panel title.

Time axes are displayed in µs when the sweep range exceeds 5 µs.
"""

from __future__ import annotations

from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calibration_utils.common_utils.parity_streams import get_parity_item_names


def _get_qubit_names_from_ds(
    ds: xr.Dataset,
    qubits: List[Any],
    analysis_signal: str,
) -> List[str]:
    return get_parity_item_names(
        ds,
        analysis_signal,
        item_names=[getattr(q, "name", f"Q{i}") for i, q in enumerate(qubits)],
    )


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    ds_fit: xr.Dataset | None,
    qubits: List[Any],
    fit_results: dict,
    analysis_signal: str = "E_p2_given_p1_0",
) -> plt.Figure:
    """Create a multi-panel XY8 figure (one row per qubit).

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset with ``tau`` (ns) and joint-stream / analysis vars.
    ds_fit : xr.Dataset or None
        Unused — kept for API consistency with other parity-diff nodes.
    qubits : list
        Qubit objects (used for layout; qubit names are taken from the dataset).
    fit_results : dict
        Output of :func:`~.analysis.fit_raw_data`.
    analysis_signal : str
        Which conditional expectation to plot (must match processing).

    Returns
    -------
    matplotlib.figure.Figure
    """
    qubit_names = _get_qubit_names_from_ds(ds, qubits, analysis_signal)
    if not qubit_names:
        fig, _ = plt.subplots(figsize=(6, 4))
        return fig

    tau_ns = np.asarray(ds.tau.values, dtype=float)

    n = len(qubit_names)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3.5 * n), squeeze=False)

    for i, qname in enumerate(qubit_names):
        ax = axes[i, 0]
        fr = fit_results.get(qname, {})
        diag = fr.get("_diag", {})

        y = diag.get("signal")
        fitted = diag.get("fitted_curve")

        tau_plot = tau_ns.astype(float)
        use_us = float(tau_plot.max()) > 5_000.0
        x_plot = tau_plot / 1e3 if use_us else tau_plot
        time_unit = "µs" if use_us else "ns"

        if y is not None:
            ax.plot(x_plot, y, "b-", lw=0.8, alpha=0.7)
            ax.scatter(x_plot, y, c="b", s=8, alpha=0.5, zorder=3, label="Data")

            if fitted is not None and len(fitted) == len(y):
                ax.plot(
                    x_plot, fitted, "-", lw=2, color="C1", alpha=0.9, label="Exp. fit"
                )
        else:
            ax.text(0.5, 0.5, "No fit data", transform=ax.transAxes, ha="center")

        ax.set_xlabel(f"Half inter-pulse spacing τ ({time_unit})")
        ax.set_ylabel(analysis_signal)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="best", fontsize=8)

        t2 = fr.get("T2_xy8", float("nan"))
        amp = fr.get("amplitude", float("nan"))
        off = fr.get("offset", float("nan"))
        status = "OK" if fr.get("success") else "FAIL"

        if use_us and np.isfinite(t2):
            t2_str = f"T₂_XY8 = {t2 / 1e3:.2f} µs"
        else:
            t2_str = f"T₂_XY8 = {t2:.1f} ns"

        ax.set_title(
            f"{qname}\n[{status}] {t2_str},  A = {amp:.4f},  offset = {off:.4f}"
        )

    fig.suptitle(
        f"XY8 Dynamical Decoupling T₂ ({analysis_signal})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig
