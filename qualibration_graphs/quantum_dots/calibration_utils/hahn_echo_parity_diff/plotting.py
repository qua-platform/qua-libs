"""Plotting utilities for the Hahn echo T₂ measurement.

Generates a multi-row figure (one row per qubit) showing:
  - Raw parity-difference data vs per-arm idle time τ.
  - Fitted exponential decay  P(τ) = offset + A·exp(−2τ / T₂_echo).
  - Annotated T₂_echo, amplitude, and offset in each panel title.

Time axes are displayed in µs when the sweep range exceeds 5 µs.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_raw_data_with_fit(
    ds_raw: xr.Dataset,
    fit_results: dict[str, dict[str, Any]],
    qubits: list[Any],
) -> plt.Figure:
    """Create a multi-panel Hahn echo figure (one row per qubit).

    Parameters
    ----------
    ds_raw : xr.Dataset
        Raw dataset with ``tau`` coordinate (ns) and ``pdiff_<qubit>`` vars.
    fit_results : dict
        Output of :func:`~.analysis.fit_raw_data`.
    qubits : list
        Qubit objects (each must have a ``.name`` attribute).

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_qubits = len(qubits)
    fig, axes = plt.subplots(
        n_qubits,
        1,
        figsize=(8, 3.5 * n_qubits),
        squeeze=False,
    )

    tau_ns = ds_raw.coords["tau"].values.astype(np.float64)
    use_us = float(tau_ns.max()) > 5_000.0
    tau_plot = tau_ns / 1e3 if use_us else tau_ns
    time_unit = "µs" if use_us else "ns"

    for idx, qubit in enumerate(qubits):
        ax = axes[idx, 0]
        qname = qubit.name
        var_name = f"pdiff_{qname}"

        if var_name not in ds_raw.data_vars:
            ax.set_title(f"{qname} — no data")
            continue

        pdiff = ds_raw[var_name].values.astype(np.float64)
        r = fit_results.get(qname, {})

        # Raw data
        ax.plot(tau_plot, pdiff, "o", ms=3, alpha=0.5, label="data")

        # Fitted curve
        fitted = r.get("fitted_curve")
        if fitted is not None and len(fitted) == len(tau_plot):
            ax.plot(tau_plot, fitted, "-", lw=2, color="C1", label="fit")

        # Annotation
        t2 = r.get("T2_echo", float("nan"))
        amp = r.get("amplitude", float("nan"))
        off = r.get("offset", float("nan"))
        status = "OK" if r.get("success") else "FAIL"

        if use_us and np.isfinite(t2):
            t2_str = f"T₂_echo = {t2 / 1e3:.2f} µs"
        else:
            t2_str = f"T₂_echo = {t2:.1f} ns"

        ax.set_title(f"{qname}  [{status}]  {t2_str},  A = {amp:.4f},  offset = {off:.4f}")
        ax.set_xlabel(f"Per-arm idle time τ ({time_unit})")
        ax.set_ylabel("Parity difference")
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Hahn Echo T₂ Measurement", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
