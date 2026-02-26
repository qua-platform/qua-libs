"""Plotting for the Ramsey chevron parity-difference analysis.

Produces a multi-panel figure per qubit with two columns:

1. **Chevron heatmap** — 2-D parity-difference map (detuning vs idle
   time) with the fitted resonance frequency overlaid.
2. **Resonance profile** — tau-averaged parity vs detuning, showing the
   measured data and the analytic sum-of-cosines model fit.
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


def _plot_chevron_ax(
    ax: "plt.Axes",
    pdiff: np.ndarray,
    tau_ns: np.ndarray,
    detuning_mhz: np.ndarray,
    qubit_name: str,
    fit_result: dict | None = None,
) -> None:
    """Plot raw chevron heatmap."""
    ax.pcolormesh(
        tau_ns,
        detuning_mhz,
        pdiff,
        cmap="RdBu_r",
        vmin=0,
        vmax=1,
        shading="auto",
    )
    ax.set_xlabel("Idle time (ns)")
    ax.set_ylabel("Detuning (MHz)")
    ax.set_title(f"{qubit_name} — Ramsey chevron")

    if fit_result and fit_result.get("success"):
        freq_off_mhz = fit_result.get("freq_offset", 0) * 1e-6
        ax.axhline(
            freq_off_mhz,
            color="lime",
            ls="--",
            lw=1.5,
            alpha=0.9,
            label=f"f_offset = {freq_off_mhz:.3f} MHz",
        )
        ax.legend(loc="upper right", fontsize=7)


def _plot_resonance_ax(
    ax: "plt.Axes",
    detuning_mhz: np.ndarray,
    qubit_name: str,
    fit_result: dict | None = None,
) -> None:
    """Plot mean parity vs detuning with analytic Ramsey model fit."""
    diag = (fit_result or {}).get("_diag")
    if diag is None:
        ax.text(0.5, 0.5, "No diagnostics", transform=ax.transAxes, ha="center")
        ax.set_title(f"{qubit_name} — Resonance")
        return

    mean_parity = diag["mean_parity"]
    mean_parity_fit = diag.get("mean_parity_fit")

    ax.plot(detuning_mhz, mean_parity, "bo-", ms=3, lw=1, label="Mean parity")
    if mean_parity_fit is not None:
        ax.plot(
            detuning_mhz,
            mean_parity_fit,
            "r-",
            lw=1.5,
            label="Analytic fit",
        )

    if fit_result and fit_result.get("success"):
        freq_off_mhz = fit_result.get("freq_offset", 0) * 1e-6
        t2 = fit_result.get("t2_star", np.nan)
        label = f"Resonance: {freq_off_mhz:.3f} MHz"
        if np.isfinite(t2):
            label += f"\nT₂* = {t2:.0f} ns"
        ax.axvline(
            freq_off_mhz,
            color="lime",
            ls="--",
            lw=1.5,
            alpha=0.9,
            label=label,
        )

    ax.set_xlabel("Detuning (MHz)")
    ax.set_ylabel("Mean parity")
    ax.set_title(f"{qubit_name} — Resonance finding")
    ax.legend(loc="upper right", fontsize=7)


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    ds_fit: xr.Dataset | None,
    qubits: List[Any],
    fit_results: dict,
) -> "plt.Figure":
    """Plot Ramsey chevron for each qubit.

    Layout (per qubit row):
    * Column 1 — Raw chevron heatmap with resonance marker.
    * Column 2 — Mean parity vs detuning with model fit and T2*.
    """
    qubit_names = _get_qubit_names_from_ds(ds)
    if not qubit_names:
        fig, _ = plt.subplots(figsize=(6, 4))
        return fig

    n = len(qubit_names)
    ncol = 2
    fig, axes = plt.subplots(n, ncol, figsize=(6 * ncol, 4 * n), squeeze=False)

    for i, qname in enumerate(qubit_names):
        pdiff_var = f"pdiff_{qname}"
        fr = fit_results.get(qname, {})

        tau_ns = np.asarray(ds.tau.values, dtype=float)
        detuning_mhz = np.asarray(ds.detuning.values, dtype=float) * 1e-6

        if pdiff_var not in ds.data_vars:
            for j in range(ncol):
                axes[i, j].text(
                    0.5,
                    0.5,
                    f"No data for {qname}",
                    transform=axes[i, j].transAxes,
                    ha="center",
                )
            continue

        pdiff = np.asarray(ds[pdiff_var].values, dtype=float)

        _plot_chevron_ax(axes[i, 0], pdiff, tau_ns, detuning_mhz, qname, fr)
        _plot_resonance_ax(axes[i, 1], detuning_mhz, qname, fr)

    fig.suptitle("Ramsey Chevron (parity diff)")
    fig.tight_layout()
    return fig
