"""Plotting for the Ramsey chevron joint-outcome / conditional-readout analysis.

Produces a multi-panel figure per qubit with two columns:

1. **Chevron heatmap** — 2-D map of the analysis signal (detuning vs idle
   time) with the fitted resonance frequency overlaid.
2. **Resonance profile** — tau-averaged signal vs detuning, showing the
   measured data and the analytic sum-of-cosines model fit.
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


def _plot_chevron_ax(
    ax: "plt.Axes",
    signal_2d: np.ndarray,
    tau_ns: np.ndarray,
    detuning_mhz: np.ndarray,
    qubit_name: str,
    fit_result: dict | None = None,
) -> None:
    """Plot raw chevron heatmap."""
    ax.pcolormesh(
        tau_ns,
        detuning_mhz,
        signal_2d,
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
    """Plot mean parity vs detuning with detuning on the y-axis."""
    diag = (fit_result or {}).get("_diag")
    if diag is None:
        ax.text(0.5, 0.5, "No diagnostics", transform=ax.transAxes, ha="center")
        ax.set_title(f"{qubit_name} — Resonance")
        return

    mean_parity = diag["mean_parity"]
    mean_parity_fit = diag.get("mean_parity_fit")

    ax.scatter(mean_parity, detuning_mhz, s=9, color="blue", alpha=0.6, label="Mean signal")
    if mean_parity_fit is not None:
        ax.plot(
            mean_parity_fit,
            detuning_mhz,
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
        ax.axhline(
            freq_off_mhz,
            color="lime",
            ls="--",
            lw=1.5,
            alpha=0.9,
            label=label,
        )

    ax.set_xlabel("Mean signal")
    ax.set_title(f"{qubit_name} — Resonance finding")
    ax.legend(loc="upper right", fontsize=7)


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    ds_fit: xr.Dataset | None,
    qubits: List[Any],
    fit_results: dict,
    analysis_signal: str = "E_p2_given_p1_0",
) -> "plt.Figure":
    """Plot Ramsey chevron for each qubit.

    Layout (per qubit row):
    * Column 1 — Raw chevron heatmap with resonance marker.
    * Column 2 — Mean signal vs detuning with model fit and T2*.

    Parameters
    ----------
    analysis_signal
        Data-variable prefix for the plotted 2-D trace (same as
        ``node.parameters.analysis_signal``).
    """
    qubit_names = _get_qubit_names_from_ds(ds, qubits, analysis_signal)
    if not qubit_names:
        fig, _ = plt.subplots(figsize=(6, 4))
        return fig

    n = len(qubit_names)
    ncol = 2
    fig, axes = plt.subplots(
        n, ncol,
        figsize=(9, 4 * n),
        squeeze=False,
        sharey="row",
        gridspec_kw={"width_ratios": [2, 1]},
    )

    for i, qname in enumerate(qubit_names):
        signal_var = f"{analysis_signal}_{qname}"
        fr = fit_results.get(qname, {})

        tau_ns = np.asarray(ds.tau.values, dtype=float)
        detuning_mhz = np.asarray(ds.detuning.values, dtype=float) * 1e-6

        if signal_var not in ds.data_vars:
            for j in range(ncol):
                axes[i, j].text(
                    0.5,
                    0.5,
                    f"No data for {qname}",
                    transform=axes[i, j].transAxes,
                    ha="center",
                )
            continue

        signal_2d = np.asarray(ds[signal_var].values, dtype=float)

        _plot_chevron_ax(axes[i, 0], signal_2d, tau_ns, detuning_mhz, qname, fr)
        _plot_resonance_ax(axes[i, 1], detuning_mhz, qname, fr)

    fig.suptitle("Ramsey Chevron")
    fig.tight_layout()
    return fig
