"""Plotting for the Ramsey chevron analysis.

Produces a multi-panel figure per qubit with two columns:

1. **Chevron heatmap** — 2-D map of the state response (detuning vs idle
   time) with the fitted resonance frequency overlaid.
2. **Resonance profile** — tau-averaged state response vs detuning, showing the
   measured data and the analytic sum-of-cosines model fit.
"""

from __future__ import annotations

from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calibration_utils.common_utils.plot_style import apply_qubit_outcome_style, empty_figure


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
    success = (fit_result or {}).get("success")
    apply_qubit_outcome_style(ax, qubit_name, success, subtitle="Ramsey chevron")

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
    """Plot mean state response vs detuning with detuning on the y-axis."""
    diag = (fit_result or {}).get("_diag")
    success = (fit_result or {}).get("success")
    if diag is None:
        ax.text(0.5, 0.5, "No diagnostics", transform=ax.transAxes, ha="center")
        apply_qubit_outcome_style(ax, qubit_name, success, subtitle="Resonance")
        return

    mean_state = diag["mean_state"]
    mean_state_fit = diag.get("mean_state_fit")

    ax.scatter(mean_state, detuning_mhz, s=9, color="blue", alpha=0.6, label="Mean signal")
    if mean_state_fit is not None:
        ax.plot(
            mean_state_fit,
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
    apply_qubit_outcome_style(ax, qubit_name, success, subtitle="Resonance finding")
    ax.legend(loc="upper right", fontsize=7)


def plot_raw_data_with_fit(
    ds_fit: xr.Dataset,
    qubits: List[Any],
    fit_results: dict,
) -> "plt.Figure":
    """Plot Ramsey chevron for each qubit.

    Layout (per qubit row):
    * Column 1 — Raw chevron heatmap with resonance marker.
    * Column 2 — Mean signal vs detuning with model fit and T2*.

    """
    qubit_names = [str(v) for v in ds_fit.qubit.values]
    if not qubit_names:
        return empty_figure("No qubit data available for Ramsey chevron.")

    n = len(qubit_names)
    ncol = 2
    fig, axes = plt.subplots(
        n,
        ncol,
        figsize=(9, 4 * n),
        squeeze=False,
        sharey="row",
        gridspec_kw={"width_ratios": [2, 1]},
    )

    for i, qname in enumerate(qubit_names):
        fr = fit_results.get(qname, {})

        tau_ns = np.asarray(ds_fit.tau.values, dtype=float)
        detuning_mhz = np.asarray(ds_fit.detuning.values, dtype=float) * 1e-6

        if "state" not in ds_fit.data_vars:
            for j in range(ncol):
                axes[i, j].text(
                    0.5,
                    0.5,
                    f"No data for {qname}",
                    transform=axes[i, j].transAxes,
                    ha="center",
                )
            continue

        signal_2d = ds_fit.state.sel(qubit=qname, drop=True).transpose("detuning", "tau").values.astype(float)

        _plot_chevron_ax(axes[i, 0], signal_2d, tau_ns, detuning_mhz, qname, fr)
        _plot_resonance_ax(axes[i, 1], detuning_mhz, qname, fr)

    fig.suptitle("Ramsey Chevron")
    fig.tight_layout()
    return fig


def plot_all(
    ds_fit: xr.Dataset,
    qubits: List[Any],
    fit_results: dict | None = None,
) -> dict[str, "plt.Figure"]:
    """Build and return all 11c Ramsey-chevron figures."""
    figures = {
        "raw_data_with_fit": plot_raw_data_with_fit(
            ds_fit,
            qubits,
            fit_results or {},
        )
    }
    return figures
