"""Plotting for the error-amplified power-Rabi analysis.

Produces a multi-panel figure per qubit with two columns (mirroring the
Ramsey chevron layout):

1. **Heatmap** — 2-D map of the analysis signal (amplitude vs
   n_pulses) with the fitted optimal amplitude overlaid.
2. **Resonance profile** — n_pulses-averaged signal vs amplitude,
   showing the measured data and the analytic model fit.
"""

from __future__ import annotations

from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calibration_utils.measurement_utils.measurement_streams import get_parity_item_names


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


def _plot_heatmap_ax(
    ax: "plt.Axes",
    signal_2d: np.ndarray,
    amps: np.ndarray,
    n_pulses: np.ndarray,
    qubit_name: str,
    fit_result: dict | None = None,
) -> None:
    """Plot raw 2-D heatmap: amplitude (x) vs n_pulses (y)."""
    vmin = float(np.nanmin(signal_2d))
    vmax = float(np.nanmax(signal_2d))
    vcenter = (vmin + vmax) / 2.0
    half = max(vmax - vcenter, vcenter - vmin, 1e-9)
    ax.pcolormesh(
        amps,
        n_pulses,
        signal_2d,
        cmap="RdBu_r",
        vmin=vcenter - half,
        vmax=vcenter + half,
        shading="auto",
    )
    ax.set_xlabel("Amplitude prefactor")
    ax.set_ylabel("Number of pulses")
    ax.set_title(f"{qubit_name} — Error-amplified Rabi")

    if fit_result and fit_result.get("success"):
        opt = fit_result.get("opt_amp", 0)
        ax.axvline(
            opt,
            color="lime",
            ls="--",
            lw=1.5,
            alpha=0.9,
            label=f"a_π = {opt:.4f}",
        )
        ax.legend(loc="upper right", fontsize=7)


def _plot_resonance_ax(
    ax: "plt.Axes",
    amps: np.ndarray,
    qubit_name: str,
    fit_result: dict | None = None,
    mean_signal: np.ndarray | None = None,
    mean_signal_fit: np.ndarray | None = None,
) -> None:
    """Plot mean signal vs amplitude with analytic model fit."""
    if mean_signal is None:
        ax.text(0.5, 0.5, "No diagnostics", transform=ax.transAxes, ha="center")
        ax.set_title(f"{qubit_name} — Resonance")
        return

    ax.plot(amps, mean_signal, "bo-", ms=3, lw=1, label="Mean signal")
    if mean_signal_fit is not None:
        ax.plot(
            amps,
            mean_signal_fit,
            "r-",
            lw=1.5,
            label="Analytic fit",
        )

    if fit_result and fit_result.get("success"):
        opt = fit_result.get("opt_amp", 0)
        n_eff = fit_result.get("n_eff", np.nan)
        label = f"a_π = {opt:.4f}"
        if np.isfinite(n_eff):
            label += f"\nN_eff = {n_eff:.0f}"
        ax.axvline(
            opt,
            color="lime",
            ls="--",
            lw=1.5,
            alpha=0.9,
            label=label,
        )

    ax.set_xlabel("Amplitude prefactor")
    ax.set_ylabel("Mean signal")
    ax.set_title(f"{qubit_name} — Optimal amplitude")
    ax.legend(loc="upper right", fontsize=7)


def plot_raw_data_error_amplified(
    ds: xr.Dataset,
    ds_fit: xr.Dataset | None,  # noqa: ARG001
    qubits: List[Any],  # noqa: ARG001
    fit_results: dict,
    analysis_signal: str = "E_p1_given_p0_0",
) -> "plt.Figure":
    """Plot error-amplified power Rabi for each qubit.

    ``ds`` should be the processed dataset (``ds_fit``) containing
    ``{analysis_signal}_{qubit}`` variables.
    """
    plot_ds = ds_fit if ds_fit is not None else ds
    qubit_names = _get_qubit_names_from_ds(plot_ds, qubits, analysis_signal)
    if not qubit_names:
        fig, _ = plt.subplots(figsize=(6, 4))
        return fig

    n = len(qubit_names)
    ncol = 2
    fig, axes = plt.subplots(n, ncol, figsize=(6 * ncol, 4 * n), squeeze=False)

    for i, qname in enumerate(qubit_names):
        signal_var = f"{analysis_signal}_{qname}"
        fr = fit_results.get(qname, {})

        amps = np.asarray(plot_ds.amp_prefactor.values, dtype=float)
        n_pulses = np.asarray(plot_ds.n_pulses.values, dtype=float)

        if signal_var not in plot_ds.data_vars:
            for j in range(ncol):
                axes[i, j].text(
                    0.5,
                    0.5,
                    f"No data for {qname}",
                    transform=axes[i, j].transAxes,
                    ha="center",
                )
            continue

        signal_2d = np.asarray(plot_ds[signal_var].values, dtype=float)
        mean_var = f"{signal_var}_mean"
        mean_fit_var = f"{signal_var}_mean_fit"
        mean_signal = (
            np.asarray(plot_ds[mean_var].values, dtype=float)
            if mean_var in plot_ds.data_vars
            else None
        )
        mean_signal_fit = (
            np.asarray(plot_ds[mean_fit_var].values, dtype=float)
            if mean_fit_var in plot_ds.data_vars
            else None
        )

        _plot_heatmap_ax(axes[i, 0], signal_2d, amps, n_pulses, qname, fr)
        _plot_resonance_ax(
            axes[i, 1],
            amps,
            qname,
            fr,
            mean_signal=mean_signal,
            mean_signal_fit=mean_signal_fit,
        )

    fig.suptitle(f"Error-amplified Power Rabi ({analysis_signal})")
    fig.tight_layout()
    return fig
