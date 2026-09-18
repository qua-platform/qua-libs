"""Plotting for error-amplified power-Rabi analysis."""

from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from calibration_utils.common_utils.plot_style import (
    apply_qubit_outcome_style,
    empty_figure,
    qubit_success,
)


def _plot_heatmap_ax(
    ax: plt.Axes,
    signal_2d: np.ndarray,
    amps: np.ndarray,
    n_pulses: np.ndarray,
    qubit_name: str,
    fit_result: dict | None = None,
    success: bool | None = None,
) -> None:
    vmin = float(np.nanmin(signal_2d))
    vmax = float(np.nanmax(signal_2d))
    vcenter = (vmin + vmax) / 2.0
    half = max(vmax - vcenter, vcenter - vmin, 1e-9)
    im = ax.pcolormesh(
        amps,
        n_pulses,
        signal_2d,
        cmap="RdBu_r",
        vmin=vcenter - half,
        vmax=vcenter + half,
        shading="auto",
    )
    plt.colorbar(im, ax=ax, label="state")
    ax.set_xlabel("Amplitude prefactor")
    ax.set_ylabel("Number of pulses")
    apply_qubit_outcome_style(ax, qubit_name, success, subtitle="Error-amplified Rabi")

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
    ax: plt.Axes,
    amps: np.ndarray,
    qubit_name: str,
    fit_result: dict | None = None,
    mean_signal: np.ndarray | None = None,
    mean_signal_fit: np.ndarray | None = None,
    success: bool | None = None,
) -> None:
    if mean_signal is None:
        apply_qubit_outcome_style(ax, qubit_name, success, subtitle="Optimal amplitude")
        ax.text(0.5, 0.5, "No diagnostics", transform=ax.transAxes, ha="center")
        return

    ax.plot(amps, mean_signal, "bo-", ms=3, lw=1, label="Mean state")
    if mean_signal_fit is not None and np.any(np.isfinite(mean_signal_fit)):
        ax.plot(amps, mean_signal_fit, "r-", lw=1.5, label="Analytic fit")

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
    ax.set_ylabel("state")
    apply_qubit_outcome_style(ax, qubit_name, success, subtitle="Optimal amplitude")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=7)


def _empty_message() -> str:
    return (
        "No qubit data found in ds_fit.\n"
        "Check that generate_simulated_data / analyse_data ran successfully\n"
        "and that node.parameters.qubits (or active_qubit_names) is set."
    )


def plot_heatmaps(
    ds_fit: xr.Dataset,
    qubits: List[Any],
    fit_results: dict,
) -> Figure:
    qubit_names = [str(v) for v in ds_fit.qubit.values]
    if not qubit_names:
        return empty_figure(_empty_message())

    n = len(qubit_names)
    fig, axes = plt.subplots(1, n, figsize=(max(5 * n, 8), 4), squeeze=False)
    axes = axes.flatten()

    amps = np.asarray(ds_fit.amp_prefactor.values, dtype=float)
    n_pulses = np.asarray(ds_fit.n_pulses.values, dtype=float)

    for ax, qname in zip(axes, qubit_names):
        success = qubit_success(fit_results, qname)
        fr = fit_results.get(qname, {})
        signal_2d = np.asarray(ds_fit.state.sel(qubit=qname).values, dtype=float)
        _plot_heatmap_ax(ax, signal_2d, amps, n_pulses, qname, fr, success)

    fig.suptitle("Error-amplified Power Rabi heatmap (state)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_resonance_profiles(
    ds_fit: xr.Dataset,
    qubits: List[Any],
    fit_results: dict,
) -> Figure:
    qubit_names = [str(v) for v in ds_fit.qubit.values]
    if not qubit_names:
        return empty_figure(_empty_message())

    n = len(qubit_names)
    fig, axes = plt.subplots(1, n, figsize=(max(5 * n, 8), 4), squeeze=False)
    axes = axes.flatten()

    amps = np.asarray(ds_fit.amp_prefactor.values, dtype=float)

    for ax, qname in zip(axes, qubit_names):
        success = qubit_success(fit_results, qname)
        fr = fit_results.get(qname, {})

        mean_signal = None
        mean_signal_fit = None
        if "state_mean" in ds_fit.data_vars:
            mean_signal = np.asarray(ds_fit.state_mean.sel(qubit=qname).values, dtype=float)
        if "state_mean_fit" in ds_fit.data_vars:
            mean_signal_fit = np.asarray(ds_fit.state_mean_fit.sel(qubit=qname).values, dtype=float)

        _plot_resonance_ax(
            ax,
            amps,
            qname,
            fr,
            mean_signal=mean_signal,
            mean_signal_fit=mean_signal_fit,
            success=success,
        )

    fig.suptitle("Error-amplified Power Rabi resonance (state)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_all(
    ds_fit: xr.Dataset,
    qubits: List[Any],
    fit_results: dict | None = None,
) -> Dict[str, Figure]:
    fit_results = fit_results or {}
    return {
        "heatmap": plot_heatmaps(ds_fit, qubits, fit_results),
        "resonance": plot_resonance_profiles(ds_fit, qubits, fit_results),
    }
