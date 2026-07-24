"""Plotting for the error-amplified power-Rabi analysis."""

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
    ax: plt.Axes,
    signal_2d: np.ndarray,
    amps: np.ndarray,
    n_pulses: np.ndarray,
    qubit_name: str,
    fit_result: dict | None = None,
    success: bool | None = None,
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
    """Plot mean signal vs amplitude with analytic model fit."""
    if mean_signal is None:
        apply_qubit_outcome_style(ax, qubit_name, success, subtitle="Optimal amplitude")
        ax.text(0.5, 0.5, "No diagnostics", transform=ax.transAxes, ha="center")
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
    apply_qubit_outcome_style(ax, qubit_name, success, subtitle="Optimal amplitude")
    ax.legend(loc="upper right", fontsize=7)


def plot_heatmaps(
    ds_fit: xr.Dataset,
    qubits: List[Any],
    fit_results: dict,
    analysis_signal: str = "E_p1_given_p0_0",
) -> Figure:
    """Plot error-amplified Rabi heatmaps (one panel per qubit)."""
    qubit_names = _get_qubit_names_from_ds(ds_fit, qubits, analysis_signal)
    if not qubit_names:
        return empty_figure("No qubit data found in ds_fit.")

    n = len(qubit_names)
    fig, axes = plt.subplots(1, n, figsize=(max(5 * n, 8), 4), squeeze=False)
    axes = axes.flatten()

    amps = np.asarray(ds_fit.amp_prefactor.values, dtype=float)
    n_pulses = np.asarray(ds_fit.n_pulses.values, dtype=float)

    for ax, qname in zip(axes, qubit_names):
        signal_var = f"{analysis_signal}_{qname}"
        success = qubit_success(fit_results, qname)
        fr = fit_results.get(qname, {})

        if signal_var not in ds_fit.data_vars:
            apply_qubit_outcome_style(ax, qname, success, subtitle="Error-amplified Rabi")
            ax.text(0.5, 0.5, f"No data for {qname}", transform=ax.transAxes, ha="center")
            continue

        signal_2d = np.asarray(ds_fit[signal_var].values, dtype=float)
        _plot_heatmap_ax(ax, signal_2d, amps, n_pulses, qname, fr, success)

    fig.suptitle(f"Error-amplified Power Rabi heatmap ({analysis_signal})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_resonance_profiles(
    ds_fit: xr.Dataset,
    qubits: List[Any],
    fit_results: dict,
    analysis_signal: str = "E_p1_given_p0_0",
) -> Figure:
    """Plot n_pulses-averaged resonance profiles (one panel per qubit)."""
    qubit_names = _get_qubit_names_from_ds(ds_fit, qubits, analysis_signal)
    if not qubit_names:
        return empty_figure("No qubit data found in ds_fit.")

    n = len(qubit_names)
    fig, axes = plt.subplots(1, n, figsize=(max(5 * n, 8), 4), squeeze=False)
    axes = axes.flatten()

    amps = np.asarray(ds_fit.amp_prefactor.values, dtype=float)

    for ax, qname in zip(axes, qubit_names):
        signal_var = f"{analysis_signal}_{qname}"
        success = qubit_success(fit_results, qname)
        fr = fit_results.get(qname, {})

        if signal_var not in ds_fit.data_vars:
            apply_qubit_outcome_style(ax, qname, success, subtitle="Optimal amplitude")
            ax.text(0.5, 0.5, f"No data for {qname}", transform=ax.transAxes, ha="center")
            continue

        mean_var = f"{signal_var}_mean"
        mean_fit_var = f"{signal_var}_mean_fit"
        mean_signal = (
            np.asarray(ds_fit[mean_var].values, dtype=float)
            if mean_var in ds_fit.data_vars
            else None
        )
        mean_signal_fit = (
            np.asarray(ds_fit[mean_fit_var].values, dtype=float)
            if mean_fit_var in ds_fit.data_vars
            else None
        )
        _plot_resonance_ax(
            ax,
            amps,
            qname,
            fr,
            mean_signal=mean_signal,
            mean_signal_fit=mean_signal_fit,
            success=success,
        )

    fig.suptitle(f"Error-amplified Power Rabi resonance ({analysis_signal})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_all(
    ds_fit: xr.Dataset,
    qubits: List[Any],
    fit_results: dict | None = None,
    analysis_signal: str = "E_p1_given_p0_0",
) -> Dict[str, Figure]:
    """Return all standard figures for error-amplified power-Rabi analysis."""
    fit_results = fit_results or {}
    return {
        "heatmap": plot_heatmaps(ds_fit, qubits, fit_results, analysis_signal),
        "resonance": plot_resonance_profiles(ds_fit, qubits, fit_results, analysis_signal),
    }