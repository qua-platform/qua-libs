"""Plot 1D time-Rabi conditional expectations: raw trace and FFT diagnostics."""

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
from calibration_utils.time_rabi.analysis import (
    FFT_FREQ_MIN,
    FFT_FREQ_MAX,
    compute_fft_diagnostic,
)


def _plot_rabi_trace_ax(
    ax: plt.Axes,
    pdiff: np.ndarray,
    duration_ns: np.ndarray,
    qubit_name: str,
    analysis_signal: str,
    fit_result: dict | None = None,
    fitted_curve: np.ndarray | None = None,
    success: bool | None = None,
) -> None:
    """Plot raw analysis trace vs pulse duration on the given axes."""
    ax.plot(duration_ns, pdiff, "b-", lw=1, alpha=0.8)
    ax.scatter(duration_ns, pdiff, c="b", s=6, alpha=0.5, zorder=3)
    ax.set_xlabel("Pulse duration (ns)")
    ax.set_ylabel(analysis_signal)
    apply_qubit_outcome_style(ax, qubit_name, success, subtitle="Rabi oscillation")
    ax.set_ylim(-0.05, 1.05)

    if fit_result and fit_result.get("success"):
        t_pi = fit_result.get("optimal_duration", 0)

        if fitted_curve is not None:
            ax.plot(
                duration_ns,
                fitted_curve,
                "r-",
                lw=1.5,
                alpha=0.9,
                label="Damped sinusoid fit",
            )

        ax.axvline(t_pi, color="lime", ls="--", lw=1.5, alpha=0.9, label=f"t_π = {t_pi:.0f} ns")
        ax.legend(loc="upper right", fontsize=8)


def _plot_fft_ax(
    ax: plt.Axes,
    qubit_name: str,
    trace: np.ndarray,
    duration_ns: np.ndarray,
    fit_result: dict | None = None,
    success: bool | None = None,
) -> None:
    """Plot FFT magnitude spectrum with peak fit on the given axes."""
    diag = compute_fft_diagnostic(trace, duration_ns)
    freqs_fft = diag["fft_freqs"]
    magnitude = diag["fft_magnitude"]
    peak_curve = diag.get("peak_curve")

    mask = (freqs_fft >= FFT_FREQ_MIN) & (freqs_fft <= FFT_FREQ_MAX)
    f_plot = freqs_fft[mask] * 1e3  # cycles/ns → 1/μs

    ax.plot(f_plot, magnitude[mask], "b-", lw=1, label="FFT")
    if peak_curve is not None:
        ax.plot(f_plot, peak_curve[mask], "r-", lw=1.5, label="Peak fit")

    ax.set_xlabel("Frequency (1/μs)")
    ax.set_ylabel("|FFT|")
    apply_qubit_outcome_style(ax, qubit_name, success, subtitle="FFT spectrum")
    ax.set_xlim(f_plot[0], f_plot[-1])

    if fit_result and fit_result.get("success"):
        omega = fit_result.get("rabi_frequency", 0)
        f_rabi_us = omega / (2.0 * np.pi) * 1e3  # rad/ns → 1/μs
        ax.axvline(
            f_rabi_us,
            color="lime",
            ls="--",
            lw=1,
            alpha=0.9,
            label=f"f_Rabi = {f_rabi_us:.1f} /μs",
        )

    ax.legend(loc="upper right", fontsize=8)


def plot_rabi_traces(
    ds_fit: xr.Dataset,
    qubits: List[Any],
    fit_results: dict,
    analysis_signal: str = "E_p1_given_p0_0",
) -> Figure:
    """Plot time-Rabi traces with fit overlays (one panel per qubit)."""
    qubit_names = get_parity_item_names(
        ds_fit,
        analysis_signal,
        item_names=[getattr(q, "name", f"Q{i}") for i, q in enumerate(qubits)],
    )
    if not qubit_names:
        return empty_figure("No qubit data found in ds_fit.")

    n = len(qubit_names)
    fig, axes = plt.subplots(1, n, figsize=(max(5 * n, 8), 4), squeeze=False)
    axes = axes.flatten()

    durations_ns = np.asarray(ds_fit.pulse_duration.values, dtype=float)
    for ax, qname in zip(axes, qubit_names):
        signal_var = f"{analysis_signal}_{qname}"
        success = qubit_success(fit_results, qname)
        fr = fit_results.get(qname, {})

        if signal_var not in ds_fit.data_vars:
            apply_qubit_outcome_style(ax, qname, success, subtitle="Rabi oscillation")
            ax.text(0.5, 0.5, f"No data for {qname}", transform=ax.transAxes, ha="center")
            continue

        trace = np.asarray(ds_fit[signal_var].values, dtype=float)
        fit_var = f"{signal_var}_fit"
        fitted_curve = np.asarray(ds_fit[fit_var].values, dtype=float) if fit_var in ds_fit.data_vars else None
        _plot_rabi_trace_ax(
            ax,
            trace,
            durations_ns,
            qname,
            analysis_signal,
            fit_result=fr,
            fitted_curve=fitted_curve,
            success=success,
        )

    fig.suptitle(f"Time Rabi ({analysis_signal})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_fft_spectra(
    ds_fit: xr.Dataset,
    qubits: List[Any],
    fit_results: dict,
    analysis_signal: str = "E_p1_given_p0_0",
) -> Figure:
    """Plot FFT magnitude spectra (one panel per qubit)."""
    qubit_names = get_parity_item_names(
        ds_fit,
        analysis_signal,
        item_names=[getattr(q, "name", f"Q{i}") for i, q in enumerate(qubits)],
    )
    if not qubit_names:
        return empty_figure("No qubit data found in ds_fit.")

    n = len(qubit_names)
    fig, axes = plt.subplots(1, n, figsize=(max(5 * n, 8), 4), squeeze=False)
    axes = axes.flatten()

    durations_ns = np.asarray(ds_fit.pulse_duration.values, dtype=float)
    for ax, qname in zip(axes, qubit_names):
        signal_var = f"{analysis_signal}_{qname}"
        success = qubit_success(fit_results, qname)
        fr = fit_results.get(qname, {})

        if signal_var not in ds_fit.data_vars:
            apply_qubit_outcome_style(ax, qname, success, subtitle="FFT spectrum")
            ax.text(0.5, 0.5, f"No data for {qname}", transform=ax.transAxes, ha="center")
            continue

        trace = np.asarray(ds_fit[signal_var].values, dtype=float)
        _plot_fft_ax(ax, qname, trace, durations_ns, fit_result=fr, success=success)

    fig.suptitle(f"Time Rabi FFT ({analysis_signal})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_all(
    ds_fit: xr.Dataset,
    qubits: List[Any],
    fit_results: dict | None = None,
    analysis_signal: str = "E_p1_given_p0_0",
) -> Dict[str, Figure]:
    """Return all standard figures for time-Rabi analysis."""
    fit_results = fit_results or {}
    return {
        "rabi": plot_rabi_traces(ds_fit, qubits, fit_results, analysis_signal),
        "fft": plot_fft_spectra(ds_fit, qubits, fit_results, analysis_signal),
    }
