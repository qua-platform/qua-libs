"""Plot 1D time-Rabi parity difference: raw trace and FFT diagnostics."""

from __future__ import annotations

from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calibration_utils.time_rabi_parity_diff.analysis import (
    FFT_FREQ_MIN,
    FFT_FREQ_MAX,
    compute_fft_diagnostic,
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


def _plot_rabi_trace_ax(
    ax: "plt.Axes",
    pdiff: np.ndarray,
    duration_ns: np.ndarray,
    qubit_name: str,
    analysis_signal: str,
    fit_result: dict | None = None,
    fitted_curve: np.ndarray | None = None,
) -> None:
    """Plot raw analysis trace vs pulse duration on the given axes."""
    ax.plot(duration_ns, pdiff, "b-", lw=1, alpha=0.8)
    ax.scatter(duration_ns, pdiff, c="b", s=6, alpha=0.5, zorder=3)
    ax.set_xlabel("Pulse duration (ns)")
    ax.set_ylabel(analysis_signal)
    ax.set_title(f"{qubit_name} — Rabi oscillation")
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

        ax.axvline(
            t_pi, color="lime", ls="--", lw=1.5, alpha=0.9, label=f"t_π = {t_pi:.0f} ns"
        )
        ax.legend(loc="upper right", fontsize=8)


def _plot_fft_ax(
    ax: "plt.Axes",
    qubit_name: str,
    trace: np.ndarray,
    duration_ns: np.ndarray,
    fit_result: dict | None = None,
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
    ax.set_title(f"{qubit_name} — FFT spectrum")
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


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    ds_fit: xr.Dataset | None,
    qubits: List[Any],
    fit_results: dict,
    analysis_signal: str = "E_p1_given_p0_0",
) -> "plt.Figure":
    """Plot Rabi trace and FFT for each qubit.

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
        ax_trace, ax_fft = axes[i, 0], axes[i, 1]
        signal_var = f"{analysis_signal}_{qname}"
        fr = fit_results.get(qname, {})

        durations_ns = np.asarray(plot_ds.pulse_duration.values, dtype=float)

        if signal_var not in plot_ds.data_vars:
            ax_trace.text(
                0.5,
                0.5,
                f"No data for {qname}",
                transform=ax_trace.transAxes,
                ha="center",
            )
            ax_fft.text(
                0.5,
                0.5,
                f"No data for {qname}",
                transform=ax_fft.transAxes,
                ha="center",
            )
            continue

        trace = np.asarray(plot_ds[signal_var].values, dtype=float)
        fit_var = f"{signal_var}_fit"
        fitted_curve = None
        if plot_ds is not None and fit_var in plot_ds.data_vars:
            fitted_curve = np.asarray(plot_ds[fit_var].values, dtype=float)

        _plot_rabi_trace_ax(
            ax_trace,
            trace,
            durations_ns,
            qname,
            analysis_signal,
            fit_result=fr,
            fitted_curve=fitted_curve,
        )
        _plot_fft_ax(ax_fft, qname, trace, durations_ns, fit_result=fr)

    fig.suptitle(f"Time Rabi ({analysis_signal})")
    fig.tight_layout()
    return fig
