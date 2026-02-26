"""Plot 1D time-Rabi parity difference: raw trace and FFT diagnostics."""

from __future__ import annotations

from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calibration_utils.time_rabi_parity_diff.analysis import FFT_FREQ_MIN, FFT_FREQ_MAX


def _get_qubit_names_from_ds(ds: xr.Dataset) -> List[str]:
    """Resolve qubit names from dataset pdiff_ vars (pdiff_Q1 -> Q1)."""
    pdiff_vars = [v for v in ds.data_vars if v.startswith("pdiff_") and not v.endswith("_fit")]
    return [v.replace("pdiff_", "") for v in sorted(pdiff_vars)]


def _plot_rabi_trace_ax(
    ax: "plt.Axes",
    pdiff: np.ndarray,
    duration_ns: np.ndarray,
    qubit_name: str,
    fit_result: dict | None = None,
) -> None:
    """Plot raw parity difference vs pulse duration on the given axes."""
    ax.plot(duration_ns, pdiff, "b-", lw=1, alpha=0.8)
    ax.scatter(duration_ns, pdiff, c="b", s=6, alpha=0.5, zorder=3)
    ax.set_xlabel("Pulse duration (ns)")
    ax.set_ylabel("Parity difference")
    ax.set_title(f"{qubit_name} — Rabi oscillation")
    ax.set_ylim(-0.05, 1.05)

    if fit_result and fit_result.get("success"):
        t_pi = fit_result.get("optimal_duration", 0)

        # Overlay damped-sinusoid fit curve if available
        sinusoid = (fit_result or {}).get("_sinusoid_fit")
        if sinusoid is not None:
            t_shifted = sinusoid["t_shifted"]
            t_plot = t_shifted + duration_ns[0]  # shift back to original time axis
            ax.plot(t_plot, sinusoid["fitted_curve"], "r-", lw=1.5, alpha=0.9, label="Damped sinusoid fit")

        ax.axvline(t_pi, color="lime", ls="--", lw=1.5, alpha=0.9, label=f"t_π = {t_pi:.0f} ns")
        ax.legend(loc="upper right", fontsize=8)


def _plot_fft_ax(
    ax: "plt.Axes",
    qubit_name: str,
    fit_result: dict | None = None,
) -> None:
    """Plot FFT magnitude spectrum with peak fit on the given axes."""
    diag = (fit_result or {}).get("_fft_diag")
    if diag is None:
        ax.text(0.5, 0.5, "No FFT data", transform=ax.transAxes, ha="center")
        ax.set_title(f"{qubit_name} — FFT")
        return

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
        ax.axvline(f_rabi_us, color="lime", ls="--", lw=1, alpha=0.9, label=f"f_Rabi = {f_rabi_us:.1f} /μs")

    ax.legend(loc="upper right", fontsize=8)


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    ds_fit: xr.Dataset | None,
    qubits: List[Any],
    fit_results: dict,
) -> "plt.Figure":
    """Plot Rabi trace and FFT for each qubit.

    Layout (per qubit row):
    * Column 1 — Raw parity difference vs pulse duration with t_π marker.
    * Column 2 — FFT magnitude spectrum with peak fit overlay.
    """
    qubit_names = _get_qubit_names_from_ds(ds)
    if not qubit_names:
        fig, _ = plt.subplots(figsize=(6, 4))
        return fig

    n = len(qubit_names)
    ncol = 2
    fig, axes = plt.subplots(n, ncol, figsize=(6 * ncol, 4 * n), squeeze=False)

    for i, qname in enumerate(qubit_names):
        ax_trace, ax_fft = axes[i, 0], axes[i, 1]
        pdiff_var = f"pdiff_{qname}"
        fr = fit_results.get(qname, {})

        durations_ns = np.asarray(ds.pulse_duration.values, dtype=float)

        if pdiff_var not in ds.data_vars:
            ax_trace.text(0.5, 0.5, f"No data for {qname}", transform=ax_trace.transAxes, ha="center")
            ax_fft.text(0.5, 0.5, f"No data for {qname}", transform=ax_fft.transAxes, ha="center")
            continue

        pdiff = np.asarray(ds[pdiff_var].values, dtype=float)

        _plot_rabi_trace_ax(ax_trace, pdiff, durations_ns, qname, fit_result=fr)
        _plot_fft_ax(ax_fft, qname, fit_result=fr)

    fig.suptitle("Time Rabi (parity diff)")
    fig.tight_layout()
    return fig
