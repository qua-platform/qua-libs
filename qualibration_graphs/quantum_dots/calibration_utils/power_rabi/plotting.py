"""Plot power-Rabi parity difference: raw trace and FFT diagnostics."""

from __future__ import annotations

from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calibration_utils.power_rabi.analysis import FFT_FREQ_MIN, FFT_FREQ_MAX


def _get_qubit_names_from_ds(ds: xr.Dataset) -> List[str]:
    pdiff_vars = [v for v in ds.data_vars if v.startswith("pdiff_") and not v.endswith("_fit")]
    return [v.replace("pdiff_", "") for v in sorted(pdiff_vars)]


def _plot_rabi_trace_ax(
    ax: "plt.Axes",
    pdiff: np.ndarray,
    amps: np.ndarray,
    qubit_name: str,
    fit_result: dict | None = None,
) -> None:
    """Plot raw parity difference vs amplitude prefactor."""
    ax.plot(amps, pdiff, "b-", lw=1, alpha=0.8)
    ax.scatter(amps, pdiff, c="b", s=6, alpha=0.5, zorder=3)
    ax.set_xlabel("Amplitude prefactor")
    ax.set_ylabel("Parity difference")
    ax.set_title(f"{qubit_name} — Power Rabi")
    ax.set_ylim(-0.05, 1.05)

    if fit_result and fit_result.get("success"):
        a_pi = fit_result.get("opt_amp", 0)

        sinusoid = (fit_result or {}).get("_sinusoid_fit")
        if sinusoid is not None:
            a_shifted = sinusoid["a_shifted"]
            a_plot = a_shifted + amps[0]
            ax.plot(a_plot, sinusoid["fitted_curve"], "r-", lw=1.5, alpha=0.9, label="Damped sinusoid fit")

        ax.axvline(a_pi, color="lime", ls="--", lw=1.5, alpha=0.9, label=f"a_pi = {a_pi:.3f}")
        ax.legend(loc="upper right", fontsize=8)


def _plot_fft_ax(
    ax: "plt.Axes",
    qubit_name: str,
    fit_result: dict | None = None,
) -> None:
    """Plot FFT magnitude spectrum with peak fit."""
    diag = (fit_result or {}).get("_fft_diag")
    if diag is None:
        ax.text(0.5, 0.5, "No FFT data", transform=ax.transAxes, ha="center")
        ax.set_title(f"{qubit_name} — FFT")
        return

    freqs_fft = diag["fft_freqs"]
    magnitude = diag["fft_magnitude"]
    peak_curve = diag.get("peak_curve")

    mask = (freqs_fft >= FFT_FREQ_MIN) & (freqs_fft <= FFT_FREQ_MAX)
    f_plot = freqs_fft[mask]

    ax.plot(f_plot, magnitude[mask], "b-", lw=1, label="FFT")
    if peak_curve is not None:
        ax.plot(f_plot, peak_curve[mask], "r-", lw=1.5, label="Peak fit")

    ax.set_xlabel("Frequency (cycles / unit amp)")
    ax.set_ylabel("|FFT|")
    ax.set_title(f"{qubit_name} — FFT spectrum")
    ax.set_xlim(f_plot[0], f_plot[-1])

    if fit_result and fit_result.get("success"):
        omega = fit_result.get("rabi_frequency", 0)
        f_rabi = omega / (2.0 * np.pi)
        ax.axvline(f_rabi, color="lime", ls="--", lw=1, alpha=0.9, label=f"f = {f_rabi:.2f} c/u.a.")

    ax.legend(loc="upper right", fontsize=8)


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    ds_fit: xr.Dataset | None,
    qubits: List[Any],
    fit_results: dict,
) -> "plt.Figure":
    """Plot power-Rabi trace and FFT for each qubit.

    Layout (per qubit row):
    * Column 1 — Raw parity difference vs amplitude with a_π marker.
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

        amps = np.asarray(ds.amp_prefactor.values, dtype=float)

        if pdiff_var not in ds.data_vars:
            ax_trace.text(0.5, 0.5, f"No data for {qname}", transform=ax_trace.transAxes, ha="center")
            ax_fft.text(0.5, 0.5, f"No data for {qname}", transform=ax_fft.transAxes, ha="center")
            continue

        pdiff = np.asarray(ds[pdiff_var].values, dtype=float)

        _plot_rabi_trace_ax(ax_trace, pdiff, amps, qname, fit_result=fr)
        _plot_fft_ax(ax_fft, qname, fit_result=fr)

    fig.suptitle("Power Rabi (parity diff)")
    fig.tight_layout()
    return fig
