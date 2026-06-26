"""Plot cross-talk Rabi data: per-combo traces, FFTs and a summary matrix."""

from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calibration_utils.power_rabi.analysis import FFT_FREQ_MIN, FFT_FREQ_MAX
from calibration_utils.cross_talk_rabi.analysis import (
    _combo_names_from_ds,
    parse_combo_name,
)


def _plot_crosstalk_trace_ax(
    ax: "plt.Axes",
    trace: np.ndarray,
    amps: np.ndarray,
    pair_name: str,
    drive_name: str,
    analysis_signal: str,
    fit_result: dict | None = None,
) -> None:
    """Plot raw analysis trace vs drive-amplitude prefactor."""
    ax.plot(amps, trace, "b-", lw=1, alpha=0.8)
    ax.scatter(amps, trace, c="b", s=6, alpha=0.5, zorder=3)
    ax.set_xlabel("Drive amplitude prefactor")
    ax.set_ylabel(analysis_signal)
    ax.set_title(f"drive {drive_name} -> {pair_name}")
    ax.set_ylim(-0.05, 1.05)

    if fit_result:
        sinusoid = fit_result.get("_sinusoid_fit")
        if fit_result.get("fit_success") and sinusoid is not None:
            a_plot = sinusoid["a_shifted"] + amps[0]
            ax.plot(
                a_plot,
                sinusoid["fitted_curve"],
                "r-",
                lw=1.5,
                alpha=0.9,
                label="Damped sinusoid fit",
            )

        amplitude = fit_result.get("crosstalk_amplitude")
        contrast = fit_result.get("crosstalk_contrast")
        label_parts = []
        if contrast is not None and np.isfinite(contrast):
            label_parts.append(f"contrast = {contrast:.3f}")
        if amplitude is not None and np.isfinite(amplitude):
            label_parts.append(f"amp = {amplitude:.3f}")
        if label_parts:
            ax.plot([], [], " ", label=" | ".join(label_parts))
        ax.legend(loc="upper right", fontsize=8)


def _plot_fft_ax(
    ax: "plt.Axes",
    title: str,
    fit_result: dict | None = None,
) -> None:
    """Plot FFT magnitude spectrum with peak fit."""
    diag = (fit_result or {}).get("_fft_diag")
    if diag is None:
        ax.text(0.5, 0.5, "No FFT data", transform=ax.transAxes, ha="center")
        ax.set_title(f"{title} — FFT")
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
    ax.set_title(f"{title} — FFT")
    if f_plot.size:
        ax.set_xlim(f_plot[0], f_plot[-1])

    if fit_result and fit_result.get("fit_success"):
        f_rabi = fit_result.get("rabi_frequency", 0) / (2.0 * np.pi)
        ax.axvline(
            f_rabi,
            color="lime",
            ls="--",
            lw=1,
            alpha=0.9,
            label=f"f = {f_rabi:.2f} c/u.a.",
        )

    ax.legend(loc="upper right", fontsize=8)


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    ds_fit: xr.Dataset | None,
    fit_results: dict,
    analysis_signal: str = "E_p2_given_p1_0",
) -> "plt.Figure":
    """Plot the cross-talk Rabi trace and FFT for each (pair, drive) combo.

    One row per combination: column 1 is the raw trace vs drive amplitude with the
    cross-talk metrics, column 2 is the FFT magnitude spectrum.
    """
    combo_names = _combo_names_from_ds(ds, analysis_signal)
    if not combo_names:
        fig, _ = plt.subplots(figsize=(6, 4))
        return fig

    n = len(combo_names)
    ncol = 2
    fig, axes = plt.subplots(n, ncol, figsize=(6 * ncol, 4 * n), squeeze=False)

    amps = np.asarray(ds.amp_prefactor.values, dtype=float)

    for i, cname in enumerate(combo_names):
        pair_name, drive_name = parse_combo_name(cname)
        ax_trace, ax_fft = axes[i, 0], axes[i, 1]

        if f"{analysis_signal}_{cname}" in ds.data_vars:
            signal_var = f"{analysis_signal}_{cname}"
            y_signal_label = analysis_signal
        else:
            signal_var = f"p_{cname}"
            y_signal_label = "P(measure)"

        fr = fit_results.get(cname, {})

        if signal_var not in ds.data_vars:
            for ax in (ax_trace, ax_fft):
                ax.text(
                    0.5, 0.5, f"No data for {cname}", transform=ax.transAxes, ha="center"
                )
            continue

        trace = np.asarray(ds[signal_var].values, dtype=float)
        _plot_crosstalk_trace_ax(
            ax_trace, trace, amps, pair_name, drive_name, y_signal_label, fit_result=fr
        )
        _plot_fft_ax(ax_fft, f"drive {drive_name} -> {pair_name}", fit_result=fr)

    fig.suptitle(f"Cross-talk Rabi ({analysis_signal})")
    fig.tight_layout()
    return fig


def plot_crosstalk_matrix(matrix_data: Dict[str, Any]) -> "plt.Figure":
    """Plot the measured-pair x drive-qubit cross-talk matrix as a heatmap."""
    pairs: List[str] = matrix_data["pairs"]
    drives: List[str] = matrix_data["drives"]
    matrix = np.asarray(matrix_data["matrix"], dtype=float)
    metric = matrix_data.get("metric", "crosstalk")

    fig, ax = plt.subplots(
        figsize=(1.4 * max(len(drives), 1) + 3, 1.4 * max(len(pairs), 1) + 2)
    )
    im = ax.imshow(matrix, cmap="viridis", aspect="auto", origin="upper")

    ax.set_xticks(range(len(drives)))
    ax.set_xticklabels(drives)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs)
    ax.set_xlabel("Drive qubit")
    ax.set_ylabel("Measured pair")
    ax.set_title(f"Cross-talk matrix ({metric})")

    for i in range(len(pairs)):
        for j in range(len(drives)):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(
                    j, i, f"{value:.3f}", ha="center", va="center", color="w", fontsize=9
                )

    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    return fig
