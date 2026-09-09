from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calibration_utils.common_utils.plot_style import (
    apply_qubit_pair_outcome_style,
    qubit_pair_success,
)


def plot_all(
    ds_fit: xr.Dataset,
    qubit_pair_names: list[str],
    *,
    fit_results: Optional[Dict] = None,
    plot_fft: bool = False,
) -> dict[str, plt.Figure]:
    """Standard node plotting API returning a figure dict."""
    figures: dict[str, plt.Figure] = {}
    figures["avg_state_vs_ramp_duration"] = plot_avg_state_vs_ramp_duration(
        ds_fit, qubit_pair_names, fit_results=fit_results
    )
    figures["iq_vs_ramp_duration"] = plot_iq_vs_ramp_duration(ds_fit, qubit_pair_names, fit_results=fit_results)
    if plot_fft:
        figures["fft_vs_ramp_duration"] = plot_fft_vs_ramp_duration(
            ds_fit, qubit_pair_names, fit_results=fit_results
        )
    return figures


def _compute_fft_1d(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return positive FFT frequencies and magnitudes for a uniformly sampled 1D trace."""
    dx_ns = float(x_values[1] - x_values[0]) if len(x_values) > 1 else 1.0
    dx_us = dx_ns * 1e-3
    freqs = np.fft.rfftfreq(len(x_values), d=dx_us)[1:]
    spectrum = np.abs(np.fft.rfft(y_values - np.mean(y_values)))[1:]
    return freqs, spectrum


def plot_avg_state_vs_ramp_duration(
    ds_raw: xr.Dataset,
    qubit_pair_names: list[str],
    fit_results: Optional[Dict] = None,
) -> plt.Figure:
    """Plot average state assignment as a function of initialisation ramp duration.

    One subplot per qubit pair.  If *fit_results* is provided the identified
    optimum is highlighted with a dashed line and star marker.
    """
    n_pairs = max(len(qubit_pair_names), 1)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 4), squeeze=False)
    axes = axes[0]

    for idx, qp_name in enumerate(qubit_pair_names):
        ax = axes[idx]
        ramp_durations = ds_raw["ramp_duration"].values
        avg_state = ds_raw.state.sel(qubit_pair=qp_name, drop=True).transpose("ramp_duration").values

        ax.plot(ramp_durations, avg_state, "o-", label="avg state assignment")

        if fit_results and qp_name in fit_results:
            r = fit_results[qp_name]
            if r["success"]:
                ax.axvline(
                    r["optimal_ramp_duration"],
                    color="r",
                    linestyle="--",
                    alpha=0.7,
                    label=f"optimum = {r['optimal_ramp_duration']} ns",
                )
                ax.plot(
                    r["optimal_ramp_duration"],
                    r["optimal_avg_state"],
                    "r*",
                    markersize=15,
                )

        ax.set_xlabel("Ramp duration (ns)")
        ax.set_ylabel("Average state assignment")
        apply_qubit_pair_outcome_style(
            ax,
            qp_name,
            qubit_pair_success(fit_results, qp_name),
            subtitle="Average state vs ramp duration",
        )
        ax.set_ylim(-0.05, 1.05)
        ax.legend()

    fig.suptitle("Initialization ramp-duration calibration")
    fig.tight_layout()
    return fig


def plot_iq_vs_ramp_duration(
    ds_raw: xr.Dataset,
    qubit_pair_names: list[str],
    *,
    fit_results: Optional[Dict] = None,
) -> plt.Figure:
    """Plot average I and Q signal as a function of initialization ramp duration.

    One subplot per qubit pair; I on the left y-axis, Q on the right y-axis.
    """
    n_pairs = max(len(qubit_pair_names), 1)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 4), squeeze=False)
    axes = axes[0]

    for idx, qp_name in enumerate(qubit_pair_names):
        ax = axes[idx]
        ramp_durations = ds_raw["ramp_duration"].values

        if "I" in ds_raw:
            i_vals = ds_raw.I.sel(qubit_pair=qp_name, drop=True).transpose("ramp_duration").values
            ax.plot(ramp_durations, i_vals, "o-", color="C0", label="I")

        if "Q" in ds_raw:
            q_vals = ds_raw.Q.sel(qubit_pair=qp_name, drop=True).transpose("ramp_duration").values
            ax2 = ax.twinx()
            ax2.plot(ramp_durations, q_vals, "s--", color="C1", label="Q (mean)")
            ax2.set_ylabel("Average Q")
            lines2, labels2 = ax2.get_legend_handles_labels()
        else:
            lines2, labels2 = [], []

        lines1, labels1 = ax.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2)

        ax.set_xlabel("Ramp duration (ns)")
        ax.set_ylabel("Average I")
        apply_qubit_pair_outcome_style(
            ax,
            qp_name,
            qubit_pair_success(fit_results, qp_name),
            subtitle="Average IQ vs ramp duration",
        )

    fig.suptitle("IQ signal vs initialization ramp duration")
    fig.tight_layout(w_pad=3.0)
    return fig


def plot_fft_vs_ramp_duration(
    ds_raw: xr.Dataset,
    qubit_pair_names: list[str],
    *,
    fit_results: Optional[Dict] = None,
) -> plt.Figure:
    """Plot FFT spectra of average state assignment vs initialization ramp duration."""
    n_pairs = max(len(qubit_pair_names), 1)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 4), squeeze=False)
    axes = axes[0]

    for idx, qp_name in enumerate(qubit_pair_names):
        ax = axes[idx]
        ramp_durations = ds_raw["ramp_duration"].values
        avg_state = ds_raw.state.sel(qubit_pair=qp_name, drop=True).transpose("ramp_duration").values
        freqs, fft_mag = _compute_fft_1d(ramp_durations, avg_state)

        if len(freqs) > 0:
            ax.plot(freqs, fft_mag, "o-", label="FFT(state)")

        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("|FFT|")
        apply_qubit_pair_outcome_style(
            ax,
            qp_name,
            qubit_pair_success(fit_results, qp_name),
            subtitle="FFT(state)",
        )
        if len(freqs) > 0:
            ax.legend()

    fig.suptitle("FFT of average state vs initialization ramp duration")
    fig.tight_layout()
    return fig
