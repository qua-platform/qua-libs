from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_all(
    ds_fit: xr.Dataset,
    qubit_pair_names: list[str],
    *,
    fit_results: Optional[Dict] = None,
) -> dict[str, plt.Figure]:
    """Standard node plotting API returning a figure dict."""
    figures: dict[str, plt.Figure] = {}
    figures["avg_state_vs_ramp_duration"] = plot_avg_state_vs_ramp_duration(
        ds_fit, qubit_pair_names, fit_results=fit_results
    )
    figures["iq_vs_ramp_duration"] = plot_iq_vs_ramp_duration(ds_fit, qubit_pair_names)
    return figures


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
        state = ds_raw[f"state_{qp_name}"]
        avg_state = state.values

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
        ax.set_title(qp_name)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()

    fig.suptitle("Initialisation ramp rate calibration")
    fig.tight_layout()
    return fig


def plot_iq_vs_ramp_duration(
    ds_raw: xr.Dataset,
    qubit_pair_names: list[str],
) -> plt.Figure:
    """Plot average I and Q signal as a function of initialisation ramp duration.

    One subplot per qubit pair; I on the left y-axis, Q on the right y-axis.
    """
    n_pairs = max(len(qubit_pair_names), 1)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 4), squeeze=False)
    axes = axes[0]

    for idx, qp_name in enumerate(qubit_pair_names):
        ax = axes[idx]
        ramp_durations = ds_raw["ramp_duration"].values

        i_key = f"I_{qp_name}"
        q_key = f"Q_{qp_name}"

        if i_key in ds_raw:
            i_data = ds_raw[i_key]
            i_vals = i_data.values
            ax.plot(ramp_durations, i_vals, "o-", color="C0", label="I")

        if q_key in ds_raw:
            q_data = ds_raw[q_key]
            q_vals = q_data.values
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
        ax.set_title(qp_name)

    fig.suptitle("IQ signal vs initialisation ramp duration")
    fig.tight_layout(w_pad=3.0)
    return fig
