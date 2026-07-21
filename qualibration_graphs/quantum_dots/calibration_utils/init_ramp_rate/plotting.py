from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import xarray as xr


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
        avg_state = state.mean(dim="shot").values if "shot" in state.dims else state.values

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
    Q may be 2-D (shots × ramp_duration) — averaged over shots before plotting.
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
            i_vals = i_data.mean(dim="shot").values if "shot" in i_data.dims else i_data.values
            ax.plot(ramp_durations, i_vals, "o-", color="C0", label="I")

        if q_key in ds_raw:
            q_vals = np.asarray(ds_raw[q_key].values, dtype=float)
            if q_vals.ndim == 2:
                q_vals = q_vals.mean(axis=0)  # average over shots
            ax2 = ax.twinx()
            ax2.plot(ramp_durations, q_vals, "s--", color="C1", label="Q (mean)")
            ax2.set_ylabel("Average Q")
            lines2, labels2 = ax2.get_legend_handles_labels()
        else:
            lines2, labels2 = [], []

        lines1, labels1 = ax.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        ax.set_xlabel("Ramp duration (ns)")
        ax.set_ylabel("Average I")
        ax.set_title(qp_name)

    fig.suptitle("IQ signal vs initialisation ramp duration")
    fig.tight_layout()
    return fig


def plot_i_density_vs_ramp_duration(
    ds_raw: xr.Dataset,
    qubit_pair_names: list[str],
    *,
    n_grid: int = 200,
    scatter_alpha: float = 0.3,
    scatter_size: float = 4,
) -> plt.Figure:
    """2-D density of per-shot I values vs ramp duration with state-coloured scatter.

    For each ramp duration a Gaussian KDE is estimated over the shot distribution
    of I, displayed as a background image.  Individual shots are overlaid as a
    scatter plot, coloured by their boolean state assignment (0 = blue, 1 = red).

    ``I_{qp_name}`` and ``state_{qp_name}`` must have shape
    ``(n_shots, n_ramp_durations)``.
    """
    n_pairs = max(len(qubit_pair_names), 1)
    fig, axes = plt.subplots(1, n_pairs, figsize=(7 * n_pairs, 5), squeeze=False)
    axes = axes[0]

    state_colors = {0: "C0", 1: "C3"}

    for idx, qp_name in enumerate(qubit_pair_names):
        ax = axes[idx]
        i_key = f"I_{qp_name}"
        s_key = f"state_{qp_name}"

        if i_key not in ds_raw:
            ax.set_title(f"{qp_name} (no I data)")
            continue

        i_arr = np.asarray(ds_raw[i_key].values, dtype=float)
        ramp_durations = np.asarray(ds_raw["ramp_duration"].values)

        if i_arr.ndim == 1:
            ax.set_title(f"{qp_name} (I not per-shot)")
            continue

        # Ensure shape (n_shots, n_ramp)
        if i_arr.shape[0] == len(ramp_durations) and i_arr.shape[1] != len(ramp_durations):
            i_arr = i_arr.T

        i_min, i_max = i_arr.min(), i_arr.max()
        i_pad = 0.1 * (i_max - i_min) if i_max > i_min else 1e-6
        i_grid = np.linspace(i_min - i_pad, i_max + i_pad, n_grid)

        density = np.zeros((n_grid, len(ramp_durations)), dtype=float)
        for col, shots in enumerate(i_arr.T):
            finite = shots[np.isfinite(shots)]
            if finite.size < 2 or finite.std() < 1e-20:
                continue
            kde = gaussian_kde(finite)
            density[:, col] = kde(i_grid)

        im = ax.imshow(
            density,
            origin="lower",
            aspect="auto",
            extent=[ramp_durations[0], ramp_durations[-1], i_grid[0], i_grid[-1]],
            cmap="Greys",
        )
        fig.colorbar(im, ax=ax, label="Density")

        # # Scatter per-shot I coloured by state
        # if s_key in ds_raw:
        #     s_arr = np.asarray(ds_raw[s_key].values, dtype=int)
        #     if s_arr.shape[0] == len(ramp_durations) and s_arr.shape[1] != len(ramp_durations):
        #         s_arr = s_arr.T
        #     for state_val, color in state_colors.items():
        #         mask = s_arr == state_val
        #         x_scatter = np.broadcast_to(ramp_durations, i_arr.shape)[mask]
        #         y_scatter = i_arr[mask]
        #         ax.scatter(
        #             x_scatter,
        #             y_scatter,
        #             c=color,
        #             s=scatter_size,
        #             alpha=scatter_alpha,
        #             label=f"state {state_val}",
        #             linewidths=0,
        #         )
        #     ax.legend(markerscale=3, loc="upper right")

        ax.set_xlabel("Ramp duration (ns)")
        ax.set_ylabel("I")
        ax.set_title(qp_name)

    fig.suptitle("I distribution density vs ramp duration")
    fig.tight_layout()
    return fig


def plot_q_density_vs_ramp_duration(
    ds_raw: xr.Dataset,
    qubit_pair_names: list[str],
    *,
    n_grid: int = 200,
) -> plt.Figure:
    """2-D density plot of per-shot Q values vs initialisation ramp duration.

    For each ramp duration a Gaussian KDE is estimated over the shot distribution
    of Q.  The densities are stacked into an image (x = ramp duration, y = Q value)
    and displayed with ``imshow``.

    ``Q_{qp_name}`` must have shape ``(n_shots, n_ramp_durations)``.
    """
    n_pairs = max(len(qubit_pair_names), 1)
    fig, axes = plt.subplots(1, n_pairs, figsize=(7 * n_pairs, 5), squeeze=False)
    axes = axes[0]

    for idx, qp_name in enumerate(qubit_pair_names):
        ax = axes[idx]
        q_key = f"Q_{qp_name}"

        if q_key not in ds_raw:
            ax.set_title(f"{qp_name} (no Q data)")
            continue

        q_arr = np.asarray(ds_raw[q_key].values, dtype=float)
        ramp_durations = np.asarray(ds_raw["ramp_duration"].values)

        # Expect shape (n_shots, n_ramp_durations); transpose if needed.
        if q_arr.ndim == 1:
            ax.set_title(f"{qp_name} (Q not per-shot)")
            continue
        if q_arr.shape[0] == len(ramp_durations) and q_arr.shape[1] != len(ramp_durations):
            q_arr = q_arr.T  # make (n_shots, n_ramp_durations)

        q_min, q_max = q_arr.min(), q_arr.max()
        q_pad = 0.1 * (q_max - q_min) if q_max > q_min else 1e-6
        q_grid = np.linspace(q_min - q_pad, q_max + q_pad, n_grid)

        density = np.zeros((n_grid, len(ramp_durations)), dtype=float)
        for col, shots in enumerate(q_arr.T):
            finite = shots[np.isfinite(shots)]
            if finite.size < 2 or finite.std() < 1e-20:
                continue
            kde = gaussian_kde(finite)
            density[:, col] = kde(q_grid)

        im = ax.imshow(
            density,
            origin="lower",
            aspect="auto",
            extent=[ramp_durations[0], ramp_durations[-1], q_grid[0], q_grid[-1]],
            cmap="viridis",
        )
        fig.colorbar(im, ax=ax, label="Density")
        ax.set_xlabel("Ramp duration (ns)")
        ax.set_ylabel("Q")
        ax.set_title(qp_name)

    fig.suptitle("Q distribution density vs ramp duration")
    fig.tight_layout()
    return fig
