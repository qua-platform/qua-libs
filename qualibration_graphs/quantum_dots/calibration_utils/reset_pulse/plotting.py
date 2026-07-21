from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .analysis import analyze_reset_pulse_maps


def _plot_heatmap(fig, ax, x, y, z, title, cbar_label, cmap, vmin=None, vmax=None):
    im = ax.pcolormesh(x, y, z.T, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel("Reset pulse detuning [MHz]")
    ax.set_ylabel("Reset pulse amplitude scale")
    ax.set_title(title)


def plot_2d_summary(
    ds_raw: xr.Dataset,
    qubit_names: list[str],
    fit_results: Optional[Dict] = None,
    plot_init_i: bool = True,
) -> Dict[str, plt.Figure]:
    """Return init, raw pulse/no-pulse, and diff figures.

    Args:
        ds_raw: Raw fetched dataset.
        qubit_names: Qubits to plot.
        fit_results: Optional fit markers for diff maps.
        plot_init_i: If False, hide the initialised-I row in the init figure.
    """
    fit_results = fit_results or {}
    _, _, map_results = analyze_reset_pulse_maps(ds_raw, qubit_names)

    detuning_mhz = ds_raw["frequency_detuning"].values / 1e6
    amplitude = ds_raw["amplitude_scale"].values
    n_q = max(len(qubit_names), 1)

    # 1) Initialised maps (averaged over pulse/no-pulse condition)
    n_init_rows = 2 if plot_init_i else 1
    fig_init, axes_init = plt.subplots(
        n_init_rows, n_q, figsize=(6 * n_q, 5 * n_init_rows), squeeze=False
    )
    for idx, q_name in enumerate(qubit_names):
        ax_state = axes_init[0, idx]
        _plot_heatmap(
            fig_init,
            ax_state,
            detuning_mhz,
            amplitude,
            map_results[q_name]["init_state_avg"],
            f"{q_name} — initialised state (avg)",
            "State",
            "RdBu_r",
            vmin=0,
            vmax=1,
        )
        if plot_init_i:
            ax_i = axes_init[1, idx]
            _plot_heatmap(
                fig_init,
                ax_i,
                detuning_mhz,
                amplitude,
                map_results[q_name]["init_i_avg"],
                f"{q_name} — initialised I (avg)",
                "I",
                "viridis",
            )
    if plot_init_i:
        fig_init.suptitle("Initialised maps (averaged over pulse/no-pulse)")
    else:
        fig_init.suptitle("Initialised state maps (averaged over pulse/no-pulse)")
    fig_init.tight_layout()

    # 2) Raw state/I maps (pulse vs no pulse)
    fig_raw, axes_raw = plt.subplots(2, 2 * n_q, figsize=(12 * n_q, 10), squeeze=False)
    for idx, q_name in enumerate(qubit_names):
        _plot_heatmap(
            fig_raw, axes_raw[0, 2 * idx], detuning_mhz, amplitude,
            map_results[q_name]["state_no"], f"{q_name} — state (no pulse)", "State", "RdBu_r", 0, 1
        )
        _plot_heatmap(
            fig_raw, axes_raw[0, 2 * idx + 1], detuning_mhz, amplitude,
            map_results[q_name]["state_yes"], f"{q_name} — state (with pulse)", "State", "RdBu_r", 0, 1
        )
        _plot_heatmap(
            fig_raw, axes_raw[1, 2 * idx], detuning_mhz, amplitude,
            map_results[q_name]["i_no"], f"{q_name} — I (no pulse)", "I", "viridis"
        )
        _plot_heatmap(
            fig_raw, axes_raw[1, 2 * idx + 1], detuning_mhz, amplitude,
            map_results[q_name]["i_yes"], f"{q_name} — I (with pulse)", "I", "viridis"
        )
    fig_raw.suptitle("Raw pulse vs no-pulse maps")
    fig_raw.tight_layout()

    # 3) Diff maps
    fig_diff, axes_diff = plt.subplots(2, n_q, figsize=(6 * n_q, 10), squeeze=False)
    for idx, q_name in enumerate(qubit_names):
        state_diff = map_results[q_name]["state_diff"]
        i_diff = map_results[q_name]["i_diff"]
        state_lim = max(float(np.nanmax(np.abs(state_diff))), 1e-12)
        i_lim = max(float(np.nanmax(np.abs(i_diff))), 1e-12)

        _plot_heatmap(
            fig_diff, axes_diff[0, idx], detuning_mhz, amplitude,
            state_diff, f"{q_name} — state diff", "State diff", "RdBu", -state_lim, state_lim
        )
        _plot_heatmap(
            fig_diff, axes_diff[1, idx], detuning_mhz, amplitude,
            i_diff, f"{q_name} — I diff", "I diff", "RdBu", -i_lim, i_lim
        )

        if q_name in fit_results and fit_results[q_name].get("success", False):
            fit = fit_results[q_name]
            axes_diff[0, idx].plot(
                fit["optimal_frequency_detuning_hz"] / 1e6,
                fit["optimal_amplitude_scale"],
                "k*",
                markersize=14,
                markeredgecolor="white",
                markeredgewidth=1.0,
            )
    fig_diff.suptitle("Pulse - no-pulse differences")
    fig_diff.tight_layout()

    return {
        "initialised_state": fig_init,
        "raw_maps": fig_raw,
        "diff_maps": fig_diff,
    }
