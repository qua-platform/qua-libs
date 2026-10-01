from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_swap_oscillations(
    ds_raw: xr.Dataset,
    qubit_pairs: Sequence,
    ds_fit: xr.Dataset | None = None,
    fit_results: dict[str, dict[str, Any]] | None = None,
) -> plt.Figure:
    """Plot 2D swap-oscillation heatmaps with optional 2π overlay.

    Creates one row per qubit pair with two columns: left for the control
    qubit measurement, right for the target qubit measurement.  When
    *ds_fit* is provided, the extracted 2π oscillation curve is overlaid
    on each heatmap.
    """
    n_pairs = len(qubit_pairs)
    fig, axes = plt.subplots(
        n_pairs, 2,
        figsize=(14, 5 * n_pairs),
        squeeze=False,
    )

    for i, qp in enumerate(qubit_pairs):
        for j, role in enumerate(["control", "target"]):
            ax = axes[i, j]
            key = f"state_{role}_{qp.name}"
            data = ds_raw[key]

            durations = data.coords["exchange_duration"].values
            amplitudes = data.coords["exchange_amplitude"].values

            im = ax.pcolormesh(
                durations,
                amplitudes,
                data.values,
                shading="auto",
                cmap="RdBu_r",
            )
            ax.set_xlim(durations[0], durations[-1])
            ax.set_ylim(amplitudes[0], amplitudes[-1])
            ax.set_xlabel("Exchange duration (ns)")
            ax.set_ylabel("Barrier gate voltage (V)")
            ax.set_title(f"{qp.name} — {role} qubit")
            fig.colorbar(im, ax=ax, label="P(1)")

            _overlay_t2pi(ax, qp.name, ds_fit, fit_results)

    fig.suptitle("Swap Oscillations", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def _overlay_t2pi(
    ax: plt.Axes,
    pair_name: str,
    ds_fit: xr.Dataset | None,
    fit_results: dict[str, dict[str, Any]] | None,
) -> None:
    """Overlay the extracted 2π curve, fitted model, and best-point marker."""
    if ds_fit is None:
        return

    t2pi_key = f"t_2pi_{pair_name}"
    valid_key = f"valid_{pair_name}"
    if t2pi_key not in ds_fit.data_vars or valid_key not in ds_fit.data_vars:
        return

    amplitudes = ds_fit.coords["exchange_amplitude"].values
    t_2pi = ds_fit[t2pi_key].values
    valid = ds_fit[valid_key].values.astype(bool)
    extracted = np.isfinite(t_2pi)
    below_threshold = extracted & ~valid

    if below_threshold.any():
        ax.plot(
            t_2pi[below_threshold],
            amplitudes[below_threshold],
            "o",
            color="red",
            ms=3,
            alpha=0.5,
            markeredgecolor="darkred",
            markeredgewidth=0.3,
            label="$T_{2\\pi}$ (low SNR)",
            zorder=3,
        )

    if not np.any(valid):
        ax.legend(loc="upper right", fontsize=7)
        return

    ax.plot(
        t_2pi[valid],
        amplitudes[valid],
        "o",
        color="lime",
        ms=4,
        markeredgecolor="black",
        markeredgewidth=0.5,
        label="$T_{2\\pi}$ (FFT)",
        zorder=5,
    )

    if fit_results is not None:
        r = fit_results.get(pair_name, {})
        m = r.get("exchange_decay_model", {})
        if m and r.get("model_fit_success"):
            v_fine = np.linspace(
                amplitudes[valid][0], amplitudes[valid][-1], 200
            )
            coeffs = m["coeffs"]
            t_model = np.polyval(coeffs, v_fine)
            deg = m.get("degree", len(coeffs) - 1)
            label_str = f"poly deg {deg}"
            ax.plot(
                t_model,
                v_fine,
                "-",
                color="lime",
                lw=2,
                alpha=0.9,
                label=label_str,
                zorder=4,
            )

        if r.get("success"):
            ax.plot(
                r["best_t_2pi"],
                r["best_amplitude"],
                "*",
                color="yellow",
                ms=14,
                markeredgecolor="black",
                markeredgewidth=0.8,
                label=f"best: V={r['best_amplitude']:.3f} V, T={r['best_t_2pi']:.0f} ns",
                zorder=6,
            )

    ax.legend(loc="upper right", fontsize=7)
