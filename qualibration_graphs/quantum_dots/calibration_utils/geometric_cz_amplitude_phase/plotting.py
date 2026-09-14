"""Plotting utilities for the 2-D CZ amplitude × analysis-phase calibration.

Layout per pair (2 rows × 3 columns):
  Top-left:    raw parity S(ctrl=|0⟩) as a heatmap (amplitude × phase)
  Top-centre:  raw parity S(ctrl=|1⟩) as a heatmap (amplitude × phase)
  Top-right:   parity difference D = S₁ − S₀ as a heatmap (amplitude × phase)
  Bottom-left: D_avg = ⟨D⟩_θ vs exchange amplitude with V* marked
  Bottom-centre: S₀ and S₁ phase cuts at V* (1-D Ramsey curves)
  Bottom-right:  D phase cut at V* with zero-line
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    amplitudes: np.ndarray,
    phases_rad: np.ndarray,
    *,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "RdBu_r",
    optimal_amplitude: float | None = None,
) -> None:
    """Plot a 2-D heatmap (amplitude on Y, phase on X)."""
    extent = [phases_rad[0], phases_rad[-1], amplitudes[0], amplitudes[-1]]
    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax)
    if optimal_amplitude is not None and optimal_amplitude > 0:
        ax.axhline(optimal_amplitude, color="red", ls="--", lw=1.2, label=f"V*={optimal_amplitude:.4f} V")
        ax.legend(loc="upper right", fontsize=7)
    ax.set_xlabel("Analysis phase (rad)")
    ax.set_ylabel("Exchange amplitude (V)")
    ax.set_title(title, fontsize=9)
    xticks = np.linspace(0, 2 * np.pi, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])


def _plot_mean_abs_diff(
    ax: plt.Axes,
    amplitudes: np.ndarray,
    mean_abs_diff: np.ndarray,
    std_over_phase: np.ndarray | None = None,
    *,
    optimal_amplitude: float | None = None,
    success: bool = False,
) -> None:
    """Plot ⟨|D|⟩_θ and std_θ(D) vs exchange amplitude."""
    ax.plot(amplitudes, mean_abs_diff, "-", color="C0", lw=1.8, label="⟨|D|⟩_θ")
    if std_over_phase is not None:
        ax.plot(amplitudes, std_over_phase, "--", color="C1", lw=1.4, label="std_θ(D)")
    ax.axhline(0.0, color="0.5", ls=":", lw=0.8, alpha=0.6)
    if optimal_amplitude is not None and optimal_amplitude > 0:
        ax.axvline(optimal_amplitude, color="red", ls="--", lw=1.2)
        if success:
            opt_val = float(np.interp(optimal_amplitude, amplitudes, mean_abs_diff))
            ax.plot(
                [optimal_amplitude],
                [opt_val],
                "*",
                color="red",
                ms=13,
                label=f"V*={optimal_amplitude:.4f} V",
            )
    ax.set_xlabel("Exchange amplitude (V)")
    ax.set_ylabel("⟨|D|⟩_θ  /  std_θ(D)")
    ax.legend(loc="best", fontsize=7)


def _plot_phase_cuts(
    ax: plt.Axes,
    s0_cut: np.ndarray,
    s1_cut: np.ndarray,
    phases_rad: np.ndarray,
    *,
    amplitude_label: str,
) -> None:
    """Plot 1-D Ramsey curves for both control states at the selected amplitude."""
    ax.plot(phases_rad, s0_cut, "-o", ms=3, color="C0", lw=1.2, label="ctrl |0⟩")
    ax.plot(phases_rad, s1_cut, "-s", ms=3, color="C1", lw=1.2, label="ctrl |1⟩")
    ax.set_xlabel("Analysis phase (rad)")
    ax.set_ylabel("Parity signal")
    ax.set_title(f"Phase cuts at V* = {amplitude_label}", fontsize=9)
    ax.legend(loc="best", fontsize=7)
    xticks = np.linspace(0, 2 * np.pi, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])


def _plot_diff_cut(
    ax: plt.Axes,
    diff_cut: np.ndarray,
    phases_rad: np.ndarray,
    *,
    amplitude_label: str,
) -> None:
    """Plot the parity difference D at the selected amplitude vs phase."""
    ax.plot(phases_rad, diff_cut, "-", color="k", lw=1.8, label="D = S₁−S₀")
    ax.axhline(0.0, color="0.5", ls=":", lw=0.8)
    ax.set_xlabel("Analysis phase (rad)")
    ax.set_ylabel("S(ctrl |1⟩)−S(ctrl |0⟩)")
    ax.set_title(f"Difference cut at V* = {amplitude_label}", fontsize=9)
    ax.legend(loc="best", fontsize=7)
    xticks = np.linspace(0, 2 * np.pi, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])


def plot_raw_data_with_fit(
    ds_raw: xr.Dataset,
    ds_fit: xr.Dataset | None,
    qubit_pairs: list[Any],
    fit_results: dict[str, dict[str, Any]],
    *,
    analysis_signal: str = "E_p2_given_p1_0",
) -> plt.Figure:
    """Create the 2-D amplitude-phase CZ calibration figure.

    2 rows × 3 columns per qubit pair:
      Row 0: heatmaps of S(ctrl=0), S(ctrl=1), and D = S₁−S₀
      Row 1: D_avg vs amplitude, phase-cut of both S at V*, phase-cut of D at V*
    """
    n_pairs = len(qubit_pairs)
    fig, axes = plt.subplots(
        n_pairs * 2,
        3,
        figsize=(18, max(6, 5.5 * n_pairs)),
        squeeze=False,
    )

    amplitudes = ds_raw.coords["exchange_amplitude"].values.astype(np.float64)
    phases_rad = ds_raw.coords["analysis_phase"].values.astype(np.float64)

    for pair_idx, qp in enumerate(qubit_pairs):
        name = qp.name
        var_name = f"{analysis_signal}_{name}"
        r = fit_results.get(name, {})
        opt_amp = r.get("optimal_amplitude")
        success = r.get("success", False)

        row0 = pair_idx * 2
        row1 = row0 + 1

        if var_name not in ds_raw.data_vars:
            for col in range(3):
                axes[row0, col].set_title(f"{name} — no data")
                axes[row1, col].set_visible(False)
            continue

        data = ds_raw[var_name]
        s0 = data.sel(control_state=0).values.astype(np.float64)  # (amp, phase)
        s1 = data.sel(control_state=1).values.astype(np.float64)
        diff_2d = s1 - s0

        sym = max(np.abs(diff_2d).max(), 1e-9)

        _heatmap(
            axes[row0, 0], s0, amplitudes, phases_rad,
            title=f"{name} — S(ctrl |0⟩)",
            cmap="viridis",
            optimal_amplitude=opt_amp,
        )
        _heatmap(
            axes[row0, 1], s1, amplitudes, phases_rad,
            title=f"{name} — S(ctrl |1⟩)",
            cmap="viridis",
            optimal_amplitude=opt_amp,
        )
        _heatmap(
            axes[row0, 2], diff_2d, amplitudes, phases_rad,
            title=f"{name} — D = S₁−S₀",
            vmin=-sym, vmax=sym,
            cmap="RdBu_r",
            optimal_amplitude=opt_amp,
        )

        # Mean absolute difference from ds_fit if available, else compute inline
        mad_key = f"mean_abs_diff_{name}"
        if ds_fit is not None and mad_key in ds_fit.data_vars:
            mean_abs_diff = ds_fit[mad_key].values.astype(np.float64)
        else:
            mean_abs_diff = np.mean(np.abs(diff_2d), axis=-1)

        std_over_phase = np.std(diff_2d, axis=-1)
        _plot_mean_abs_diff(
            axes[row1, 0], amplitudes, mean_abs_diff, std_over_phase,
            optimal_amplitude=opt_amp,
            success=success,
        )
        axes[row1, 0].set_title(f"{name} — ⟨|D|⟩_θ vs amplitude", fontsize=9)

        if opt_amp is not None and np.isfinite(opt_amp):
            opt_idx = int(np.argmin(np.abs(amplitudes - opt_amp)))
            amp_label = f"{opt_amp:.4f} V"
            _plot_phase_cuts(
                axes[row1, 1], s0[opt_idx], s1[opt_idx], phases_rad,
                amplitude_label=amp_label,
            )
            _plot_diff_cut(
                axes[row1, 2], diff_2d[opt_idx], phases_rad,
                amplitude_label=amp_label,
            )
        else:
            axes[row1, 1].set_title(f"{name} — no optimal amplitude found")
            axes[row1, 2].set_visible(False)

    fig.suptitle(
        "Geometric CZ Amplitude–Phase Calibration",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig