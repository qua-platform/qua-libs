"""Plotting utilities for fixed-amplitude geometric CZ cphase duration calibration.

Layout per pair (1 row × 4 columns):
  Col 0: raw parity I/Q signals vs exchange duration
  Col 1: raw I₁−I₀ and Q₁−Q₀ differences
  Col 2: Stark-free envelopes |Z₁−Z₀| and |Z₀+Z₁−2c|
  Col 3: extracted phases φ₀, φ₁, φ₁−φ₀−π with twin-axis cos(φ₁−φ₀−π)
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _analysis_signal_colorbar_label(analysis_signal: str) -> str:
    if analysis_signal == "E_p2_given_p1_0":
        return "E[p2 | p1=0]"
    if analysis_signal == "E_p2_given_p1_1":
        return "E[p2 | p1=1]"
    return analysis_signal


def _plot_raw_parity(
    ax: plt.Axes,
    data: xr.DataArray,
    durations_us: np.ndarray,
    *,
    signal_center: float,
) -> None:
    """Plot raw parity I and Q signals vs duration for both control states."""
    colors = {0: "C0", 1: "C1"}
    labels = {0: "|0⟩", 1: "|1⟩"}

    for ctrl in (0, 1):
        i_vals = data.sel(control_state=ctrl, analysis_axis=0).values.astype(float)
        q_vals = data.sel(control_state=ctrl, analysis_axis=1).values.astype(float)
        ax.plot(
            durations_us, i_vals,
            "-o", ms=3, color=colors[ctrl], lw=1.2,
            label=f"I ctrl {labels[ctrl]}",
        )
        ax.plot(
            durations_us, q_vals,
            "--s", ms=3, color=colors[ctrl], lw=1.2, alpha=0.7,
            label=f"Q ctrl {labels[ctrl]}",
        )

    ax.axhline(signal_center, color="0.5", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("Exchange duration (μs)")
    ax.set_ylabel(_analysis_signal_colorbar_label("E_p2_given_p1_0"))
    ax.legend(loc="best", fontsize=6, ncol=2)


def _plot_raw_differences(
    ax: plt.Axes,
    data: xr.DataArray,
    durations_us: np.ndarray,
    *,
    optimal_duration_ns: float | None = None,
) -> None:
    """Plot raw I₁−I₀ and Q₁−Q₀ vs duration (Stark fringes still visible)."""
    i0 = data.sel(control_state=0, analysis_axis=0).values.astype(float)
    i1 = data.sel(control_state=1, analysis_axis=0).values.astype(float)
    q0 = data.sel(control_state=0, analysis_axis=1).values.astype(float)
    q1 = data.sel(control_state=1, analysis_axis=1).values.astype(float)

    ax.plot(
        durations_us, i1 - i0,
        "-", color="C0", lw=1.2,
        label="I₁ − I₀",
    )
    ax.plot(
        durations_us, q1 - q0,
        "-", color="C1", lw=1.2,
        label="Q₁ − Q₀",
    )
    ax.axhline(0.0, color="0.5", ls=":", lw=0.8, alpha=0.6)

    if optimal_duration_ns is not None and optimal_duration_ns > 0:
        t_pi_us = optimal_duration_ns / 1e3
        ax.axvline(t_pi_us, color="red", ls="--", lw=1.2)

    ax.set_xlabel("Exchange duration (μs)")
    ax.set_ylabel("difference (a.u.)")
    ax.legend(loc="best", fontsize=6, ncol=2)


def _plot_difference_envelopes(
    ax: plt.Axes,
    data: xr.DataArray,
    durations_us: np.ndarray,
    signal_center: float,
    *,
    optimal_duration_ns: float | None = None,
    success: bool = False,
) -> None:
    """Plot Stark-free envelopes from I/Q control-state combinations.

    |Z₁−Z₀| = √((I₁−I₀)²+(Q₁−Q₀)²) = 2A·|cos(cphase/2)|
        → goes to 0 at the CZ point (cphase = π)
    |Z₀+Z₁−2c| = √((I₀+I₁−2c)²+(Q₀+Q₁−2c)²) = 2A·|sin(cphase/2)|
        → grows from 0, reaches maximum at the CZ point

    Both are completely free of Stark oscillations.
    """
    i0 = data.sel(control_state=0, analysis_axis=0).values.astype(float)
    i1 = data.sel(control_state=1, analysis_axis=0).values.astype(float)
    q0 = data.sel(control_state=0, analysis_axis=1).values.astype(float)
    q1 = data.sel(control_state=1, analysis_axis=1).values.astype(float)

    env_diff = np.sqrt((i1 - i0) ** 2 + (q1 - q0) ** 2)
    env_sum = np.sqrt(
        (i0 + i1 - 2 * signal_center) ** 2
        + (q0 + q1 - 2 * signal_center) ** 2
    )

    ax.plot(
        durations_us, env_diff,
        "-", color="C0", lw=1.8,
        label="|Z₁−Z₀| ∝ |cos(Δφ/2)|",
    )
    ax.plot(
        durations_us, env_sum,
        "-", color="C1", lw=1.8,
        label="|Z₀+Z₁−2c| ∝ |sin(Δφ/2)|",
    )

    if optimal_duration_ns is not None and optimal_duration_ns > 0:
        t_pi_us = optimal_duration_ns / 1e3
        ax.axvline(t_pi_us, color="red", ls="--", lw=1.2)
        if success:
            ax.plot(
                [t_pi_us], [0.0],
                "*", color="red", ms=13,
                label=f"t*={optimal_duration_ns:.0f} ns",
            )

    ax.set_xlabel("Exchange duration (μs)")
    ax.set_ylabel("envelope (a.u.)")
    ax.legend(loc="best", fontsize=6, ncol=2)


def _plot_phase_panel(
    ax: plt.Axes,
    durations_us: np.ndarray,
    phi0: np.ndarray,
    phi1: np.ndarray,
    dphi: np.ndarray,
    *,
    optimal_duration_ns: float | None = None,
    success: bool = False,
) -> None:
    """Plot φ₀, φ₁, conditional phase with twin-axis cos(φ₁−φ₀−π)."""
    ax.plot(
        durations_us, phi0,
        linestyle="none", marker="o", ms=3, color="C0", alpha=0.65,
        label="φ₀  |0⟩",
    )
    ax.plot(
        durations_us, phi1,
        linestyle="none", marker="s", ms=3, color="C1", alpha=0.65,
        label="φ₁  |1⟩",
    )
    ax.plot(
        durations_us, dphi,
        "-", color="k", lw=1.8, alpha=0.85,
        label="φ₁−φ₀−π (cphase)",
    )
    ax.axhline(np.pi, color="0.45", ls=":", lw=0.75, alpha=0.7)
    ax.set_ylabel("phase (rad)", color="k")
    ax.tick_params(axis="y", labelcolor="k")

    ax_cos = ax.twinx()
    cos_cphase = np.cos(dphi)
    ax_cos.plot(
        durations_us, cos_cphase,
        "--", color="C2", lw=1.2, alpha=0.9,
        label="cos(φ₁−φ₀−π)",
    )
    ax_cos.axhline(-1.0, color="C2", ls=":", lw=0.8, alpha=0.45)
    ax_cos.set_ylabel("cos(φ₁−φ₀−π)", color="C2")
    ax_cos.tick_params(axis="y", labelcolor="C2")
    ax_cos.set_ylim(-1.12, 1.12)

    if optimal_duration_ns is not None and optimal_duration_ns > 0:
        t_pi_us = optimal_duration_ns / 1e3
        ax.axvline(t_pi_us, color="red", ls="--", lw=1.2)
        if success:
            ax.plot(
                [t_pi_us], [np.pi],
                "*", color="red", ms=13,
                label=f"t*={optimal_duration_ns:.0f} ns",
            )

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_cos.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=6.5, ncol=2)

    ax.set_xlabel("Exchange duration (μs)")


def plot_raw_data_with_fit(
    ds_raw: xr.Dataset,
    ds_fit: xr.Dataset | None,
    qubit_pairs: list[Any],
    fit_results: dict[str, dict[str, Any]],
    *,
    analysis_signal: str = "E_p2_given_p1_0",
    quadrature_signal_center: float = 0.5,
) -> plt.Figure:
    """Create the fixed-amplitude geometric CZ duration calibration figure.

    Layout per pair (1 row × 4 columns):
      Col 0: raw parity I/Q signals vs duration
      Col 1: raw I₁−I₀ and Q₁−Q₀ differences
      Col 2: Stark-free envelopes |Z₁−Z₀| and |Z₀+Z₁−2c|
      Col 3: extracted phases φ₀, φ₁, φ₁−φ₀−π with twin-axis cos(φ₁−φ₀−π)
    """
    n_pairs = len(qubit_pairs)
    fig, axes = plt.subplots(
        n_pairs, 4,
        figsize=(28, max(4.5, 4.2 * n_pairs)),
        squeeze=False,
    )

    durations = ds_raw.coords["exchange_duration"].values
    durations_us = durations / 1e3
    center = float(quadrature_signal_center)

    for pair_idx, qp in enumerate(qubit_pairs):
        name = qp.name
        var_name = f"{analysis_signal}_{name}"
        ax_parity = axes[pair_idx, 0]
        ax_raw_diff = axes[pair_idx, 1]
        ax_env = axes[pair_idx, 2]
        ax_phase = axes[pair_idx, 3]

        if var_name not in ds_raw.data_vars:
            ax_parity.set_title(f"{name} — no data")
            ax_raw_diff.set_title(f"{name} — no data")
            ax_env.set_title(f"{name} — no data")
            ax_phase.set_title(f"{name} — no data")
            continue

        data_array = ds_raw[var_name]
        r = fit_results.get(name, {})

        _plot_raw_parity(ax_parity, data_array, durations_us, signal_center=center)
        ax_parity.set_title(f"{name} — raw parity I/Q")

        _plot_raw_differences(
            ax_raw_diff, data_array, durations_us,
            optimal_duration_ns=r.get("optimal_duration"),
        )
        ax_raw_diff.set_title(f"{name} — raw I/Q differences")

        _plot_difference_envelopes(
            ax_env, data_array, durations_us, center,
            optimal_duration_ns=r.get("optimal_duration"),
            success=r.get("success", False),
        )
        ax_env.set_title(f"{name} — Stark-free envelopes")

        cond_key = f"conditional_phase_{name}"
        if ds_fit is not None and cond_key in ds_fit.data_vars:
            phi0 = ds_fit[f"phi_ctrl_ground_{name}"].values
            phi1 = ds_fit[f"phi_ctrl_excited_{name}"].values
            dphi = ds_fit[cond_key].values
        else:
            i0 = data_array.sel(control_state=0, analysis_axis=0).values.astype(np.float64)
            q0 = data_array.sel(control_state=0, analysis_axis=1).values.astype(np.float64)
            i1 = data_array.sel(control_state=1, analysis_axis=0).values.astype(np.float64)
            q1 = data_array.sel(control_state=1, analysis_axis=1).values.astype(np.float64)
            phi0 = np.unwrap(np.arctan2(q0 - center, i0 - center))
            phi1 = np.unwrap(np.arctan2(q1 - center, i1 - center))
            z0 = (i0 - center) + 1j * (q0 - center)
            z1 = (i1 - center) + 1j * (q1 - center)
            dphi = np.unwrap(np.angle(-z1 * np.conj(z0)))

        _plot_phase_panel(
            ax_phase, durations_us, phi0, phi1, dphi,
            optimal_duration_ns=r.get("optimal_duration"),
            success=r.get("success", False),
        )
        ax_phase.set_title(f"{name} — branch phases & cond. phase")

    fig.suptitle("Geometric CZ Duration Calibration", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
