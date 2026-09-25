"""Plotting utilities for fixed-duration geometric CZ cphase amplitude calibration.

Layout per pair (1 row × 2 columns):
  Left:   raw parity I/Q signals vs exchange amplitude
  Right:  extracted phases φ₀, φ₁, sum φ₀+φ₁, diff φ₁−φ₀ with twin-axis cos(φ₁−φ₀−π)
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
    amplitudes: np.ndarray,
    *,
    signal_center: float,
) -> None:
    """Plot raw parity I and Q signals vs amplitude for both control states."""
    colors = {0: "C0", 1: "C1"}
    labels = {0: "|0⟩", 1: "|1⟩"}

    for ctrl in (0, 1):
        i_vals = data.sel(control_state=ctrl, analysis_axis=0).values.astype(float)
        q_vals = data.sel(control_state=ctrl, analysis_axis=1).values.astype(float)
        ax.plot(
            amplitudes, i_vals,
            "-o", ms=3, color=colors[ctrl], lw=1.2,
            label=f"I ctrl {labels[ctrl]}",
        )
        ax.plot(
            amplitudes, q_vals,
            "--s", ms=3, color=colors[ctrl], lw=1.2, alpha=0.7,
            label=f"Q ctrl {labels[ctrl]}",
        )

    ax.axhline(signal_center, color="0.5", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("Exchange amplitude (V)")
    ax.set_ylabel(_analysis_signal_colorbar_label("E_p2_given_p1_0"))
    ax.legend(loc="best", fontsize=6, ncol=2)


def _plot_difference_envelopes(
    ax: plt.Axes,
    data: xr.DataArray,
    amplitudes: np.ndarray,
    signal_center: float,
    *,
    optimal_amplitude: float | None = None,
    success: bool = False,
) -> None:
    """Plot Stark-free envelopes from I/Q control-state combinations.

    |Z₁−Z₀| = 2A·|cos(cphase/2)| → goes to 0 at the CZ point
    |Z₀+Z₁−2c| = 2A·|sin(cphase/2)| → grows from 0, max at CZ
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
        amplitudes, env_diff,
        "-", color="C0", lw=1.8,
        label="|Z₁−Z₀| ∝ |cos(Δφ/2)|",
    )
    ax.plot(
        amplitudes, env_sum,
        "-", color="C1", lw=1.8,
        label="|Z₀+Z₁−2c| ∝ |sin(Δφ/2)|",
    )

    if optimal_amplitude is not None and optimal_amplitude > 0:
        ax.axvline(optimal_amplitude, color="red", ls="--", lw=1.2)
        if success:
            ax.plot(
                [optimal_amplitude], [0.0],
                "*", color="red", ms=13,
                label=f"V*={optimal_amplitude:.4f} V",
            )

    ax.set_xlabel("Exchange amplitude (V)")
    ax.set_ylabel("envelope (a.u.)")
    ax.legend(loc="best", fontsize=6, ncol=2)


def _plot_phase_panel(
    ax: plt.Axes,
    amplitudes: np.ndarray,
    phi0: np.ndarray,
    phi1: np.ndarray,
    dphi: np.ndarray,
    *,
    optimal_amplitude: float | None = None,
    success: bool = False,
) -> None:
    """Plot φ₀, φ₁, φ₀+φ₁ (sum), φ₁−φ₀ (diff) with twin-axis cos(φ₁−φ₀−π)."""
    ax.plot(
        amplitudes, phi0, marker="o", ms=3, color="C0", alpha=0.65,
        label="φ₀  |0⟩",
    )
    ax.plot(
        amplitudes, phi1, marker="s", ms=3, color="C1", alpha=0.65,
        label="φ₁  |1⟩",
    )
    phi_sum = phi0 + phi1
    phi_diff = phi1 - phi0
    ax.plot(
        amplitudes, phi_sum, marker="^", ms=3, color="C3", alpha=0.85,
        label="φ₀+φ₁ (sum)",
    )
    ax.plot(
        amplitudes, phi_diff, marker="d", ms=3, color="k", alpha=1.0,
        label="φ₁−φ₀ (diff)",
    )
    ax.set_ylabel("phase (rad)", color="k")
    ax.tick_params(axis="y", labelcolor="k")

    # ax_cos = ax.twinx()
    # cos_cphase = np.cos(dphi)
    # cos_sum = np.cos(phi_sum)
    # ax_cos.plot(
    #     amplitudes, cos_cphase,
    #     "--", color="C2", lw=1.2, alpha=0.9,
    #     label="cos(φ₁−φ₀−π)",
    # )
    # ax_cos.plot(
    #     amplitudes, cos_sum,
    #     "--", color="C3", lw=1.2, alpha=0.9,
    #     label="cos(φ₀+φ₁)",
    # )
    # ax_cos.axhline(-1.0, color="C2", ls=":", lw=0.8, alpha=0.45)
    # ax_cos.set_ylabel("cos", color="C2")
    # ax_cos.tick_params(axis="y", labelcolor="C2")
    # ax_cos.set_ylim(-1.12, 1.12)

    if optimal_amplitude is not None and optimal_amplitude > 0:
        ax.axvline(optimal_amplitude, color="red", ls="--", lw=1.2)
        if success:
            ax.plot(
                [optimal_amplitude], [np.pi],
                "*", color="red", ms=13,
                label=f"V*={optimal_amplitude:.4f} V",
            )

    h1, l1 = ax.get_legend_handles_labels()
    # h2, l2 = ax_cos.get_legend_handles_labels()
    ax.legend(h1, l1, loc="best", fontsize=6.5, ncol=2)

    ax.set_xlabel("Exchange amplitude (V)")



def _plot_wrap_panel(
    ax: plt.Axes,
    amplitudes: np.ndarray,
    phi0: np.ndarray,
    phi1: np.ndarray,
    dphi: np.ndarray,
    *,
    optimal_amplitude: float | None = None,
    success: bool = False,
) -> None:
    """Plot cos(φ₀), cos(φ₁), cos(φ₀+φ₁), cos(φ₁−φ₀) vs amplitude."""
    ax.plot(
        amplitudes, np.cos(phi0), marker="o", ms=3, color="C0", alpha=0.65,
        label="cos φ₀  |0⟩",
    )
    ax.plot(
        amplitudes, np.cos(phi1), marker="s", ms=3, color="C1", alpha=0.65,
        label="cos φ₁  |1⟩",
    )
    phi_sum = np.cos(phi0 + phi1)
    phi_diff = np.cos(phi1 - phi0)
    ax.plot(
        amplitudes, phi_sum, marker="^", ms=3, color="C3", alpha=0.85,
        label="cos(φ₀+φ₁)",
    )
    ax.plot(
        amplitudes, phi_diff, marker="d", ms=3, color="k", alpha=1.0,
        label="cos(φ₁−φ₀)",
    )
    ax.set_ylabel("cos(phase)", color="k")
    ax.tick_params(axis="y", labelcolor="k")

    if optimal_amplitude is not None and optimal_amplitude > 0:
        ax.axvline(optimal_amplitude, color="red", ls="--", lw=1.2)
        if success:
            ax.plot(
                [optimal_amplitude], [-1.0],
                "*", color="red", ms=13,
                label=f"V*={optimal_amplitude:.4f} V",
            )

    ax.legend(loc="best", fontsize=6.5, ncol=2)
    ax.set_xlabel("Exchange amplitude (V)")


def plot_raw_data_with_fit(
    ds_raw: xr.Dataset,
    ds_fit: xr.Dataset | None,
    qubit_pairs: list[Any],
    fit_results: dict[str, dict[str, Any]],
    *,
    analysis_signal: str = "E_p2_given_p1_0",
    quadrature_signal_center: float = 0.5,
) -> plt.Figure:
    """Create the fixed-duration geometric CZ amplitude calibration figure.

    Layout per pair (1 row × 2 columns):
      Left:   raw parity I/Q signals vs amplitude
      Right:  extracted phases φ₀, φ₁, sum, diff with twin-axis cos(φ₁−φ₀−π)
    """
    n_pairs = len(qubit_pairs)
    fig, axes = plt.subplots(
        n_pairs, 3,
        figsize=(14, max(4.5, 4.2 * n_pairs)),
        squeeze=False,
    )

    amplitudes = ds_raw.coords["exchange_amplitude"].values
    center = float(quadrature_signal_center)

    for pair_idx, qp in enumerate(qubit_pairs):
        name = qp.name
        var_name = f"{analysis_signal}_{name}"
        ax_parity = axes[pair_idx, 0]
        ax_phase = axes[pair_idx, 1]
        ax_wrap = axes[pair_idx, 2]

        if var_name not in ds_raw.data_vars:
            ax_parity.set_title(f"{name} — no data")
            ax_phase.set_title(f"{name} — no data")
            continue

        data_array = ds_raw[var_name]
        r = fit_results.get(name, {})

        use_quadrature = "analysis_axis" in data_array.dims

        if use_quadrature:
            _plot_raw_parity(ax_parity, data_array, amplitudes, signal_center=center)
            ax_parity.set_title(f"{name} — raw parity I/Q")

        cond_key = f"conditional_phase_{name}"
        if ds_fit is not None and cond_key in ds_fit.data_vars:
            phi0 = ds_fit[f"phi_ctrl_ground_{name}"].values
            phi1 = ds_fit[f"phi_ctrl_excited_{name}"].values
            dphi = ds_fit[cond_key].values
        elif use_quadrature:
            i0 = data_array.sel(control_state=0, analysis_axis=0).values.astype(np.float64)
            q0 = data_array.sel(control_state=0, analysis_axis=1).values.astype(np.float64)
            i1 = data_array.sel(control_state=1, analysis_axis=0).values.astype(np.float64)
            q1 = data_array.sel(control_state=1, analysis_axis=1).values.astype(np.float64)
            phi0 = np.unwrap(np.arctan2(q0 - center, i0 - center))
            phi1 = np.unwrap(np.arctan2(q1 - center, i1 - center))
            z0 = (i0 - center) + 1j * (q0 - center)
            z1 = (i1 - center) + 1j * (q1 - center)
            dphi = np.unwrap(np.angle(-z1 * np.conj(z0)))
        else:
            phi0 = None
            phi1 = None
            dphi = None

        if phi0 is not None and phi1 is not None and dphi is not None:
            _plot_phase_panel(
                ax_phase, amplitudes, phi0, phi1, dphi,
                optimal_amplitude=r.get("optimal_amplitude"),
                success=r.get("success", False),
            )
            ax_phase.set_title(f"{name} — branch phases & cond. phase")
            _plot_wrap_panel(
                ax_wrap, amplitudes, phi0, phi1, dphi,
                optimal_amplitude=r.get("optimal_amplitude"),
                success=r.get("success", False),
            )
            ax_wrap.set_title(f"{name} — cos(phase)")
        else:
            ax_phase.set_title(f"{name} — no phase data")

    fig.suptitle("Geometric CZ Amplitude Calibration", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
