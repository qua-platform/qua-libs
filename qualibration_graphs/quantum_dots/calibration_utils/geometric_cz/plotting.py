"""Geometric CZ gate calibration plotting — quadrature readout.

Visualises the four raw experiment colormaps (2 control states × 2 analysis
axes), raw I/Q control-state differences, Stark-free envelope colormaps,
the 2D conditional-phase landscape (amplitude × duration) with t_pi(V) overlay,
and the 1D slice at the target duration used to select the CZ operating point.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib.pyplot as plt


_QUADRATURE_LABELS = {
    (0, 0): "ctrl |0⟩, X90 (I)",
    (0, 1): "ctrl |0⟩, Y90 (Q)",
    (1, 0): "ctrl |1⟩, X90 (I)",
    (1, 1): "ctrl |1⟩, Y90 (Q)",
}


def _mark_cz_point(
    ax: plt.Axes,
    target_duration_us: float | None,
    optimal_amplitude: float | None,
    success: bool,
) -> None:
    """Overlay the CZ operating point (V*, t*) on any 2D subplot."""
    if not success or target_duration_us is None or optimal_amplitude is None:
        return
    if not np.isfinite(target_duration_us) or not np.isfinite(optimal_amplitude):
        return
    ax.plot(
        [target_duration_us],
        [optimal_amplitude],
        "*",
        color="red",
        ms=12,
        zorder=5,
        markeredgecolor="k",
        markeredgewidth=0.5,
    )


def _plot_raw_quadrature_colormaps(
    axes: list[list[plt.Axes]],
    data_array,
    amplitudes: np.ndarray,
    durations_us: np.ndarray,
    *,
    target_duration_us: float | None = None,
    optimal_amplitude: float | None = None,
    success: bool = False,
) -> None:
    """Plot all 4 raw experiment colormaps in a 2×2 grid.

    Layout:
        [0,0] ctrl|0⟩ X90 (I)   [0,1] ctrl|0⟩ Y90 (Q)
        [1,0] ctrl|1⟩ X90 (I)   [1,1] ctrl|1⟩ Y90 (Q)
    """
    for row, ctrl in enumerate([0, 1]):
        for col, axis in enumerate([0, 1]):
            ax = axes[row][col]
            sig = data_array.sel(
                control_state=ctrl, analysis_axis=axis
            ).values.astype(np.float64)
            im = ax.pcolormesh(
                durations_us,
                amplitudes,
                sig,
                shading="auto",
                cmap="RdBu_r",
            )
            ax.set_ylabel("Amplitude (V)")
            ax.set_xlabel("Duration (μs)")
            ax.set_title(_QUADRATURE_LABELS[(ctrl, axis)])
            plt.colorbar(im, ax=ax, label="parity")
            _mark_cz_point(ax, target_duration_us, optimal_amplitude, success)


def _plot_envelope_colormaps(
    axes: list[plt.Axes],
    data_array,
    amplitudes: np.ndarray,
    durations_us: np.ndarray,
    signal_center: float,
    *,
    t_pi: np.ndarray | None = None,
    target_duration_us: float | None = None,
    optimal_amplitude: float | None = None,
    success: bool = False,
) -> None:
    """Plot Stark-free envelope colormaps.

    |Z₁−Z₀| = 2A·|cos(cphase/2)| → 0 at CZ  (left)
    |Z₀+Z₁−2c| = 2A·|sin(cphase/2)| → max at CZ  (right)
    """
    i0 = data_array.sel(control_state=0, analysis_axis=0).values.astype(np.float64)
    q0 = data_array.sel(control_state=0, analysis_axis=1).values.astype(np.float64)
    i1 = data_array.sel(control_state=1, analysis_axis=0).values.astype(np.float64)
    q1 = data_array.sel(control_state=1, analysis_axis=1).values.astype(np.float64)

    env_diff = np.sqrt((i1 - i0) ** 2 + (q1 - q0) ** 2)
    env_sum = np.sqrt(
        (i0 + i1 - 2 * signal_center) ** 2
        + (q0 + q1 - 2 * signal_center) ** 2
    )

    for col, (env, title, cmap) in enumerate([
        (env_diff, "|Z₁−Z₀|  ∝ |cos(Δφ/2)|  (→0 at CZ)", "viridis_r"),
        (env_sum, "|Z₀+Z₁−2c|  ∝ |sin(Δφ/2)|  (max at CZ)", "viridis"),
    ]):
        ax = axes[col]
        im = ax.pcolormesh(
            durations_us,
            amplitudes,
            env,
            shading="auto",
            cmap=cmap,
        )
        ax.set_ylabel("Amplitude (V)")
        ax.set_xlabel("Duration (μs)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="envelope (a.u.)")

        if t_pi is not None:
            finite = np.isfinite(t_pi)
            if np.any(finite):
                ax.plot(
                    t_pi[finite] / 1e3,
                    amplitudes[finite],
                    "w-o",
                    ms=3,
                    lw=1.5,
                    alpha=0.85,
                )

        _mark_cz_point(ax, target_duration_us, optimal_amplitude, success)


def _plot_raw_difference_colormaps(
    axes: list[plt.Axes],
    data_array,
    amplitudes: np.ndarray,
    durations_us: np.ndarray,
    signal_center: float,
    *,
    t_pi: np.ndarray | None = None,
    target_duration_us: float | None = None,
    optimal_amplitude: float | None = None,
    success: bool = False,
) -> None:
    """Plot raw I/Q control-state difference colormaps.

    Left:  I₁−I₀  (Stark fringes × cos(Δφ/2) envelope)
    Right: Q₁−Q₀  (Stark fringes × sin(Δφ/2) envelope)

    Fringes vanish at the CZ point where Δφ = π.
    """
    i0 = data_array.sel(control_state=0, analysis_axis=0).values.astype(np.float64)
    q0 = data_array.sel(control_state=0, analysis_axis=1).values.astype(np.float64)
    i1 = data_array.sel(control_state=1, analysis_axis=0).values.astype(np.float64)
    q1 = data_array.sel(control_state=1, analysis_axis=1).values.astype(np.float64)

    for col, (diff, label) in enumerate([
        (i1 - i0, "I₁ − I₀"),
        (q1 - q0, "Q₁ − Q₀"),
    ]):
        ax = axes[col]
        vlim = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
        im = ax.pcolormesh(
            durations_us,
            amplitudes,
            diff,
            shading="auto",
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
        )
        ax.set_ylabel("Amplitude (V)")
        ax.set_xlabel("Duration (μs)")
        ax.set_title(label)
        plt.colorbar(im, ax=ax, label=label)

        if t_pi is not None:
            finite = np.isfinite(t_pi)
            if np.any(finite):
                ax.plot(
                    t_pi[finite] / 1e3,
                    amplitudes[finite],
                    "w-o",
                    ms=3,
                    lw=1.5,
                    alpha=0.85,
                )

        _mark_cz_point(ax, target_duration_us, optimal_amplitude, success)


def _plot_conditional_phase_colormap(
    ax: plt.Axes,
    cond_phase: np.ndarray,
    amplitudes: np.ndarray,
    durations_us: np.ndarray,
    t_pi: np.ndarray | None = None,
    optimal_amplitude: float | None = None,
    target_duration_us: float | None = None,
    success: bool = False,
) -> None:
    """2D colormap of conditional phase with t_pi(V) overlay."""
    im = ax.pcolormesh(
        durations_us,
        amplitudes,
        cond_phase,
        shading="auto",
        cmap="twilight_shifted",
    )
    plt.colorbar(im, ax=ax, label="φ₁−φ₀−π (rad)")
    ax.set_ylabel("Amplitude (V)")
    ax.set_xlabel("Duration (μs)")
    ax.set_title("Conditional phase φ₁−φ₀−π")

    if t_pi is not None:
        finite = np.isfinite(t_pi)
        if np.any(finite):
            ax.plot(
                t_pi[finite] / 1e3,
                amplitudes[finite],
                "w-o",
                ms=3,
                lw=1.5,
                alpha=0.85,
                label="t(π cphase)",
            )

    if target_duration_us is not None:
        ax.axvline(
            target_duration_us,
            color="yellow",
            ls="--",
            lw=1.2,
            alpha=0.8,
            label=f"t* = {target_duration_us * 1e3:.0f} ns",
        )

    if optimal_amplitude is not None and success:
        ax.axhline(optimal_amplitude, color="red", ls="--", lw=1.0, alpha=0.7)
        if target_duration_us is not None:
            ax.plot(
                [target_duration_us],
                [optimal_amplitude],
                "*",
                color="red",
                ms=14,
                zorder=5,
                label=f"CZ: V*={optimal_amplitude:.4f} V",
            )

    ax.legend(fontsize=7, loc="upper right")


def _plot_cos_conditional_phase_colormap(
    ax: plt.Axes,
    cos_cond_phase: np.ndarray,
    amplitudes: np.ndarray,
    durations_us: np.ndarray,
    t_pi: np.ndarray | None = None,
    optimal_amplitude: float | None = None,
    target_duration_us: float | None = None,
    success: bool = False,
) -> None:
    """2D colormap of cos(conditional phase) with t_pi(V) overlay."""
    im = ax.pcolormesh(
        durations_us,
        amplitudes,
        cos_cond_phase,
        shading="auto",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )
    plt.colorbar(im, ax=ax, label="cos(φ₁−φ₀−π)")
    ax.set_ylabel("Amplitude (V)")
    ax.set_xlabel("Duration (μs)")
    ax.set_title("cos(conditional phase)")

    if t_pi is not None:
        finite = np.isfinite(t_pi)
        if np.any(finite):
            ax.plot(
                t_pi[finite] / 1e3,
                amplitudes[finite],
                "w-o",
                ms=3,
                lw=1.5,
                alpha=0.85,
                label="t(π cphase)",
            )

    if target_duration_us is not None:
        ax.axvline(
            target_duration_us,
            color="yellow",
            ls="--",
            lw=1.2,
            alpha=0.8,
            label=f"t* = {target_duration_us * 1e3:.0f} ns",
        )

    if optimal_amplitude is not None and success:
        ax.axhline(optimal_amplitude, color="red", ls="--", lw=1.0, alpha=0.7)
        if target_duration_us is not None:
            ax.plot(
                [target_duration_us],
                [optimal_amplitude],
                "*",
                color="red",
                ms=14,
                zorder=5,
                label=f"CZ: V*={optimal_amplitude:.4f} V",
            )

    ax.legend(fontsize=7, loc="upper right")


def _plot_phase_vs_amplitude(
    ax: plt.Axes,
    amplitudes: np.ndarray,
    cond_at_target: np.ndarray,
    optimal_amplitude: float | None = None,
    target_dur_ns: float = 0.0,
    success: bool = False,
) -> None:
    """1D slice: conditional phase at target duration vs amplitude."""
    ax.plot(amplitudes, cond_at_target, "k-", lw=1.5, label="φ₁−φ₀−π")
    ax.axhline(np.pi, color="0.45", ls=":", lw=0.75, alpha=0.7, label="π")
    ax.set_xlabel("Amplitude (V)")
    ax.set_ylabel("Conditional phase (rad)")
    ax.set_title(f"Conditional phase at t = {target_dur_ns:.0f} ns")

    if optimal_amplitude is not None and success:
        ax.axvline(optimal_amplitude, color="red", ls="--", lw=1.0, alpha=0.7)
        ax.plot(
            [optimal_amplitude],
            [np.pi],
            "*",
            color="red",
            ms=13,
            label=f"V* = {optimal_amplitude:.4f} V",
        )

    ax.legend(fontsize=7)


def plot_raw_data_with_fit(
    ds_raw,
    ds_fit,
    qubit_pairs: list[Any],
    fit_results: dict[str, dict[str, Any]],
    *,
    analysis_signal: str = "E_p2_given_p1_0",
    quadrature_signal_center: float = 0.5,
) -> plt.Figure:
    """Create the multi-panel geometric CZ calibration figure.

    Layout per qubit pair (6 rows × 2 columns):
      Row 1: ctrl|0⟩ X90 (I)                ctrl|0⟩ Y90 (Q)
      Row 2: ctrl|1⟩ X90 (I)                ctrl|1⟩ Y90 (Q)
      Row 3: I₁−I₀ raw difference           Q₁−Q₀ raw difference
      Row 4: |Z₁−Z₀| envelope (→0 at CZ)   |Z₀+Z₁−2c| envelope (max at CZ)
      Row 5: Conditional phase 2D colormap   cos(conditional phase) 2D colormap
      Row 6: phase vs amplitude 1D slice     (empty)
    """
    n_pairs = len(qubit_pairs)
    n_rows_per_pair = 6
    n_cols = 2

    fig, axes = plt.subplots(
        n_pairs * n_rows_per_pair,
        n_cols,
        figsize=(14, 4 * n_pairs * n_rows_per_pair),
        squeeze=False,
    )

    amplitudes = ds_raw.coords["exchange_amplitude"].values
    durations = ds_raw.coords["exchange_duration"].values
    durations_us = durations / 1e3

    for pair_idx, qp in enumerate(qubit_pairs):
        base_row = pair_idx * n_rows_per_pair
        var_name = f"{analysis_signal}_{qp.name}"
        r = fit_results.get(qp.name, {})

        if var_name not in ds_raw.data_vars:
            for c in range(n_cols):
                for rr in range(n_rows_per_pair):
                    axes[base_row + rr, c].set_title(f"{qp.name} — no data")
            continue

        data_array = ds_raw[var_name]

        # --- Extract fit result for this pair ---
        cond_key = f"conditional_phase_{qp.name}"
        tpi_key = f"t_pi_cphase_{qp.name}"
        cond_target_key = f"cond_phase_at_target_{qp.name}"

        cond_phase = ds_fit[cond_key].values if (ds_fit is not None and cond_key in ds_fit.data_vars) else None
        t_pi = ds_fit[tpi_key].values if (ds_fit is not None and tpi_key in ds_fit.data_vars) else None

        target_dur_ns = r.get("optimal_duration", 0.0)
        target_dur_us = target_dur_ns / 1e3 if np.isfinite(target_dur_ns) else None
        opt_amp = r.get("optimal_amplitude")
        success = r.get("success", False)

        # --- Rows 1-2: all 4 raw experiment colormaps ---
        raw_axes = [
            [axes[base_row, 0], axes[base_row, 1]],
            [axes[base_row + 1, 0], axes[base_row + 1, 1]],
        ]
        _plot_raw_quadrature_colormaps(
            raw_axes,
            data_array,
            amplitudes,
            durations_us,
            target_duration_us=target_dur_us,
            optimal_amplitude=opt_amp,
            success=success,
        )

        # --- Row 3: raw I/Q difference colormaps ---
        diff_axes = [axes[base_row + 2, 0], axes[base_row + 2, 1]]
        _plot_raw_difference_colormaps(
            diff_axes,
            data_array,
            amplitudes,
            durations_us,
            float(quadrature_signal_center),
            t_pi=t_pi,
            target_duration_us=target_dur_us,
            optimal_amplitude=opt_amp,
            success=success,
        )

        # --- Row 4: Stark-free envelope colormaps ---
        env_axes = [axes[base_row + 3, 0], axes[base_row + 3, 1]]
        _plot_envelope_colormaps(
            env_axes,
            data_array,
            amplitudes,
            durations_us,
            float(quadrature_signal_center),
            t_pi=t_pi,
            target_duration_us=target_dur_us,
            optimal_amplitude=opt_amp,
            success=success,
        )

        # --- Row 5, left: conditional phase 2D colormap ---
        if cond_phase is not None:
            _plot_conditional_phase_colormap(
                axes[base_row + 4, 0],
                cond_phase,
                amplitudes,
                durations_us,
                t_pi=t_pi,
                optimal_amplitude=opt_amp,
                target_duration_us=target_dur_us,
                success=success,
            )
        else:
            axes[base_row + 4, 0].set_title(f"{qp.name} — no fit data")

        # --- Row 5, right: cos(conditional phase) 2D colormap ---
        cos_cond_key = f"cos_conditional_phase_{qp.name}"
        cos_cond_phase = (
            ds_fit[cos_cond_key].values
            if (ds_fit is not None and cos_cond_key in ds_fit.data_vars)
            else (np.cos(cond_phase) if cond_phase is not None else None)
        )
        if cos_cond_phase is not None:
            _plot_cos_conditional_phase_colormap(
                axes[base_row + 4, 1],
                cos_cond_phase,
                amplitudes,
                durations_us,
                t_pi=t_pi,
                optimal_amplitude=opt_amp,
                target_duration_us=target_dur_us,
                success=success,
            )
        else:
            axes[base_row + 4, 1].set_title(f"{qp.name} — no cos(cphase) data")

        # --- Row 6, left: phase at target duration vs amplitude ---
        cond_at_target = (
            ds_fit[cond_target_key].values
            if (ds_fit is not None and cond_target_key in ds_fit.data_vars)
            else None
        )
        if cond_at_target is not None:
            _plot_phase_vs_amplitude(
                axes[base_row + 5, 0],
                amplitudes,
                cond_at_target,
                optimal_amplitude=opt_amp,
                target_dur_ns=target_dur_ns,
                success=success,
            )
        else:
            axes[base_row + 5, 0].set_title(f"{qp.name} — no target slice")

        axes[base_row + 5, 1].set_visible(False)

    fig.tight_layout()
    return fig
