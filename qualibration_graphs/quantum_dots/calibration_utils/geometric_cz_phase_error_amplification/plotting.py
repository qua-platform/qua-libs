"""Plotting for the 2-D CZ phase error-amplification calibration.

Layout per pair — 4 rows × 3 columns:
  Row 0: D heatmap, S(ctrl=0), S(ctrl=1) for the target-qubit phase sweep
  Row 1: ⟨|D|⟩_θ vs N (target), phase cuts at N*, difference cut at N*
  Row 2: D heatmap, S(cond=0), S(cond=1) for the control-qubit phase sweep
  Row 3: ⟨|D|⟩_θ vs N (control), phase cuts at N* (ctrl), diff cut (ctrl)

Rows 2–3 are omitted if no control-sweep data is present.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# ── Low-level panel helpers ────────────────────────────────────────────────────

def _heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    gate_counts: np.ndarray,
    phases_rad: np.ndarray,
    *,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "RdBu_r",
    peak_gates: int | None = None,
    phase_comp_rad: float | None = None,
) -> None:
    """2-D heatmap with num_cphase_gates on Y and analysis_phase on X."""
    extent = [phases_rad[0], phases_rad[-1], gate_counts[0], gate_counts[-1]]
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
    if peak_gates is not None:
        ax.axhline(peak_gates, color="red", ls="--", lw=1.2, label=f"N*={peak_gates}")
        ax.legend(loc="upper right", fontsize=7)
    if phase_comp_rad is not None:
        # show the fitted phase-offset (positive shift) as a vertical line
        phase_offset_rad = (-phase_comp_rad * 2 * np.pi) % (2 * np.pi)
        ax.axvline(
            phase_offset_rad, color="orange", ls=":", lw=1.2,
            label=f"φ_acc={phase_offset_rad:.2f} rad",
        )
        ax.legend(loc="upper right", fontsize=7)
    ax.set_xlabel("Analysis phase (rad)")
    ax.set_ylabel("Number of CZ gates")
    ax.set_title(title, fontsize=9)
    xticks = np.linspace(0, 2 * np.pi, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax.set_yticks(gate_counts[::max(1, len(gate_counts) // 6)])


def _plot_mean_abs_diff(
    ax: plt.Axes,
    gate_counts: np.ndarray,
    mean_abs_diff: np.ndarray,
    *,
    peak_gates: int | None = None,
    success: bool = False,
    label: str = "⟨|D|⟩_θ",
) -> None:
    ax.plot(gate_counts, mean_abs_diff, "-o", ms=5, color="C0", lw=1.8, label=label)
    ax.axhline(0.0, color="0.5", ls=":", lw=0.8, alpha=0.6)
    if peak_gates is not None:
        ax.axvline(peak_gates, color="red", ls="--", lw=1.2)
        if success:
            peak_val = float(np.interp(peak_gates, gate_counts, mean_abs_diff))
            ax.plot(
                [peak_gates], [peak_val], "*", color="red", ms=13,
                label=f"N*={peak_gates}",
            )
    ax.set_xlabel("Number of CZ gates")
    ax.set_ylabel("⟨|S(cond|1⟩)−S(cond|0⟩)|⟩_θ")
    ax.legend(loc="best", fontsize=7)


def _plot_phase_cuts(
    ax: plt.Axes,
    s0_cut: np.ndarray,
    s1_cut: np.ndarray,
    phases_rad: np.ndarray,
    *,
    gate_label: str,
    cond0_label: str = "cond |0⟩",
    cond1_label: str = "cond |1⟩",
    phase_comp_rad: float | None = None,
) -> None:
    ax.plot(phases_rad, s0_cut, "-o", ms=3, color="C0", lw=1.2, label=cond0_label)
    ax.plot(phases_rad, s1_cut, "-s", ms=3, color="C1", lw=1.2, label=cond1_label)
    if phase_comp_rad is not None:
        phi_acc = (-phase_comp_rad * 2 * np.pi) % (2 * np.pi)
        ax.axvline(phi_acc, color="orange", ls=":", lw=1.2, label=f"φ_acc={phi_acc:.2f}")
    ax.set_xlabel("Analysis phase (rad)")
    ax.set_ylabel("Qubit signal")
    ax.set_title(f"Phase cuts at N* = {gate_label}", fontsize=9)
    ax.legend(loc="best", fontsize=7)
    xticks = np.linspace(0, 2 * np.pi, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])


def _plot_diff_cut(
    ax: plt.Axes,
    diff_cut: np.ndarray,
    phases_rad: np.ndarray,
    *,
    gate_label: str,
    phase_comp_turns: float | None = None,
) -> None:
    ax.plot(phases_rad, diff_cut, "-", color="k", lw=1.8, label="D = S₁−S₀")
    ax.axhline(0.0, color="0.5", ls=":", lw=0.8)
    if phase_comp_turns is not None and np.isfinite(phase_comp_turns):
        phi_acc = (-phase_comp_turns * 2 * np.pi) % (2 * np.pi)
        ax.axvline(
            phi_acc, color="orange", ls=":", lw=1.4,
            label=f"φ_acc={phi_acc:.2f} rad\ncomp={phase_comp_turns:.4f} turns",
        )
    ax.set_xlabel("Analysis phase (rad)")
    ax.set_ylabel("S(cond|1⟩)−S(cond|0⟩)")
    ax.set_title(f"Difference cut at N* = {gate_label}", fontsize=9)
    ax.legend(loc="best", fontsize=7)
    xticks = np.linspace(0, 2 * np.pi, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])


# ── Per-pair sweep plotter ─────────────────────────────────────────────────────

def _plot_std_over_n(
    ax: plt.Axes,
    phases_rad: np.ndarray,
    std_over_n: np.ndarray,
    mean_over_n: np.ndarray,
    *,
    phase_comp_turns: float | None,
) -> None:
    """Plot std_N(D) and mean_N(D) vs analysis phase θ."""
    ax.plot(phases_rad, std_over_n, "-", color="C0", lw=1.8, label="std_N(D)")
    ax.plot(phases_rad, np.abs(mean_over_n), ":", color="C1", lw=1.4, label="|mean_N(D)|")
    ax.axhline(0.0, color="0.5", ls=":", lw=0.8, alpha=0.6)
    if phase_comp_turns is not None and np.isfinite(phase_comp_turns):
        phi_min = (-phase_comp_turns * 2 * np.pi) % (2 * np.pi)
        ax.axvline(
            phi_min, color="orange", ls="--", lw=1.4,
            label=f"φ_min={phi_min:.2f} rad\n({phase_comp_turns:.4f} turns/gate)",
        )
    ax.set_xlabel("Analysis phase (rad)")
    ax.set_ylabel("std / |mean| over N")
    ax.legend(loc="best", fontsize=7)
    xticks = np.linspace(0, 2 * np.pi, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])


def _plot_sweep_rows(
    axes_row0: np.ndarray,
    axes_row1: np.ndarray,
    s0: np.ndarray,
    s1: np.ndarray,
    gate_counts: np.ndarray,
    phases_rad: np.ndarray,
    mean_abs_diff: np.ndarray,
    std_over_n: np.ndarray | None,
    mean_over_n: np.ndarray | None,
    *,
    name: str,
    qubit_label: str,
    peak_gates: int | None,
    success: bool,
    phase_comp_turns: float | None,
    cond0_label: str,
    cond1_label: str,
) -> None:
    """Fill two rows of axes for one Ramsey sweep (target or control).

    Row 0: D heatmap, S(cond=0) heatmap, S(cond=1) heatmap.
    Row 1: ⟨|D|⟩_θ vs N,  std_N(D) vs θ (with compensation marker),  D cut at N*.
    """
    diff_2d = s1 - s0
    sym = max(np.abs(diff_2d).max(), 1e-9)

    _heatmap(
        axes_row0[0], diff_2d, gate_counts, phases_rad,
        title=f"{name} {qubit_label} — D = S₁−S₀",
        vmin=-sym, vmax=sym, cmap="RdBu_r",
        peak_gates=peak_gates,
    )
    _heatmap(
        axes_row0[1], s0, gate_counts, phases_rad,
        title=f"{name} {qubit_label} — S({cond0_label})",
        cmap="viridis", peak_gates=peak_gates,
    )
    _heatmap(
        axes_row0[2], s1, gate_counts, phases_rad,
        title=f"{name} {qubit_label} — S({cond1_label})",
        cmap="viridis", peak_gates=peak_gates,
    )

    _plot_mean_abs_diff(
        axes_row1[0], gate_counts, mean_abs_diff,
        peak_gates=peak_gates, success=success,
    )
    axes_row1[0].set_title(f"{name} {qubit_label} — ⟨|D|⟩_θ vs N gates", fontsize=9)

    # Centre panel: std_N(D) vs θ with phase compensation marker.
    if std_over_n is not None and mean_over_n is not None:
        _plot_std_over_n(
            axes_row1[1], phases_rad, std_over_n, mean_over_n,
            phase_comp_turns=phase_comp_turns,
        )
        axes_row1[1].set_title(
            f"{name} {qubit_label} — std_N(D) vs θ (comp={phase_comp_turns:.4f} turns)"
            if (phase_comp_turns is not None and np.isfinite(phase_comp_turns))
            else f"{name} {qubit_label} — std_N(D) vs θ",
            fontsize=9,
        )
    else:
        axes_row1[1].set_visible(False)

    # Right panel: D cut at N*.
    if peak_gates is not None and np.isfinite(peak_gates):
        peak_idx = int(np.argmin(np.abs(gate_counts - peak_gates)))
        _plot_diff_cut(
            axes_row1[2], diff_2d[peak_idx], phases_rad,
            gate_label=str(int(peak_gates)), phase_comp_turns=phase_comp_turns,
        )
    else:
        axes_row1[2].set_visible(False)


# ── Public API ─────────────────────────────────────────────────────────────────

def plot_raw_data_with_fit(
    ds_raw: xr.Dataset,
    ds_fit: xr.Dataset | None,
    qubit_pairs: list[Any],
    fit_results: dict[str, dict[str, Any]],
    *,
    analysis_signal: str = "E_p2_given_p1_0",
) -> plt.Figure:
    """Create the phase error-amplification figure.

    Per qubit pair:
      Rows 0–1: target qubit phase sweep (always present)
      Rows 2–3: control qubit phase sweep (present when data available)
    """
    n_pairs = len(qubit_pairs)

    # Determine how many rows we need per pair.
    rows_per_pair = []
    for qp in qubit_pairs:
        ctrl_var = f"{analysis_signal}_ctrl_{qp.name}"
        rows_per_pair.append(4 if ctrl_var in ds_raw.data_vars else 2)

    total_rows = sum(rows_per_pair)
    fig, axes = plt.subplots(
        total_rows, 3,
        figsize=(18, max(6, 5.0 * total_rows // 2)),
        squeeze=False,
    )

    gate_counts = ds_raw.coords["num_cphase_gates"].values.astype(np.int64)
    phases_rad = ds_raw.coords["analysis_phase"].values.astype(np.float64)

    row_offset = 0
    for pair_idx, qp in enumerate(qubit_pairs):
        name = qp.name
        n_rows = rows_per_pair[pair_idx]
        r = fit_results.get(name, {})

        target_var = f"{analysis_signal}_{name}"
        ctrl_var = f"{analysis_signal}_ctrl_{name}"

        # ── Target sweep (rows 0–1 of this pair) ──────────────────────
        r0, r1 = row_offset, row_offset + 1
        if target_var in ds_raw.data_vars:
            data = ds_raw[target_var]
            s0 = data.sel(control_state=0).values.astype(np.float64)
            s1 = data.sel(control_state=1).values.astype(np.float64)
            diff_2d_t = s1 - s0

            mad_key = f"mean_abs_diff_{name}"
            mad = (
                ds_fit[mad_key].values.astype(np.float64)
                if ds_fit is not None and mad_key in ds_fit.data_vars
                else np.mean(np.abs(diff_2d_t), axis=-1)
            )
            std_key = f"std_over_n_{name}"
            std_n = (
                ds_fit[std_key].values.astype(np.float64)
                if ds_fit is not None and std_key in ds_fit.data_vars
                else np.std(diff_2d_t, axis=0)
            )
            mean_key = f"mean_over_n_{name}"
            mean_n = (
                ds_fit[mean_key].values.astype(np.float64)
                if ds_fit is not None and mean_key in ds_fit.data_vars
                else np.mean(diff_2d_t, axis=0)
            )

            _plot_sweep_rows(
                axes[r0], axes[r1], s0, s1, gate_counts, phases_rad,
                mad, std_n, mean_n,
                name=name, qubit_label="(target sweep)",
                peak_gates=r.get("peak_num_gates"),
                success=r.get("success_target", False),
                phase_comp_turns=r.get("phase_compensation_target"),
                cond0_label="ctrl |0⟩", cond1_label="ctrl |1⟩",
            )
        else:
            for col in range(3):
                axes[r0, col].set_title(f"{name} — no target sweep data")
                axes[r1, col].set_visible(False)

        # ── Control sweep (rows 2–3 of this pair, if present) ─────────
        if n_rows == 4:
            r2, r3 = row_offset + 2, row_offset + 3
            if ctrl_var in ds_raw.data_vars:
                data_ctrl = ds_raw[ctrl_var]
                s0c = data_ctrl.sel(control_state=0).values.astype(np.float64)
                s1c = data_ctrl.sel(control_state=1).values.astype(np.float64)
                diff_2d_c = s1c - s0c

                mad_key_c = f"mean_abs_diff_ctrl_{name}"
                mad_c = (
                    ds_fit[mad_key_c].values.astype(np.float64)
                    if ds_fit is not None and mad_key_c in ds_fit.data_vars
                    else np.mean(np.abs(diff_2d_c), axis=-1)
                )
                std_key_c = f"std_over_n_ctrl_{name}"
                std_n_c = (
                    ds_fit[std_key_c].values.astype(np.float64)
                    if ds_fit is not None and std_key_c in ds_fit.data_vars
                    else np.std(diff_2d_c, axis=0)
                )
                mean_key_c = f"mean_over_n_ctrl_{name}"
                mean_n_c = (
                    ds_fit[mean_key_c].values.astype(np.float64)
                    if ds_fit is not None and mean_key_c in ds_fit.data_vars
                    else np.mean(diff_2d_c, axis=0)
                )

                _plot_sweep_rows(
                    axes[r2], axes[r3], s0c, s1c, gate_counts, phases_rad,
                    mad_c, std_n_c, mean_n_c,
                    name=name, qubit_label="(control sweep)",
                    peak_gates=r.get("peak_num_gates_ctrl"),
                    success=r.get("success_control", False),
                    phase_comp_turns=r.get("phase_compensation_control"),
                    cond0_label="tgt |0⟩", cond1_label="tgt |1⟩",
                )

        row_offset += n_rows

    fig.suptitle(
        "Geometric CZ Phase Error Amplification",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    return fig