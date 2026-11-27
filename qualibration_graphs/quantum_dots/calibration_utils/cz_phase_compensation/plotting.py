"""CZ phase compensation plotting.

One figure per qubit pair with two rows (one subplot per row):
    Row 0: Target in superposition — partner control |0⟩ and |1⟩ on the same axes
    Row 1: Control in superposition — partner target |0⟩ and |1⟩ on the same axes

Each row overlays raw data and fits for both partner states, and draws a vertical
line at the fitted phase correction (same horizontal units as the frame sweep:
fraction of 2π).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

_EXP_LABELS = ["target_ctrl0", "target_ctrl1", "control_tgt0", "control_tgt1"]

# (exp_type_ground, exp_type_excited) for each row
_ROW_EXPERIMENTS = [
    (0, 1),  # target under test: control |0⟩ vs |1⟩
    (2, 3),  # control under test: target |0⟩ vs |1⟩
]

_ROW_TITLES = [
    "Target qubit — control |0⟩ vs |1⟩",
    "Control qubit — target |0⟩ vs |1⟩",
]

_PARTNER_LABELS = [
    ("Control |0⟩", "Control |1⟩"),
    ("Target |0⟩", "Target |1⟩"),
]


def plot_raw_data_with_fit(
    ds_raw: xr.Dataset,
    ds_fit: xr.Dataset | None,
    qubit_pairs: list[Any],
    fit_results: dict[str, dict[str, Any]],
    *,
    analysis_signal: str = "E_p2_given_p1_0",
) -> plt.Figure:
    """Plot raw data and fitted sinusoids for all qubit pairs.

    Each pair uses two stacked subplots; partner |0⟩ and |1⟩ curves share an axis.
    A vertical line marks the phase correction applied from the ground-state trace.
    """
    n_pairs = len(qubit_pairs)
    fig, axes = plt.subplots(
        2 * n_pairs,
        1,
        figsize=(8, 4.2 * n_pairs),
        squeeze=False,
    )

    # Same hue for data + fit per partner state (|0⟩ blue, |1⟩ orange)
    color_partner_0 = "#1f77b4"
    color_partner_1 = "#ff7f0e"

    for pair_idx, qp in enumerate(qubit_pairs):
        var_name = f"{analysis_signal}_{qp.name}"
        frames = ds_raw.coords["frame"].values
        has_data = var_name in ds_raw.data_vars
        fr = fit_results.get(qp.name, {})
        row_base = pair_idx * 2

        tgt_corr = float(fr.get("target_phase_correction", 0.0))
        ctrl_corr = float(fr.get("control_phase_correction", 0.0))
        cond_tgt = float(fr.get("conditional_phase_target", 0.0))
        cond_ctrl = float(fr.get("conditional_phase_control", 0.0))
        corrections = (tgt_corr, ctrl_corr)

        for row_in_pair, ((e0, e1), title, (lab0, lab1), corr_2pi) in enumerate(
            zip(
                _ROW_EXPERIMENTS,
                _ROW_TITLES,
                _PARTNER_LABELS,
                corrections,
                strict=True,
            )
        ):
            ax = axes[row_base + row_in_pair, 0]
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Frame rotation (× 2π)")
            ax.set_ylabel(analysis_signal)
            ax.set_xlim(float(frames.min()), float(frames.max()))

            if not has_data:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                continue

            data = ds_raw[var_name].values.astype(np.float64)

            for exp_type, lab, color in (
                (e0, lab0, color_partner_0),
                (e1, lab1, color_partner_1),
            ):
                label = _EXP_LABELS[exp_type]
                ax.plot(
                    frames,
                    data[exp_type],
                    "o",
                    ms=3,
                    alpha=0.75,
                    color=color,
                    label=f"{lab} data",
                )

                fit_key = f"fitted_{label}_{qp.name}"
                if ds_fit is not None and fit_key in ds_fit.data_vars:
                    fitted = ds_fit[fit_key].values
                    ax.plot(
                        frames,
                        fitted,
                        "-",
                        lw=1.6,
                        color=color,
                        label=f"{lab} fit",
                    )

            ax.axvline(
                corr_2pi,
                color="seagreen",
                ls="--",
                lw=1.8,
                alpha=0.9,
                label=f"phase corr. = {corr_2pi:.4f} × 2π",
            )

            cond = cond_tgt if row_in_pair == 0 else cond_ctrl
            ann = f"Conditional phase (unwrap): {cond:.3f} rad\n" f"(ideal CZ ≈ π)"
            ax.text(
                0.98,
                0.98,
                ann,
                transform=ax.transAxes,
                va="top",
                ha="right",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.35", fc="wheat", alpha=0.85),
            )

            ax.legend(fontsize=7, loc="lower right", ncol=2)

        status = "OK" if fr.get("success") else "FAILED"
        suptitle = (
            f"{qp.name} [{status}]  —  "
            f"ctrl correction: {ctrl_corr:.4f}×2π, "
            f"tgt correction: {tgt_corr:.4f}×2π"
        )
        if n_pairs == 1:
            fig.suptitle(suptitle, fontsize=11, y=1.02)
        else:
            axes[row_base, 0].annotate(
                suptitle,
                xy=(0.5, 1.12),
                xycoords="axes fraction",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

    fig.tight_layout()
    return fig
