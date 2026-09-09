from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calibration_utils.common_utils.plot_style import (
    apply_qubit_pair_outcome_style,
    qubit_pair_success,
)


def plot_all(
    ds_fit: xr.Dataset,
    qubit_pair_names: list[str],
    *,
    fit_results: Optional[Dict] = None,
    plot_fft: bool = False,
) -> dict[str, plt.Figure]:
    """Standard node plotting API returning a figure dict."""
    return {"summary_2d": plot_2d_summary(ds_fit, qubit_pair_names, fit_results=fit_results, plot_fft=plot_fft)}


def _compute_fft_2d(
    data_2d: np.ndarray,
    detunings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """FFT each row and return (spatial_frequencies_1_per_V, magnitudes).

    Returns
    -------
    freqs : (n_freq,) — positive frequencies in 1/V, excluding DC.
    fft_mag : (n_ramp, n_freq)
    """
    dV = float(detunings[1] - detunings[0]) if len(detunings) > 1 else 1.0
    freqs = np.fft.rfftfreq(len(detunings), d=dV)[1:]  # 1/V, no DC

    fft_mag = np.zeros((data_2d.shape[0], len(freqs)))
    for r_idx in range(data_2d.shape[0]):
        trace = data_2d[r_idx, :]
        spectrum = np.abs(np.fft.rfft(trace - trace.mean()))
        fft_mag[r_idx, :] = spectrum[1:]

    return freqs, fft_mag


def plot_2d_summary(
    ds_raw: xr.Dataset,
    qubit_pair_names: list[str],
    fit_results: Optional[Dict] = None,
    *,
    plot_fft: bool = False,
) -> plt.Figure:
    """Summary per qubit pair.

    Layout without FFT: 1 row × 3 columns per qubit pair:
        Avg state | Avg I | Avg Q

    Layout with FFT: 2 rows × 3 columns per qubit pair:
        Row 1 (heatmaps): Avg state | Avg I | Avg Q
        Row 2 (FFTs):     FFT(state) | FFT(I) | FFT(Q)

    Multiple qubit pairs are tiled as extra column groups.
    """
    n_pairs = max(len(qubit_pair_names), 1)
    n_rows = 2 if plot_fft else 1
    fig, axes = plt.subplots(
        n_rows,
        3 * n_pairs,
        figsize=(6 * 3 * n_pairs, 5 * n_rows),
        squeeze=False,
    )

    for p_idx, qp_name in enumerate(qubit_pair_names):
        success = qubit_pair_success(fit_results, qp_name)
        col_base = 3 * p_idx
        ax_state = axes[0, col_base]
        ax_i = axes[0, col_base + 1]
        ax_q = axes[0, col_base + 2]
        if plot_fft:
            ax_state_fft = axes[1, col_base]
            ax_i_fft = axes[1, col_base + 1]
            ax_q_fft = axes[1, col_base + 2]

        ramp = ds_raw["ramp_duration"].values
        detuning = ds_raw["detuning"].values

        # ── Avg state heatmap ──────────────────────────────────────────
        if "state" in ds_raw:
            state_2d = ds_raw.state.sel(qubit_pair=qp_name, drop=True).transpose("ramp_duration", "detuning").values
            im = ax_state.pcolormesh(
                detuning,
                ramp,
                state_2d,
                shading="nearest",
                cmap="RdBu_r",
                vmin=0,
                vmax=1,
            )
            fig.colorbar(im, ax=ax_state, label="Avg state")

            if fit_results and qp_name in fit_results:
                r = fit_results[qp_name]
                if r["success"]:
                    ax_state.plot(
                        r["optimal_detuning"],
                        r["optimal_ramp_duration"],
                        "k*",
                        markersize=18,
                        markeredgecolor="white",
                        markeredgewidth=1.0,
                        label=(f"opt ramp={r['optimal_ramp_duration']} ns, " f"detuning={r['optimal_detuning']:.4g} V"),
                    )
                    ax_state.legend(fontsize=7)

            if plot_fft:
                freqs, fft_mag = _compute_fft_2d(state_2d, detuning)
                im_fft = ax_state_fft.pcolormesh(
                    freqs,
                    ramp,
                    fft_mag,
                    shading="nearest",
                    cmap="inferno",
                )
                fig.colorbar(im_fft, ax=ax_state_fft, label="|FFT|")
                ax_state_fft.set_xlabel("Spatial frequency (1/V)")
                ax_state_fft.set_ylabel("Ramp duration (ns)")
        if plot_fft:
            apply_qubit_pair_outcome_style(ax_state_fft, qp_name, success, subtitle="FFT(state)")

        ax_state.set_xlabel("Detuning (V)")
        ax_state.set_ylabel("Ramp duration (ns)")
        apply_qubit_pair_outcome_style(ax_state, qp_name, success, subtitle="Average state")

        # ── Avg I heatmap + FFT(I) ─────────────────────────────────────
        if "I" in ds_raw:
            i_2d = ds_raw.I.sel(qubit_pair=qp_name, drop=True).transpose("ramp_duration", "detuning").values
            im_i = ax_i.pcolormesh(
                detuning,
                ramp,
                i_2d,
                shading="nearest",
                cmap="viridis",
            )
            fig.colorbar(im_i, ax=ax_i, label="Avg I")

            if plot_fft:
                freqs_i, fft_mag_i = _compute_fft_2d(i_2d, detuning)
                im_fft_i = ax_i_fft.pcolormesh(
                    freqs_i,
                    ramp,
                    fft_mag_i,
                    shading="nearest",
                    cmap="inferno",
                )
                fig.colorbar(im_fft_i, ax=ax_i_fft, label="|FFT|")
                ax_i_fft.set_xlabel("Spatial frequency (1/V)")
                ax_i_fft.set_ylabel("Ramp duration (ns)")
        if plot_fft:
            apply_qubit_pair_outcome_style(ax_i_fft, qp_name, success, subtitle="FFT(I)")

        ax_i.set_xlabel("Detuning (V)")
        ax_i.set_ylabel("Ramp duration (ns)")
        apply_qubit_pair_outcome_style(ax_i, qp_name, success, subtitle="Average I")

        # ── Avg Q heatmap + FFT(Q) ─────────────────────────────────────
        if "Q" in ds_raw:
            q_2d = ds_raw.Q.sel(qubit_pair=qp_name, drop=True).transpose("ramp_duration", "detuning").values
            im_q = ax_q.pcolormesh(
                detuning,
                ramp,
                q_2d,
                shading="nearest",
                cmap="viridis",
            )
            fig.colorbar(im_q, ax=ax_q, label="Avg Q")

            if plot_fft:
                freqs_q, fft_mag_q = _compute_fft_2d(q_2d, detuning)
                im_fft_q = ax_q_fft.pcolormesh(
                    freqs_q,
                    ramp,
                    fft_mag_q,
                    shading="nearest",
                    cmap="inferno",
                )
                fig.colorbar(im_fft_q, ax=ax_q_fft, label="|FFT|")
                ax_q_fft.set_xlabel("Spatial frequency (1/V)")
                ax_q_fft.set_ylabel("Ramp duration (ns)")
        if plot_fft:
            apply_qubit_pair_outcome_style(ax_q_fft, qp_name, success, subtitle="FFT(Q)")

        ax_q.set_xlabel("Detuning (V)")
        ax_q.set_ylabel("Ramp duration (ns)")
        apply_qubit_pair_outcome_style(ax_q, qp_name, success, subtitle="Average Q")

    fig.suptitle("Initialization ramp-duration vs detuning summary", fontsize=14)
    fig.tight_layout()
    return fig
