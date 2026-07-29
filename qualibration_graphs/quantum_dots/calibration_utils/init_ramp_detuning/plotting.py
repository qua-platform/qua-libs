from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_all(
    ds_fit: xr.Dataset,
    qubit_pair_names: list[str],
    *,
    fit_results: Optional[Dict] = None,
) -> dict[str, plt.Figure]:
    """Standard node plotting API returning a figure dict."""
    return {
        "summary_2d": plot_2d_summary(ds_fit, qubit_pair_names, fit_results=fit_results),
    }


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
) -> plt.Figure:
    """6-panel summary per qubit pair.

    Layout (2 rows × 3 columns per qubit pair):
        Row 1 (heatmaps): Avg state | Avg I | Avg Q
        Row 2 (FFTs):     FFT(state) | FFT(I) | FFT(Q)

    Multiple qubit pairs are tiled as extra column groups.
    """
    n_pairs = max(len(qubit_pair_names), 1)
    fig, axes = plt.subplots(
        2,
        3 * n_pairs,
        figsize=(6 * 3 * n_pairs, 5 * 2),
        squeeze=False,
    )

    for p_idx, qp_name in enumerate(qubit_pair_names):
        col_base = 3 * p_idx
        ax_state = axes[0, col_base]
        ax_i = axes[0, col_base + 1]
        ax_q = axes[0, col_base + 2]

        ax_state_fft = axes[1, col_base]
        ax_i_fft = axes[1, col_base + 1]
        ax_q_fft = axes[1, col_base + 2]

        ramp = ds_raw["ramp_duration"].values
        detuning = ds_raw["detuning"].values

        # ── Avg state heatmap ──────────────────────────────────────────
        state_key = f"state_{qp_name}"
        if state_key in ds_raw:
            state_2d = ds_raw[state_key].values
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

            # ── FFT of state ───────────────────────────────────────────
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
            ax_state_fft.set_title(f"{qp_name} — FFT(state)")
        else:
            ax_state.set_title(f"{qp_name} (no state data)")
            ax_state_fft.set_title(f"{qp_name} (no state data)")

        ax_state.set_xlabel("Detuning (V)")
        ax_state.set_ylabel("Ramp duration (ns)")
        ax_state.set_title(f"{qp_name} — Avg state")

        # ── Avg I heatmap + FFT(I) ─────────────────────────────────────
        i_key = f"I_{qp_name}"
        if i_key in ds_raw:
            i_2d = ds_raw[i_key].values
            im_i = ax_i.pcolormesh(
                detuning,
                ramp,
                i_2d,
                shading="nearest",
                cmap="viridis",
            )
            fig.colorbar(im_i, ax=ax_i, label="Avg I")

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
            ax_i_fft.set_title(f"{qp_name} — FFT(I)")
        else:
            ax_i.set_title(f"{qp_name} (no I data)")
            ax_i_fft.set_title(f"{qp_name} (no I data)")

        ax_i.set_xlabel("Detuning (V)")
        ax_i.set_ylabel("Ramp duration (ns)")
        ax_i.set_title(f"{qp_name} — Avg I")

        # ── Avg Q heatmap + FFT(Q) ─────────────────────────────────────
        q_key = f"Q_{qp_name}"
        if q_key in ds_raw:
            q_2d = ds_raw[q_key].values
            im_q = ax_q.pcolormesh(
                detuning,
                ramp,
                q_2d,
                shading="nearest",
                cmap="viridis",
            )
            fig.colorbar(im_q, ax=ax_q, label="Avg Q")

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
            ax_q_fft.set_title(f"{qp_name} — FFT(Q)")
        else:
            ax_q.set_title(f"{qp_name} (no Q data)")
            ax_q_fft.set_title(f"{qp_name} (no Q data)")

        ax_q.set_xlabel("Detuning (V)")
        ax_q.set_ylabel("Ramp duration (ns)")
        ax_q.set_title(f"{qp_name} — Avg Q")

    fig.suptitle("Init 2D calibration summary", fontsize=14)
    fig.tight_layout()
    return fig
