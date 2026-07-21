from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _compute_fft_2d(
    data_2d: np.ndarray,
    detunings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """FFT each row (wait-duration slice) and return (frequencies_MHz, magnitudes).

    Returns
    -------
    freqs : (n_freq,) — positive frequencies in MHz, excluding DC.
    fft_mag : (n_ramp, n_freq)
    """
    dt_ns = float(detunings[1] - detunings[0]) if len(detunings) > 1 else 1.0
    dt_us = dt_ns * 1e-3
    freqs = np.fft.rfftfreq(len(detunings), d=dt_us)[1:]  # MHz, no DC

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
    """4-panel summary per qubit pair.

    Layout (2 rows × 2 columns per qubit pair):
        (1) Avg state vs (wait, ramp)        | (2) FFT of state along wait axis
        (3) Avg I vs (wait, ramp)            | (4) FFT of I along wait axis

    Multiple qubit pairs are tiled as extra column groups.
    """
    n_pairs = max(len(qubit_pair_names), 1)
    fig, axes = plt.subplots(
        2, 2 * n_pairs,
        figsize=(6 * 2 * n_pairs, 5 * 2),
        squeeze=False,
    )

    for p_idx, qp_name in enumerate(qubit_pair_names):
        col_base = 2 * p_idx
        ax_state = axes[0, col_base]
        ax_state_fft = axes[0, col_base + 1]
        ax_iq = axes[1, col_base]
        ax_iq_fft = axes[1, col_base + 1]

        ramp = ds_raw["ramp_duration"].values
        detuning = ds_raw["detuning"].values

        # ── (1) Avg state heatmap ──────────────────────────────────────
        state_key = f"state_{qp_name}"
        if state_key in ds_raw:
            state_2d = ds_raw[state_key].values
            im = ax_state.pcolormesh(
                detuning, ramp, state_2d,
                shading="nearest", cmap="RdBu_r", vmin=0, vmax=1,
            )
            fig.colorbar(im, ax=ax_state, label="Avg state")

            if fit_results and qp_name in fit_results:
                r = fit_results[qp_name]
                if r["success"]:
                    ax_state.plot(
                        r["optimal_detuning"],
                        r["optimal_ramp_duration"],
                        "k*", markersize=18,
                        markeredgecolor="white", markeredgewidth=1.0,
                        label=(
                            f"opt ramp={r['optimal_ramp_duration']} ns, "
                            f"detunig={r['optimal_detuning']} ns"
                        ),
                    )
                    ax_state.legend(fontsize=7)

            # ── (2) FFT of state ───────────────────────────────────────
            freqs, fft_mag = _compute_fft_2d(state_2d, detuning)
            im_fft = ax_state_fft.pcolormesh(
                freqs, ramp, fft_mag,
                shading="nearest", cmap="inferno",
            )
            fig.colorbar(im_fft, ax=ax_state_fft, label="|FFT|")
            ax_state_fft.set_xlabel("Frequency (MHz)")
            ax_state_fft.set_ylabel("Ramp duration (ns)")
            ax_state_fft.set_title(f"{qp_name} — FFT(state)")
        else:
            ax_state.set_title(f"{qp_name} (no state data)")
            ax_state_fft.set_title(f"{qp_name} (no state data)")

        ax_state.set_xlabel("Detuning duration (ns)")
        ax_state.set_ylabel("Ramp duration (ns)")
        ax_state.set_title(f"{qp_name} — Avg state")

        # ── (3) Avg I heatmap ──────────────────────────────────────────
        i_key = f"I_{qp_name}"
        if i_key in ds_raw:
            i_2d = ds_raw[i_key].values
            im_i = ax_iq.pcolormesh(
                detuning, ramp, i_2d,
                shading="nearest", cmap="viridis",
            )
            fig.colorbar(im_i, ax=ax_iq, label="Avg I")

            # ── (4) FFT of I ───────────────────────────────────────────
            freqs_i, fft_mag_i = _compute_fft_2d(i_2d, detuning)
            im_fft_i = ax_iq_fft.pcolormesh(
                freqs_i, ramp, fft_mag_i,
                shading="nearest", cmap="inferno",
            )
            fig.colorbar(im_fft_i, ax=ax_iq_fft, label="|FFT|")
            ax_iq_fft.set_xlabel("Frequency (MHz)")
            ax_iq_fft.set_ylabel("Ramp duration (ns)")
            ax_iq_fft.set_title(f"{qp_name} — FFT(I)")
        else:
            ax_iq.set_title(f"{qp_name} (no I data)")
            ax_iq_fft.set_title(f"{qp_name} (no I data)")

        ax_iq.set_xlabel("Detuning")
        ax_iq.set_ylabel("Ramp duration (ns)")
        ax_iq.set_title(f"{qp_name} — Avg I")

    fig.suptitle("Init 2D calibration summary", fontsize=14)
    fig.tight_layout()
    return fig
