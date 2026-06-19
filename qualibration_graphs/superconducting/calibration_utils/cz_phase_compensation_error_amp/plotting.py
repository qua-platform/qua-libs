"""Plotting utilities for CZ phase compensation with error amplification."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_raw_data_with_fit(ds_raw: xr.Dataset, qubit_pairs, ds_fit: xr.Dataset = None):
    """Plot mean signal vs frame for control and target, highlighting the peak frame."""
    n_pairs = len(qubit_pairs)
    fig, axes = plt.subplots(2, n_pairs, figsize=(6 * n_pairs, 8), squeeze=False)

    for i, qp in enumerate(qubit_pairs):
        qp_name = qp.name

        if ds_fit is None or not bool(ds_fit.sel(qubit_pair=qp_name).success.values):
            for row in range(2):
                axes[row, i].text(
                    0.5, 0.5, "fit failed" if ds_fit else "no fit",
                    ha="center", va="center", transform=axes[row, i].transAxes,
                )
                axes[row, i].set_title(f"{qp_name} - {'control' if row == 0 else 'target'}")
                axes[row, i].set_xlabel("Frame rotation [2π]")
                axes[row, i].set_ylabel("Mean signal")
            continue

        qp_fit = ds_fit.sel(qubit_pair=qp_name)
        frames = ds_raw.sel(qubit_pair=qp_name).frame.values

        for row, (label, color) in enumerate([("control", "tab:blue"), ("target", "tab:red")]):
            ax = axes[row, i]
            mean_var = f"{label}_mean_vs_frame"
            phase_var = f"fitted_{label}_phase"
            peak_mean_var = f"{label}_mean_at_peak"

            if mean_var in qp_fit.data_vars:
                mean_curve = qp_fit[mean_var].values
                fitted_phase = float(qp_fit[phase_var].values)
                peak_mean = float(qp_fit[peak_mean_var].values)

                ax.plot(frames, mean_curve, "o-", color=color)
                ax.axvline(fitted_phase, color="k", ls="--", alpha=0.7, label=f"phase = {fitted_phase:.4f}")
                ax.plot(fitted_phase, peak_mean, "*", color="gold", ms=15, zorder=5,
                        label=f"peak mean = {peak_mean:.4f}")
                ax.legend(loc="best")

            ax.set_title(f"{qp_name} - {label}")
            ax.set_xlabel("Frame rotation [2π]")
            ax.set_ylabel("Mean signal")
            ax.grid(alpha=0.3)

    for j in range(n_pairs, axes.shape[1]):
        axes[0, j].axis("off")
        axes[1, j].axis("off")

    fig.suptitle("CZ phase compensation - error amplification (max-mean method)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
